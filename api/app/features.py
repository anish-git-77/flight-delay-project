"""
Turns a PredictRequest into the exact feature row the model was trained on,
filling in anything the caller didn't supply from historical lookup tables.

This module is the single place where "what does the model actually need"
lives, so the API route handlers stay thin.
"""
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ARTIFACTS = Path(__file__).parent.parent / "artifacts"

_model = None
_lookups = None
_meta = None
_airports = None


def load_artifacts():
    """Load once at process startup; cheap to call repeatedly after that."""
    global _model, _lookups, _meta, _airports
    if _model is None:
        _model = joblib.load(ARTIFACTS / "model.pkl")
        _lookups = joblib.load(ARTIFACTS / "lookups.pkl")
        _meta = json.loads((ARTIFACTS / "metadata.json").read_text())
        _airports = pd.read_csv(ARTIFACTS / "airports.csv")
    return _model, _lookups, _meta, _airports


def _haversine_miles(lat1, lon1, lat2, lon2):
    r = 3958.8  # Earth radius, miles
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# Near-holiday windows, mirroring the ones used to train the model
# (extended year over year rather than hard-coded to 2013 only).
_FIXED_HOLIDAYS_MMDD = [
    (1, 1),    # New Year's Day
    (7, 4),    # Independence Day
    (11, 11),  # Veterans Day
    (12, 24), (12, 25), (12, 31),  # Christmas Eve/Day, New Year's Eve
]


def _near_holiday(dt: pd.Timestamp) -> int:
    day = dt.normalize()
    for m, d in _FIXED_HOLIDAYS_MMDD:
        try:
            h = pd.Timestamp(year=day.year, month=m, day=d)
        except ValueError:
            continue
        if abs((day - h).days) <= 2:
            return 1
    # US Thanksgiving: 4th Thursday of November, +/- 2 days.
    nov1 = pd.Timestamp(year=day.year, month=11, day=1)
    first_thursday = nov1 + pd.Timedelta(days=(3 - nov1.dayofweek) % 7)
    thanksgiving = first_thursday + pd.Timedelta(weeks=3)
    if abs((day - thanksgiving).days) <= 2:
        return 1
    return 0


def build_feature_row(req) -> tuple[pd.DataFrame, dict]:
    """
    Returns (single-row DataFrame in model feature order, sources dict
    describing which values were user-supplied vs. estimated).
    """
    model, lookups, meta, airports = load_artifacts()
    sources = {}

    dt = pd.Timestamp(req.scheduled_departure)
    month, hour = dt.month, dt.hour
    day_of_week = dt.dayofweek
    day_of_year = dt.dayofyear
    is_weekend = int(day_of_week >= 5)
    near_holiday = _near_holiday(dt)
    hour_sin, hour_cos = math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24)
    month_sin, month_cos = math.sin(2 * math.pi * month / 12), math.cos(2 * math.pi * month / 12)

    origin, dest, carrier = req.origin, req.dest, req.carrier

    # ---- distance & scheduled block time -------------------------------
    route_tbl = lookups["route"]
    route_row = route_tbl[(route_tbl["origin"] == origin) & (route_tbl["dest"] == dest)]

    if req.distance is not None:
        distance = req.distance
        sources["distance_source"] = "user-provided"
    elif len(route_row):
        distance = float(route_row["distance"].iloc[0])
        sources["distance_source"] = "historical route average"
    else:
        o = airports[airports["faa"] == origin]
        d = airports[airports["faa"] == dest]
        if len(o) and len(d):
            distance = _haversine_miles(
                o["lat"].iloc[0], o["lon"].iloc[0], d["lat"].iloc[0], d["lon"].iloc[0]
            )
            sources["distance_source"] = "great-circle estimate (airport unseen in training routes)"
        else:
            distance = float(lookups["global_baseline"]["distance"])
            sources["distance_source"] = "global fallback (airport not recognized)"

    if len(route_row):
        sched_block_min = float(route_row["sched_block_min"].iloc[0])
    else:
        # ~7.2 min per 100 miles cruise-equivalent + fixed taxi/climb overhead,
        # roughly matching the training data's median block-time/distance ratio.
        sched_block_min = 30 + distance * 0.11

    # ---- carrier & route baseline delay rates ---------------------------
    carrier_tbl = lookups["carrier"]
    c_row = carrier_tbl[carrier_tbl["carrier"] == carrier]
    carrier_loo = float(c_row["pct_delayed"].iloc[0]) if len(c_row) else float(lookups["global_baseline"]["pct_delayed"])
    if not len(c_row):
        sources["carrier_note"] = f"carrier '{carrier}' not seen in training data; used global baseline"

    origin_wide_tbl = lookups["origin_wide"]
    ow_row = origin_wide_tbl[origin_wide_tbl["origin"] == origin]
    origin_wide_rate = float(ow_row["pct_delayed"].iloc[0]) if len(ow_row) else float(lookups["global_baseline"]["pct_delayed"])

    route_n = int(route_row["n"].iloc[0]) if len(route_row) else 0
    if route_n >= meta["min_route_n"]:
        route_loo = float(route_row["pct_delayed"].iloc[0])
        sources["route_history_source"] = f"historical route average (n={route_n})"
    else:
        route_loo = origin_wide_rate
        sources["route_history_source"] = "origin-airport average (route history too thin or unseen)"

    # ---- origin x hour x month baseline, and weather climatology --------
    ohm_tbl = lookups["origin_hour_month"]
    ohm_row = ohm_tbl[(ohm_tbl["origin"] == origin) & (ohm_tbl["hour"] == hour) & (ohm_tbl["month"] == month)]

    if len(ohm_row):
        origin_hour_month_loo = float(ohm_row["pct_delayed"].iloc[0])
        departures_this_hour = float(ohm_row["departures_this_hour"].iloc[0])
        climatology = ohm_row.iloc[0]
    else:
        origin_hour_month_loo = origin_wide_rate
        departures_this_hour = float(lookups["origin_hour_month"]["departures_this_hour"].mean())
        climatology = None

    weather_fields = ["temp", "dewp", "humid", "wind_dir", "wind_speed", "wind_gust", "precip", "pressure", "visib"]
    weather = {}
    any_estimated = False
    for f in weather_fields:
        user_val = getattr(req, f)
        if user_val is not None:
            weather[f] = user_val
        elif climatology is not None and pd.notna(climatology.get(f, np.nan)):
            weather[f] = float(climatology[f])
            any_estimated = True
        else:
            weather[f] = 0.0
            any_estimated = True
    sources["weather_source"] = "user-provided" if not any_estimated else (
        "historical climatology for this airport/hour/month" if climatology is not None
        else "global fallback (no climatology for this airport/hour/month)"
    )
    has_gust = int((req.wind_gust or weather["wind_gust"] or 0) > 0)

    row = {
        "distance": distance,
        "sched_block_min": sched_block_min,
        "day_of_week": day_of_week,
        "day_of_year": day_of_year,
        "hour_sin": hour_sin, "hour_cos": hour_cos,
        "month_sin": month_sin, "month_cos": month_cos,
        "departures_this_hour": departures_this_hour,
        "origin_hour_month_loo": origin_hour_month_loo,
        "carrier_loo": carrier_loo,
        "route_loo": route_loo,
        **weather,
        "is_weekend": is_weekend,
        "near_holiday": near_holiday,
        "has_gust": has_gust,
        "carrier": carrier,
        "origin": origin,
        "dest": dest,
        "hour": hour,
    }

    feats = meta["feature_order"]
    df = pd.DataFrame([row])[feats]
    return df, sources


def predict(req):
    model, lookups, meta, _ = load_artifacts()
    X, sources = build_feature_row(req)
    proba = float(model.predict_proba(X)[0, 1])
    baseline = float(lookups["global_baseline"]["pct_delayed"])

    if proba < 0.15:
        risk = "Low"
    elif proba < 0.30:
        risk = "Moderate"
    elif proba < 0.50:
        risk = "High"
    else:
        risk = "Severe"

    return {
        "probability_delayed": round(proba, 4),
        "risk_level": risk,
        "baseline_rate": round(baseline, 4),
        "lift_vs_baseline": round(proba / baseline, 2) if baseline else 0.0,
        "feature_sources": {
            "distance_source": sources.get("distance_source", "unknown"),
            "weather_source": sources.get("weather_source", "unknown"),
            "route_history_source": sources.get("route_history_source", "unknown"),
        },
    }
