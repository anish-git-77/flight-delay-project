"""Pydantic schemas for the flight delay prediction API."""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    carrier: str = Field(..., description="Two-letter IATA carrier code, e.g. 'DL', 'B6', 'AA'.")
    origin: str = Field(..., description="Origin airport FAA code. Model was trained on EWR, JFK, LGA.")
    dest: str = Field(..., description="Destination airport FAA code, e.g. 'LAX', 'ORD'.")
    scheduled_departure: datetime = Field(
        ..., description="Scheduled local departure time, ISO 8601, e.g. '2026-12-20T18:30:00'."
    )

    # Optional overrides - if omitted, the API estimates them from historical data.
    distance: Optional[float] = Field(None, description="Great-circle distance in miles. Estimated if omitted.")

    temp: Optional[float] = Field(None, description="Temperature, deg F. Estimated from climatology if omitted.")
    dewp: Optional[float] = Field(None, description="Dew point, deg F.")
    humid: Optional[float] = Field(None, description="Relative humidity, %.")
    wind_dir: Optional[float] = Field(None, description="Wind direction, degrees.")
    wind_speed: Optional[float] = Field(None, description="Wind speed, mph.")
    wind_gust: Optional[float] = Field(None, description="Wind gust speed, mph. Omit if none.")
    precip: Optional[float] = Field(None, description="Precipitation, inches in the hour.")
    pressure: Optional[float] = Field(None, description="Sea-level pressure, millibars.")
    visib: Optional[float] = Field(None, description="Visibility, miles.")

    @field_validator("carrier", "origin", "dest")
    @classmethod
    def upper(cls, v):
        return v.strip().upper()

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "carrier": "EV",
                "origin": "EWR",
                "dest": "ORD",
                "scheduled_departure": "2026-12-20T18:30:00",
            }]
        }
    }


class FeatureSourceInfo(BaseModel):
    distance_source: str
    weather_source: str
    route_history_source: str


class PredictResponse(BaseModel):
    probability_delayed: float = Field(..., description="Model estimate that departure delay exceeds 15 minutes.")
    risk_level: str = Field(..., description="Low / Moderate / High / Severe, from calibrated thresholds.")
    baseline_rate: float = Field(..., description="Overall historical delay rate, for comparison.")
    lift_vs_baseline: float = Field(..., description="probability_delayed divided by baseline_rate.")
    feature_sources: FeatureSourceInfo
    caveat: str = Field(
        default=(
            "This estimate uses historical schedule, route, and carrier baselines. "
            "It does not see today's live conditions (e.g. an aircraft or crew already "
            "running late right now) - the single strongest signal our research found "
            "requires a live operations feed this API does not have. Treat this as a "
            "baseline risk score, not a real-time prediction."
        )
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    trained_on: str
    holdout_roc_auc: float
    holdout_pr_auc: float
