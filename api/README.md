# Flight Delay Risk API

A deployable FastAPI service around the production gradient-boosting model
from the flight-delay-project. Predicts the probability that a flight's
departure is delayed more than 15 minutes.

## What this model is (and isn't)

The full research project (see `../README.md`) found that the single
strongest predictor of delay is a **live cascade signal** — how backed up
the origin airport is *right now*, in the hours immediately before
departure. That requires a live operations feed this API doesn't have.

This deployable model instead uses **historical baselines**: typical delay
rate for this airport/hour/month, this carrier, and this specific route,
plus schedule and (optionally) weather. It's honest, self-contained, and
scores in milliseconds — but it will not catch a delay that's cascading
today. Every response includes a `caveat` field saying so, and
`feature_sources` tells you exactly which inputs were estimated vs. supplied.

**Honest holdout performance** (trained on Jan–Sep 2013, tested on Oct–Dec
2013, zero leakage — see the methodology note in `build_production_model.py`
for a leakage bug that was found and fixed during development):

| Metric | Value | vs. base rate |
|---|---|---|
| ROC-AUC | 0.652 | — |
| PR-AUC | 0.316 | 1.66x the 19% base rate |

If you later wire in a live feed (e.g. FlightAware, ADS-B, or your own ops
data), swap in the research model's network-state features
(`src/train_models.py`, ROC-AUC 0.767 / PR-AUC 0.513) for a meaningfully
stronger model — the architecture here (optional request fields with
historical fallback) was designed to make that swap easy.

## Run locally

```bash
cd api
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/docs for interactive Swagger docs.

## Run with Docker

```bash
cd api
docker build -t flight-delay-api .
docker run -p 8000:8000 flight-delay-api
```

## Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "carrier": "EV",
        "origin": "EWR",
        "dest": "ORD",
        "scheduled_departure": "2026-12-20T18:30:00"
      }'
```

Only `carrier`, `origin`, `dest`, and `scheduled_departure` are required.
Everything else (`distance`, `temp`, `precip`, `wind_speed`, `visib`, etc.)
is optional — omit it and the API estimates it from historical data, or
supply your own values (e.g. from a weather forecast) for a sharper
estimate:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "carrier": "DL",
        "origin": "JFK",
        "dest": "LAX",
        "scheduled_departure": "2026-07-04T08:00:00",
        "precip": 0.3,
        "wind_speed": 30,
        "visib": 2
      }'
```

### Response

```json
{
  "probability_delayed": 0.4794,
  "risk_level": "High",
  "baseline_rate": 0.2154,
  "lift_vs_baseline": 2.23,
  "feature_sources": {
    "distance_source": "historical route average",
    "weather_source": "historical climatology for this airport/hour/month",
    "route_history_source": "historical route average (n=5851)"
  },
  "caveat": "This estimate uses historical schedule, route, and carrier baselines ..."
}
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Service info |
| GET | `/health` | Liveness + model metadata |
| GET | `/metadata` | Full training metadata, known carriers/origins |
| POST | `/predict` | Score a single flight |
| GET | `/docs` | Interactive Swagger UI |

## Known limitations

- **Training data scope**: NYC departures only (EWR/JFK/LGA), 2013. Origin
  airports outside that set fall back to a global baseline and a
  great-circle distance estimate — predictions for those will be weaker.
- **No live signal**: see above. This is a baseline-risk score, not a
  real-time prediction.
- **Carrier codes change over time**: some 2013 carriers (e.g. US Airways,
  AirTran) have since merged or ceased operating. An unrecognized carrier
  code falls back to the global average and is flagged in the response.
- **Weather defaults are climatology, not forecasts**: if you have access to
  an actual weather forecast for the departure time, pass it in — it will
  materially improve the estimate (see the weather-vs-baseline test above).

## Project structure

```
api/
├── app/
│   ├── main.py       FastAPI routes
│   ├── features.py   Request -> model feature vector, with fallback logic
│   └── schemas.py     Pydantic request/response models
├── artifacts/
│   ├── model.pkl      Trained sklearn Pipeline (preprocessing + HGB classifier)
│   ├── lookups.pkl     Historical baseline tables (route/carrier/origin-hour-month)
│   ├── metadata.json   Feature list, holdout metrics, known carriers/origins
│   └── airports.csv    Airport lat/lon, for distance fallback
├── requirements.txt
└── Dockerfile
```
