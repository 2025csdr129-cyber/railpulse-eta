# RailPulse Engine

RailPulse is a self-contained FastAPI demo for a New Delhi–Ghaziabad style rail corridor. It combines synthetic corridor telemetry, a startup-trained ETA regressor, honest uncertainty ranges, a 1D kinematic dead-reckoner, and shared controller/passenger views.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/` for the passenger HUD and `/controller` for the COA dashboard.

## API

- `POST /api/predict-eta` accepts distance, schedule, speed, headway, signal, TSR, block density, GPS quality, and dwell inputs.
- `GET /api/state` returns the current telemetry, prediction, and recent audit entries.
- `GET /api/audit` returns structured in-memory SQLite audit records.
- `POST /api/override` accepts `freight`, `overtake`, `signal_hold`, and `gps` controller actions.
- `WS /ws` broadcasts telemetry and accepts the same override JSON messages.

The model prefers XGBoost when available and falls back to scikit-learn GradientBoostingRegressor. `railpulse_model.pkl` is generated on first startup and is ignored by Git.
