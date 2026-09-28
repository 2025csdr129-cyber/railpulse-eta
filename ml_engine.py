"""Synthetic corridor training and explainable ETA inference."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

FEATURES = [
    "distance_remaining_km", "scheduled_eta_min", "speed_kmph", "headway_km",
    "signal_aspect", "active_tsr_kmph", "block_density", "gps_quality", "dwell_minutes",
]


class ETAEngine:
    def __init__(self, base_dir: Path) -> None:
        self.model_path = base_dir / "railpulse_model.pkl"
        self.backend = "sklearn-gradient-boosting"
        self.model = self._load_or_train()

    def _dataset(self) -> tuple[pd.DataFrame, np.ndarray]:
        rng = np.random.default_rng(42)
        n = 2400
        distance = rng.uniform(3, 420, n)
        scheduled = distance / rng.uniform(85, 125, n) * 60
        speed = rng.uniform(20, 130, n)
        headway = rng.uniform(0.6, 22, n)
        aspect = rng.choice([0, 1, 2, 3], n, p=[.08, .15, .25, .52])
        tsr = rng.choice([0, 30, 50, 75], n, p=[.65, .12, .12, .11])
        density = rng.integers(1, 6, n)
        gps = rng.uniform(.25, 1, n)
        dwell = rng.uniform(0, 8, n)
        delay = ((3 - aspect) * 2.5 + np.maximum(0, 3.5 - headway) * 2 + density * 1.4 +
                 (tsr > 0) * np.maximum(0, distance / np.maximum(tsr, 1) * 60 - scheduled) * .35 +
                 (1 - gps) * 4 + dwell + rng.normal(0, 1.4, n))
        y = np.maximum(0, scheduled + delay)
        return pd.DataFrame(dict(zip(FEATURES, [distance, scheduled, speed, headway, aspect, tsr, density, gps, dwell]))), y

    def _load_or_train(self) -> Any:
        if self.model_path.exists():
            try:
                with self.model_path.open("rb") as handle:
                    return pickle.load(handle)
            except (OSError, pickle.PickleError, ModuleNotFoundError, AttributeError, ImportError):
                pass
        x, y = self._dataset()
        model: Any
        try:
            from xgboost import XGBRegressor
            model = XGBRegressor(n_estimators=120, max_depth=4, learning_rate=.07, objective="reg:squarederror", n_jobs=1)
            self.backend = "xgboost"
        except (ImportError, OSError):
            model = GradientBoostingRegressor(n_estimators=140, max_depth=3, learning_rate=.06, random_state=42)
        model.fit(x[FEATURES], y)
        try:
            with self.model_path.open("wb") as handle:
                pickle.dump(model, handle)
        except OSError:
            pass
        return model

    def predict(self, request: Any) -> dict[str, Any]:
        values = request.model_dump()
        row = pd.DataFrame([{key: values[key] for key in FEATURES}])
        model_eta = float(max(0, self.model.predict(row[FEATURES])[0]))
        gps_margin = (1 - request.gps_quality) * 12
        congestion_margin = max(2, (3 - request.signal_aspect) * 2 + max(0, 3 - request.headway_km) * 1.5)
        low = max(0, round(model_eta - congestion_margin - gps_margin, 1))
        high = round(model_eta + congestion_margin + gps_margin, 1)
        confidence = round(max(.35, min(.98, request.gps_quality * .72 + request.headway_km / 100 - request.block_density * .025)), 2)
        causes: list[str] = []
        if request.signal_aspect == 0: causes.append("red signal hold")
        elif request.signal_aspect < 3: causes.append("restrictive signal aspect")
        if request.headway_km < 3: causes.append("close preceding traffic")
        if request.active_tsr_kmph: causes.append(f"temporary speed restriction ({request.active_tsr_kmph:g} km/h)")
        if request.dwell_minutes: causes.append("station dwell")
        if request.gps_quality < .6: causes.append("GPS outage; dead reckoning uncertainty")
        return {
            "predicted_delay_min": round(max(0, model_eta - request.scheduled_eta_min), 1),
            "honest_eta_range": {"min": low, "max": high},
            "confidence": confidence,
            "delay_root_cause": causes[0] if causes else "normal corridor variation",
            "delay_factors": causes,
            "model_backend": self.backend,
        }
