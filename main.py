"""RailPulse Engine API and live corridor state."""
from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from kalman_filter import KinematicDeadReckoner
from ml_engine import ETAEngine

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"


class PredictionRequest(BaseModel):
    distance_remaining_km: float = Field(25.0, ge=0, le=1000)
    scheduled_eta_min: float = Field(18.0, ge=0, le=1440)
    speed_kmph: float = Field(92.0, ge=0, le=250)
    headway_km: float = Field(12.5, ge=0, le=100)
    signal_aspect: int = Field(3, ge=0, le=3)
    active_tsr_kmph: float = Field(0.0, ge=0, le=250)
    block_density: int = Field(1, ge=0, le=20)
    gps_quality: float = Field(0.98, ge=0, le=1)
    dwell_minutes: float = Field(0.0, ge=0, le=240)


class OverrideRequest(BaseModel):
    action: str
    value: Any = None


class AuditLog:
    def __init__(self) -> None:
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        self.db.execute(
            "CREATE TABLE audit (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, actor TEXT, action TEXT, payload TEXT)"
        )
        self.db.commit()

    def record(self, actor: str, action: str, payload: Any = None) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        encoded = json.dumps(payload, separators=(",", ":"), default=str)
        self.db.execute("INSERT INTO audit(ts,actor,action,payload) VALUES (?,?,?,?)", (now, actor, action, encoded))
        self.db.commit()
        row_id = self.db.execute("SELECT last_insert_rowid()").fetchone()[0]
        return {"id": row_id, "ts": now, "actor": actor, "action": action, "payload": payload}

    def recent(self, limit: int = 30) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT id,ts,actor,action,payload FROM audit ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [{"id": r[0], "ts": r[1], "actor": r[2], "action": r[3], "payload": json.loads(r[4])} for r in rows]


class ConnectionManager:
    def __init__(self) -> None:
        self.connections: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.connections.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self.connections.discard(websocket)

    async def broadcast(self, message: dict[str, Any]) -> None:
        stale: list[WebSocket] = []
        for connection in tuple(self.connections):
            try:
                await connection.send_json(message)
            except Exception:
                stale.append(connection)
        for connection in stale:
            self.disconnect(connection)


audit = AuditLog()
manager = ConnectionManager()
eta_engine = ETAEngine(BASE_DIR)
reckoner = KinematicDeadReckoner()
state: dict[str, Any] = {
    "train_id": "12424",
    "train_name": "Rajdhani Express",
    "route": "New Delhi → Ghaziabad",
    "speed_kmph": 92.0,
    "position_km": 17.4,
    "distance_remaining_km": 7.6,
    "signal_aspect": 3,
    "signal_label": "CLEAR",
    "gps_status": "LIVE",
    "headway_km": 12.5,
    "block_density": 1,
    "active_tsr_kmph": 0.0,
    "freight_active": False,
    "freight_diverted": False,
    "signal_hold": False,
    "updated_at": datetime.now(timezone.utc).isoformat(),
}


def snapshot() -> dict[str, Any]:
    prediction = eta_engine.predict(PredictionRequest(**{
        "distance_remaining_km": state["distance_remaining_km"],
        "scheduled_eta_min": 18.0,
        "speed_kmph": state["speed_kmph"],
        "headway_km": state["headway_km"],
        "signal_aspect": state["signal_aspect"],
        "active_tsr_kmph": state["active_tsr_kmph"],
        "block_density": state["block_density"],
        "gps_quality": 1.0 if state["gps_status"] == "LIVE" else 0.35,
        "dwell_minutes": 0,
    }))
    return {"type": "telemetry", "state": state, "prediction": prediction, "audit": audit.recent(12)}


async def telemetry_loop() -> None:
    while True:
        await asyncio.sleep(2)
        if not state["signal_hold"]:
            state["position_km"] = min(25.0, state["position_km"] + state["speed_kmph"] / 3600 * 2)
            state["distance_remaining_km"] = max(0.0, 25.0 - state["position_km"])
            estimate = reckoner.update(state["speed_kmph"], 2.0, gps_available=state["gps_status"] == "LIVE")
            state["speed_kmph"] = round(max(0.0, estimate.velocity_kmph), 1)
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        if manager.connections:
            await manager.broadcast(snapshot())


@asynccontextmanager
async def lifespan(_: FastAPI):
    task = asyncio.create_task(telemetry_loop())
    audit.record("system", "engine_started", {"model": eta_engine.backend})
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="RailPulse Engine", version="1.0.0", lifespan=lifespan)


@app.get("/")
@app.get("/passenger")
@app.get("/railpulse-3d")
async def passenger_view() -> FileResponse:
    return FileResponse(TEMPLATES_DIR / "railpulse_3d.html")


@app.get("/controller")
async def controller_view() -> FileResponse:
    return FileResponse(TEMPLATES_DIR / "controller_view.html")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "service": "railpulse-engine", "model_backend": eta_engine.backend}


@app.get("/api/state")
async def get_state() -> dict[str, Any]:
    return snapshot()


@app.get("/api/audit")
async def get_audit(limit: int = 30) -> list[dict[str, Any]]:
    return audit.recent(max(1, min(limit, 100)))


@app.post("/api/predict-eta")
async def predict_eta(request: PredictionRequest) -> dict[str, Any]:
    result = eta_engine.predict(request)
    audit.record("api", "eta_prediction", result)
    return result


@app.post("/api/override")
async def override(request: OverrideRequest) -> dict[str, Any]:
    action = request.action
    if action == "freight":
        state["freight_active"] = bool(request.value)
        if state["freight_active"]:
            state.update({"speed_kmph": 42.0, "signal_aspect": 2, "headway_km": 1.8, "block_density": 3})
        else:
            state.update({"speed_kmph": 92.0, "signal_aspect": 3, "headway_km": 12.5, "block_density": 1})
    elif action == "overtake":
        state["freight_diverted"] = bool(request.value)
        if state["freight_diverted"]:
            state.update({"speed_kmph": 92.0, "signal_aspect": 3, "headway_km": 10.0})
    elif action == "signal_hold":
        state["signal_hold"] = bool(request.value)
        state.update({"speed_kmph": 0.0 if state["signal_hold"] else 92.0, "signal_aspect": 0 if state["signal_hold"] else 3})
    elif action == "gps":
        state["gps_status"] = "LIVE" if bool(request.value) else "DEAD-RECKONING"
    else:
        return {"ok": False, "error": f"Unknown override: {action}"}
    entry = audit.record("controller", action, {"value": request.value})
    await manager.broadcast(snapshot())
    return {"ok": True, "audit": entry, "state": state}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    await websocket.send_json(snapshot())
    try:
        while True:
            message = await websocket.receive_json()
            request = OverrideRequest(**message)
            if request.action:
                await override(request)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)
