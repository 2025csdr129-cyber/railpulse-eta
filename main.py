import asyncio
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

# In-memory shared state for the corridor simulation
simulation_state = {
    "progress": 0.05,
    "speed_kmh": 124.0,
    "target_speed_kmh": 124.0,
    "signal_aspect": "GREEN",
    "delay_min": 0,
    "has_freight": False,
    "is_freight_diverted": False,
    "is_station_halted": False,
    "dwell_remaining_sec": 30.0,
    "has_departed_station": False,
    "has_signal_halt": False,
    "has_fog": False
}

active_connections = set()

# Background tick loop streaming live calculations (20 updates/sec)
async def telemetry_broadcast_loop():
    while True:
        await asyncio.sleep(0.05)
        if not active_connections:
            continue

        s = simulation_state

        if s["is_station_halted"]:
            s["target_speed_kmh"] = 0.0
            s["signal_aspect"] = "RED"
        elif s["has_signal_halt"]:
            s["target_speed_kmh"] = 0.0
            s["signal_aspect"] = "RED"
            s["delay_min"] = 26
        elif s["has_freight"] and not s["is_freight_diverted"]:
            s["target_speed_kmh"] = 42.0
            s["signal_aspect"] = "DOUBLE_YELLOW"
            s["delay_min"] = 18
        elif s["has_freight"] and s["is_freight_diverted"]:
            s["target_speed_kmh"] = 124.0
            s["signal_aspect"] = "GREEN"
            s["delay_min"] = 2
        elif s["has_fog"]:
            s["target_speed_kmh"] = 60.0
            s["signal_aspect"] = "YELLOW"
            s["delay_min"] = 14
        else:
            s["target_speed_kmh"] = 124.0
            s["signal_aspect"] = "GREEN"
            s["delay_min"] = 0

        # Acceleration / Deceleration
        if s["speed_kmh"] < s["target_speed_kmh"]:
            s["speed_kmh"] = min(s["speed_kmh"] + 1.2, s["target_speed_kmh"])
        elif s["speed_kmh"] > s["target_speed_kmh"]:
            s["speed_kmh"] = max(s["speed_kmh"] - 1.8, s["target_speed_kmh"])

        step = (s["speed_kmh"] / 124.0) * 0.0012
        if not s["is_station_halted"] and not s["has_signal_halt"]:
            s["progress"] += step
            if s["progress"] >= 1.0:
                s["progress"] = 0.0
                s["has_departed_station"] = False

        if not s["is_station_halted"] and not s["has_departed_station"] and 0.485 <= s["progress"] <= 0.50:
            s["is_station_halted"] = True
            s["dwell_remaining_sec"] = 30.0

        if s["is_station_halted"]:
            s["dwell_remaining_sec"] = max(0.0, s["dwell_remaining_sec"] - 0.05)
            if s["dwell_remaining_sec"] <= 0:
                s["is_station_halted"] = False
                s["has_departed_station"] = True
                s["progress"] = 0.51
                s["speed_kmh"] = 20.0

        lat = 28.6189 + s["progress"] * 0.04
        lng = 77.2185 + s["progress"] * 0.04
        dist_to_gzb = max(0.0, (0.49 - s["progress"]) * 35)

        packet = {
            "progress": s["progress"],
            "speed_kmh": round(s["speed_kmh"], 1),
            "signal_aspect": s["signal_aspect"],
            "delay_min": s["delay_min"],
            "is_station_halted": s["is_station_halted"],
            "dwell_sec": int(s["dwell_remaining_sec"]),
            "has_freight": s["has_freight"],
            "is_freight_diverted": s["is_freight_diverted"],
            "has_signal_halt": s["has_signal_halt"],
            "has_fog": s["has_fog"],
            "gps_lat": round(lat, 4),
            "gps_lng": round(lng, 4),
            "dist_to_gzb": round(dist_to_gzb, 1)
        }

        serialized = json.dumps(packet)
        for client in list(active_connections):
            try:
                await client.send_text(serialized)
            except Exception:
                active_connections.remove(client)

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(telemetry_broadcast_loop())
    yield
    task.cancel()

app = FastAPI(title="RailPulse Digital Twin Engine", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

@app.get("/railpulse-3d", response_class=HTMLResponse)
async def serve_prototype(request: Request):
    return templates.TemplateResponse(request=request, name="railpulse_3d.html")

@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            action = payload.get("action")

            if action == "TOGGLE_FREIGHT":
                simulation_state["has_freight"] = not simulation_state["has_freight"]
                simulation_state["is_freight_diverted"] = False
            elif action == "DIVERT_FREIGHT":
                if simulation_state["has_freight"]:
                    simulation_state["is_freight_diverted"] = not simulation_state["is_freight_diverted"]
            elif action == "TOGGLE_SIGNAL":
                simulation_state["has_signal_halt"] = not simulation_state["has_signal_halt"]
            elif action == "TOGGLE_FOG":
                simulation_state["has_fog"] = not simulation_state["has_fog"]
            elif action == "FORCE_DEPARTURE":
                simulation_state["is_station_halted"] = False
                simulation_state["has_departed_station"] = True
                simulation_state["dwell_remaining_sec"] = 0.0
                simulation_state["progress"] = 0.51
            elif action == "FORCE_HALT":
                simulation_state["is_station_halted"] = True
                simulation_state["dwell_remaining_sec"] = 30.0
                simulation_state["progress"] = 0.49
    except WebSocketDisconnect:
        active_connections.remove(websocket)
