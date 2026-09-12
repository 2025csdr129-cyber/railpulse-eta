import os
import json
import pickle
from datetime import datetime, timedelta
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

app = FastAPI(title="RailPulse Multi-Train Engine")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load corridor and model
with open("corridor.json", "r", encoding="utf-8") as f:
    CORRIDOR = json.load(f)

with open("model.pkl", "rb") as f:
    MODEL = pickle.load(f)

STATIONS_GEO = {
    "NDLS": [28.6429, 77.2195],
    "GZB":  [28.6679, 77.4326],
    "ALJN": [27.8974, 78.0880],
    "CNB":  [26.4539, 80.3510]
}

# State for both trains
MULTI_TRAIN_STATE = {
    "express": {
        "train_number": "12424",
        "train_name": "Rajdhani Superfast",
        "type": "Express",
        "current_segment_idx": 1,
        "progress_pct": 30.0,
        "signal_aspect": 2,  # Double Yellow behind freight
        "active_tsr_kmph": 0
    },
    "freight": {
        "train_number": "BOXN-7042",
        "train_name": "Coal Freight Rake",
        "type": "Freight",
        "current_segment_idx": 1,
        "progress_pct": 45.0,  # Ahead on track
        "speed_kmph": 42.0,
        "on_loop_line": False  # Flag for loop line overtake
    },
    "block_density": 3
}

class MultiTelemetryUpdate(BaseModel):
    express_progress: float
    freight_progress: float
    freight_loop_line: bool
    signal_aspect: int
    active_tsr: int
    block_density: int

def interpolate_geo(seg_idx: int, pct: float):
    start = STATIONS_GEO[CORRIDOR[seg_idx]["from_station"]]
    end = STATIONS_GEO[CORRIDOR[seg_idx]["to_station"]]
    lat = start[0] + (end[0] - start[0]) * (pct / 100.0)
    lng = start[1] + (end[1] - start[1]) * (pct / 100.0)
    return [lat, lng]

def compute_multi_eta():
    now = datetime.now()
    exp = MULTI_TRAIN_STATE["express"]
    frt = MULTI_TRAIN_STATE["freight"]
    cur_idx = exp["current_segment_idx"]
    pct_left = (100.0 - exp["progress_pct"]) / 100.0

    seg_dist = CORRIDOR[cur_idx]["distance_km"]
    if frt["on_loop_line"]:
        headway = 18.0
        aspect = 3  # Green
    else:
        gap_pct = max(frt["progress_pct"] - exp["progress_pct"], 0.5)
        headway = (gap_pct / 100.0) * seg_dist
        aspect = 1 if headway < 4.0 else exp["signal_aspect"]

    cur_seg_features = pd.DataFrame([{
        'segment_idx': cur_idx,
        'nominal_time_min': CORRIDOR[cur_idx]['nominal_time_min'],
        'headway_km': headway,
        'signal_aspect': aspect,
        'active_tsr_kmph': exp["active_tsr_kmph"],
        'block_density': MULTI_TRAIN_STATE["block_density"]
    }])

    pred_full_cur_seg = float(MODEL.predict(cur_seg_features)[0])
    remaining_cur_seg = pred_full_cur_seg * pct_left

    subsequent_time = sum(
        float(MODEL.predict(pd.DataFrame([{
            'segment_idx': idx,
            'nominal_time_min': CORRIDOR[idx]['nominal_time_min'],
            'headway_km': 15.0,
            'signal_aspect': 3,
            'active_tsr_kmph': 0,
            'block_density': 1
        }]))[0])
        for idx in range(cur_idx + 1, len(CORRIDOR))
    )

    total_min = remaining_cur_seg + subsequent_time
    dynamic_eta = now + timedelta(minutes=total_min)

    static_remaining = (CORRIDOR[cur_idx]['nominal_time_min'] * pct_left) + sum(
        s['nominal_time_min'] for s in CORRIDOR[cur_idx + 1:]
    )
    static_eta = now + timedelta(minutes=static_remaining)

    exp_coords = interpolate_geo(cur_idx, exp["progress_pct"])
    frt_coords = interpolate_geo(cur_idx, frt["progress_pct"])
    if frt["on_loop_line"]:
        frt_coords = [frt_coords[0] + 0.04, frt_coords[1] + 0.04]

    return {
        "express_coords": exp_coords,
        "freight_coords": frt_coords,
        "stations_geo": STATIONS_GEO,
        "headway_km": round(headway, 1),
        "freight_status": "Diverted to Loop Line (Overtake Active)" if frt["on_loop_line"] else "Occupying Main Line Ahead",
        "signal_aspect_label": ["RED", "YELLOW", "DOUBLE YELLOW", "GREEN"][aspect],
        "static_eta": static_eta.strftime("%H:%M"),
        "dynamic_eta": dynamic_eta.strftime("%H:%M"),
        "delay_min": round(total_min - static_remaining, 1),
        "confidence": "96%" if frt["on_loop_line"] else "83%"
    }

@app.get("/api/eta")
def get_eta():
    return compute_multi_eta()

@app.post("/api/update-telemetry")
def update_telemetry(payload: MultiTelemetryUpdate):
    MULTI_TRAIN_STATE["express"]["progress_pct"] = payload.express_progress
    MULTI_TRAIN_STATE["freight"]["progress_pct"] = payload.freight_progress
    MULTI_TRAIN_STATE["freight"]["on_loop_line"] = payload.freight_loop_line
    MULTI_TRAIN_STATE["express"]["signal_aspect"] = payload.signal_aspect
    MULTI_TRAIN_STATE["express"]["active_tsr_kmph"] = payload.active_tsr
    MULTI_TRAIN_STATE["block_density"] = payload.block_density
    return compute_multi_eta()

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/digital-twin", response_class=HTMLResponse)
async def digital_twin():
    with open("templates/digital_twin.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/railpulse-3d", response_class=HTMLResponse)
async def get_railpulse_3d():
    file_path = os.path.join(os.path.dirname(__file__), "templates", "railpulse_3d.html")
    if not os.path.exists(file_path):
        return HTMLResponse(content=f"<h1>Error: {file_path} not found</h1>", status_code=404)
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
