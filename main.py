from fastapi import FastAPI
from fastapi.responses import FileResponse
import os

app = FastAPI(title="RailPulse Engine")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

@app.get("/")
@app.get("/railpulse-3d")
@app.get("/passenger")
async def passenger_view():
    file_path = os.path.join(TEMPLATES_DIR, "railpulse_3d.html")
    return FileResponse(file_path)

@app.get("/controller")
async def controller_view():
    file_path = os.path.join(TEMPLATES_DIR, "controller_view.html")
    return FileResponse(file_path)
