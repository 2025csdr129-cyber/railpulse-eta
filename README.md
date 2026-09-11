# 🚆 RailPulse: Spatio-Temporal Dynamic ETA Engine for Indian Railways

> **Smart India Hackathon 2026 Submission**  
> AI-powered arrival prediction engine accounting for signal aspects, track headway, dynamic block occupancy, and priority overtakes.

---

## 📌 Problem Overview
Current arrival forecasts on Indian Railways (NTES/IRCTC) rely on static timetables and historical buffer times. They fail to capture real-time physical realities:
- Trailing slower freight rakes in automatic block signaling zones.
- Dwell times at outer signals and loop lines.
- Temporary Speed Restrictions (TSRs) and junction congestion.

**RailPulse** ingests live telemetry and uses a machine learning regression model to dynamically recalculate arrival windows, dwell times, and confidence margins in real time.

---

## 🛠️ Tech Stack
- **Backend & APIs:** Python 3, FastAPI, Uvicorn, Pydantic
- **Machine Learning:** LightGBM, Scikit-learn, Pandas, NumPy
- **Frontend & GIS:** HTML5, Tailwind CSS, Leaflet.js (OpenStreetMap/CartoDB tiles)
- **Data Simulation:** Custom synthetic RTIS & Multi-Train Telemetry Streamer

---

## 🚀 Key Features
1. **Dynamic Recalibration vs. Static Schedule:** Proves failure of naive timetable math during track bottlenecks.
2. **Multi-Train Corridor Simulation:** Models express train trailing behind a slower freight rake on the NDLS–CNB corridor.
3. **Dispatch & Loop Overtake Control:** Interactive toggle to divert freight to loop lines and restore green signal aspects.
4. **Confidence Intervals:** Computes error margins based on signal status and block rake density.

---

## 💻 Local Setup & Installation

### 1. Clone the repository
```bash
git clone [https://github.com/](https://github.com/)<YOUR_GITHUB_USERNAME>/railpulse-eta.git
cd railpulse-eta