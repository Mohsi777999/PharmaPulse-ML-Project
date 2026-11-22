import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app.routers import (
    overview, forecasting, expiry, stockout, clustering, performance, reports
)

app = FastAPI(
    title="PharmaPulse Operations Intelligence API",
    description="High-performance FastAPI decision-support backend for retail pharmacy demand forecasting, batch expiry prevention, stockout prediction, and SKU behavioural segmentation.",
    version="2.0.0"
)

# Enable CORS for React Frontend (vite runs on 5173 by default)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers
app.include_router(overview.router)
app.include_router(forecasting.router)
app.include_router(expiry.router)
app.include_router(stockout.router)
app.include_router(clustering.router)
app.include_router(performance.router)
app.include_router(reports.router)

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "system": "PharmaPulse Intelligence Engine",
        "architecture": "FastAPI + React 19 + Scikit-Learn + XGBoost",
        "branch": "Branch 1 Regional Hub",
        "currency": "PKR (Rs.)"
    }

# Mount static frontend build if available
frontend_dist = os.path.join(PROJECT_ROOT, "frontend", "dist")
if os.path.exists(frontend_dist):
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
