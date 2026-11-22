"""
PharmaPulse - ML-Powered Pharmacy Demand & Risk Intelligence System
FastAPI Backend & React SPA Application Launcher
Architecture: Python + FastAPI + React 19 + Vite + Scikit-Learn + XGBoost
"""

import os
import sys
import uvicorn

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def main():
    print("=" * 70)
    print("  PHARMAPULSE: ML-POWERED PHARMACY INTELLIGENCE SYSTEM")
    print("  Architecture: FastAPI REST Backend + React 19 Command Center")
    print("  Access Web Dashboard at: http://127.0.0.1:8000")
    print("  Access OpenAPI Swagger Docs at: http://127.0.0.1:8000/docs")
    print("=" * 70)
    
    uvicorn.run(
        "backend.app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False
    )

if __name__ == "__main__":
    main()
