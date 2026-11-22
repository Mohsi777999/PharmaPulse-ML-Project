# PharmaPulse — ML-Powered Pharmacy Demand & Risk Intelligence System

**Coursework Project Type:** CCA — Artificial Intelligence / Machine Learning  
**Application Architecture:** Python + FastAPI + React 19 + Vite + Scikit-Learn + XGBoost  
**Operational Scope:** 1 Pharmacy, 1 Branch, 150 Medicines, 3 Years of Data (1,095 Days)  
**Primary Objective:** Apply advanced Machine Learning techniques to pharmaceutical demand forecasting, batch-level expiry risk prediction, vendor lead-time stockout prediction, and unsupervised medicine sales behaviour clustering.

---
## 👥 Project Contributors

| Name | Role | GitHub |
|---|---|---|
| Mohsin Ali | Project Developer | [@yourusername](https://github.com/yourusername) |
| Rafay Raza | Project Collaborator | [@friendusername](https://github.com/friendusername) |

---

## 1. Executive Summary & Problem Statement

Retail and hospital pharmacy operations face a dual-ended inventory challenge:
1. **Stockouts:** Running out of critical medications compromises patient clinical outcomes and causes lost revenue.
2. **Expiry Losses:** Over-ordering or failing to account for consumption lulls leads to expired batches that must be discarded at total financial loss.

**PharmaPulse** is an intelligent decision-support system that bridges the gap between historical dispensing logs and future procurement decisions. Built on a modern decoupled architecture, the platform pairs a high-performance **FastAPI REST API backend** with an enterprise-grade **React 19 single-page command center**.

---

## 2. System Architecture & Folder Layout

```text
PharmaPulse/
│
├── app.py                             # Unified FastAPI & React application launcher
├── requirements.txt                   # Backend Python dependencies
├── README.md                          # Master technical documentation
├── README_DATASET.md                  # Comprehensive dataset specification
│
├── backend/                           # FastAPI REST API Backend
│   └── app/
│       ├── main.py                    # FastAPI application & static mount
│       └── routers/
│           ├── overview.py            # Executive KPIs, velocity & alerts
│           ├── forecasting.py         # Multi-horizon forward predictions
│           ├── expiry.py              # Perishable batch exposure & playbook
│           ├── stockout.py            # 7-day replenishment lookahead
│           ├── clustering.py          # K-Means SKU profiles & PCA coords
│           ├── performance.py         # ML benchmark metrics & confusion matrices
│           └── reports.py             # 6 audit reports & CSV downloads
│
├── frontend/                          # Modern React 19 + Vite SPA Command Center
│   ├── src/
│   │   ├── components/                # Sidebar, Header, KpiCard, HeroBanner
│   │   ├── views/                     # 7 Full Analytics Views (Overview to Reports)
│   │   ├── services/api.js            # Centralized API service client
│   │   ├── App.jsx                    # Root application router & layout
│   │   └── index.css                  # Obsidian Dark Enterprise SaaS design tokens
│   ├── dist/                          # Production-optimized build assets
│   └── package.json                   # React, Vite, Recharts, Lucide-React
│
├── data/
│   ├── raw/                           # 10 raw simulation CSV tables
│   ├── processed/                     # 5 leakage-free ML datasets
│   ├── outputs/                       # Data dictionary, validation report, summary
│   ├── model_evaluations.csv          # 156 benchmark records across algorithms
│   └── model_predictions.csv          # 5,943 model prediction records
│
├── modules/
│   ├── simulation/                    # Modular 3-year simulation components
│   │   ├── demand_generator.py        # Stochastic Negative Binomial generator
│   │   ├── inventory_engine.py        # FEFO dispensing & daily balance engine
│   │   ├── supplier_engine.py         # Vendor lead time & stochastic delays
│   │   └── simulation_runner.py       # 1,095-day top-level simulation coordinator
│   │
│   ├── forecasting/                   # Demand forecasting module
│   │   ├── features.py                # Shifted past lags & calendar feature engineering
│   │   ├── train.py                   # Multi-algorithm trainer (Naive to XGBoost)
│   │   └── predict.py                 # Multi-horizon forward recursive predictor
│   │
│   ├── expiry/                        # Expiry risk prediction module
│   │   ├── features.py                # Snapshot feature pipeline & grouped splits
│   │   ├── classification.py          # Task A: Unsold risk classification (0 vs 1)
│   │   └── regression.py              # Task B: Unsold units regression & exposure
│   │
│   ├── stockout/                      # Stockout prediction module
│   │   ├── features.py                # Order-time predictors & chronological split
│   │   └── train.py                   # Recall-focused stockout classifiers
│   │
│   ├── clustering/                    # SKU behavioural clustering module
│   │   ├── features.py                # 90-day rolling feature matrix
│   │   └── train.py                   # K-Means, Agglomerative, GMM, DBSCAN
│   │
│   └── intelligence/                  # Integrated Intelligence Layer
│       └── insights.py                # Cross-module rule-based alerts synthesizer
│
├── utils/                             # Core helper utilities
│   ├── data_loader.py                 # Centralized cached CSV loader (@lru_cache)
│   ├── validation.py                  # Defensive runtime input validators
│   ├── metrics.py                     # Safe WAPE, safe MAPE, classification/regression metrics
│   └── visualization.py               # Standardized Plotly healthcare visual templates
│
├── models/                            # Serialized model artifacts (.joblib)
│   ├── forecasting/                   # XGBoost, RF, Linear models, feature columns
│   ├── expiry/                        # Classifiers, regressors, scalers, confusion matrices
│   ├── stockout/                      # Classifiers, scalers, confusion matrices
│   └── clustering/                    # K-Means, PCA coordinate scalers, cluster profiles
│
├── scripts/                           # Reproducibility scripts
│   ├── generate_master_data.py        # Master catalog generator
│   ├── simulate_pharmacy.py           # Standalone simulation runner
│   ├── build_ml_datasets.py           # ML feature builder
│   ├── generate_data_dictionary.py    # Data dictionary generator
│   ├── validate_datasets.py           # Automated integrity & reconciliation suite
│   └── train_all_models.py            # Master ML training orchestrator
│
└── pharmapulse_dataset_complete.zip   # Final deliverable archive
```

---

## 3. Quickstart & Installation Guide

### Prerequisites
- Python 3.10 to 3.14 (Python 3.13 tested).
- Node.js v18+ and npm.

### Setup Steps
```bash
# 1. Clone or navigate to the project directory
cd PharmaPulse

# 2. Install required Python packages
pip install -r requirements.txt

# 3. Install frontend dependencies and build assets
cd frontend
npm install
npm run build
cd ..

# 4. Launch the unified application (FastAPI + React SPA)
python app.py
```

The application will launch at `http://127.0.0.1:8000`.  
- Interactive React Command Center: `http://127.0.0.1:8000`  
- OpenAPI Swagger Documentation: `http://127.0.0.1:8000/docs`

---

## 4. The Four Machine Learning Modules

### 4.1 Demand Forecasting Module
- **Objective:** Predict daily customer demand per SKU across 7-day, 30-day, and multi-horizon periods up to 180 days.
- **Algorithms Compared:** Naive Baseline (Lag 1), Seasonal Naive (Lag 7), Linear Regression, Random Forest Regressor, XGBoost Regressor.
- **Evaluation Split:** Chronological 70% Train (766 days), 15% Validation (164 days), 15% Test (165 days).
- **Leakage Prevention:** All lag (`lag_1`, `lag_7`, `lag_14`, `lag_28`) and rolling statistics (`rolling_mean_7`, `rolling_mean_14`, `rolling_mean_28`, `rolling_std_7`, `rolling_std_28`) are computed on strictly shifted past demand (`demand.shift(1)`).
- **Empirical Results (Test Set):**
  - **Naive Baseline:** WAPE = 72.64%, MAE = 9.87
  - **Seasonal Naive:** WAPE = 72.56%, MAE = 9.86
  - **Linear Regression:** WAPE = 53.24%, MAE = 7.23
  - **XGBoost (Winner):** WAPE = **53.82%**, MAE = **7.31** (Best non-linear generalization)

### 4.2 Expiry Risk Prediction Module
- **Objective:** Predict whether physical inventory batches will reach expiry with unsold units and quantify financial loss exposure.
- **Task A (Classification):** Predict unsold occurrence (`target_unsold_flag`: 0 vs 1).
  - Algorithms: Logistic Regression, Random Forest, Gradient Boosting, XGBoost.
  - Winner: **Logistic Regression** (Test Recall = **0.9444**, F1 = **0.8500**) & **XGBoost** (Test F1 = **0.7333**).
- **Task B (Regression):** Predict exact unsold quantity at expiry (`target_unsold_quantity`).
  - Algorithms: Linear Regression, Decision Tree, Random Forest, Gradient Boosting.
  - Winner: **Gradient Boosting Regressor** (Test MAE = **6.72 units**, $R^2 = \mathbf{0.8665}$).
- **Financial Exposure:** $\text{Predicted Unsold Units} \times \text{Unit Acquisition Cost (PKR)}$.
- **Risk Categorization:** Low (<5%), Medium (5% to <15%), High (15% to <30%), Critical ($\ge 30\%$).
- **Leakage Prevention:** Grouped batch-level splitting ensures all snapshots from the same physical batch remain within a single partition.

### 4.3 Stockout Prediction Module
- **Objective:** Predict if usable inventory will reach zero during supplier lead time strictly between order date and delivery arrival.
- **Algorithms Compared:** Logistic Regression, Random Forest, Gradient Boosting, XGBoost. (Decision Tree explicitly excluded).
- **Class-Weight Optimization:** Heavy weight assigned to positive class to prioritize **Recall** (cost of missing a stockout is far greater than false alarm).
- **Empirical Results (Test Set):**
  - **Logistic Regression:** Recall = **0.8259**, F1 = 0.6317
  - **XGBoost (Winner):** Recall = **0.8148**, Precision = 0.5556, F1 = **0.6607**, Accuracy = 83.1%

### 4.4 Sales Behaviour Clustering Module
- **Objective:** Unsupervised segmentation of the 150 SKUs based on a 90-day rolling consumption window without labels.
- **Algorithms Compared:** K-Means, Agglomerative Clustering, DBSCAN, Gaussian Mixture Models (GMM).
- **Evaluation:** Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index.
- **Winner:** **K-Means ($k=3$)** (Silhouette = **0.4201**, Davies-Bouldin = **1.0584**).
- **Discovered Operational Segments:**
  1. *High-Volume Consistent Staples:* Predictable steady consumption, high sales frequency, low volatility.
  2. *Highly Seasonal Acute SKUs:* Winter/summer climatic consumption peaks.
  3. *Irregular & Erratic Spikers:* High zero-demand probability with intermittent spikes.

---

## 5. Daily Simulation Engine & Mathematical Reconciliation

The 3-year simulation strictly enforces physical inventory conservation across all 164,250 medicine-days:

$$\text{ClosingStock}(t) = \text{OpeningStock}(t) + \text{ReceivedQuantity}(t) - \text{ExpiredQuantity}(t) - \text{UnitsSold}(t)$$
$$\text{OpeningStock}(t+1) = \text{ClosingStock}(t)$$

- **FEFO Dispensing:** Active batches are sorted by `expiry_date ASC, received_date ASC`. Sales are allocated sequentially to earliest expiring stock.
- **Expiry Timing:** A batch expiring on date $D$ is removed at the start of day $D$ before demand allocation.
- **Reconciliation Failures:** **0** (verified across 100% of rows by `validate_datasets.py`).

---

## 6. Deliverable Verification Commands

```bash
# Run automated data validation suite
python scripts/validate_datasets.py

# Re-run ML model training and benchmarking
python scripts/train_all_models.py

# Launch unified application (FastAPI backend + React frontend)
python app.py
```
