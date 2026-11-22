// PharmaPulse Centralized API Client

const API_BASE_URL = 'http://127.0.0.1:8000/api';

export const api = {
  // Health
  getHealth: async () => {
    const res = await fetch(`${API_BASE_URL}/health`);
    return res.json();
  },

  // Overview
  getOverview: async (timeRange = 'ALL', categoryId = 'ALL') => {
    const res = await fetch(
      `${API_BASE_URL}/overview?time_range=${encodeURIComponent(timeRange)}&category_id=${encodeURIComponent(categoryId)}`
    );
    return res.json();
  },

  // Demand Forecasting
  getMedicines: async () => {
    const res = await fetch(`${API_BASE_URL}/forecasting/medicines`);
    return res.json();
  },

  predictDemand: async (medicineId, horizonDays = 30) => {
    const res = await fetch(
      `${API_BASE_URL}/forecasting/predict?medicine_id=${encodeURIComponent(medicineId)}&horizon_days=${horizonDays}`
    );
    return res.json();
  },

  // Expiry Intelligence
  getExpiry: async (categoryId = 'ALL', riskLevel = 'ALL', maxDays = 365) => {
    const res = await fetch(
      `${API_BASE_URL}/expiry?category_id=${encodeURIComponent(categoryId)}&risk_level=${encodeURIComponent(riskLevel)}&max_days=${maxDays}`
    );
    return res.json();
  },

  // Stockout Intelligence
  getStockout: async (supplierId = 'ALL', riskLevel = 'ALL') => {
    const res = await fetch(
      `${API_BASE_URL}/stockout?supplier_id=${encodeURIComponent(supplierId)}&risk_level=${encodeURIComponent(riskLevel)}`
    );
    return res.json();
  },

  // Sales Behaviour Clustering
  getClustering: async () => {
    const res = await fetch(`${API_BASE_URL}/clustering`);
    return res.json();
  },

  // Model Performance & Benchmarks
  getPerformance: async () => {
    const res = await fetch(`${API_BASE_URL}/performance`);
    return res.json();
  },

  // Reports
  getReportsList: async () => {
    const res = await fetch(`${API_BASE_URL}/reports/list`);
    return res.json();
  },

  getReport: async (reportId) => {
    const res = await fetch(`${API_BASE_URL}/reports/${encodeURIComponent(reportId)}`);
    return res.json();
  },

  getReportCsvUrl: (reportId) => `${API_BASE_URL}/reports/${encodeURIComponent(reportId)}/csv`
};
