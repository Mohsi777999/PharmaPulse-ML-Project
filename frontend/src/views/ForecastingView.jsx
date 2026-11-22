import React, { useState, useEffect } from 'react';
import HeroBanner from '../components/HeroBanner';
import KpiCard from '../components/KpiCard';
import { api } from '../services/api';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  Area, ComposedChart, Legend
} from 'recharts';
import { Sliders, TrendingUp, DollarSign, ShieldAlert, Cpu } from 'lucide-react';

export default function ForecastingView() {
  const [medicines, setMedicines] = useState([]);
  const [selectedMedId, setSelectedMedId] = useState('MED_001');
  const [horizon, setHorizon] = useState(30);
  const [forecastData, setForecastData] = useState(null);
  const [loading, setLoading] = useState(true);

  // Load medicines list once
  useEffect(() => {
    api.getMedicines().then((res) => {
      setMedicines(res);
      if (res.length > 0) {
        setSelectedMedId(res[0].medicine_id);
      }
    }).catch(console.error);
  }, []);

  // Load forecast when parameters change
  useEffect(() => {
    if (selectedMedId) {
      loadForecast();
    }
  }, [selectedMedId, horizon]);

  const loadForecast = async () => {
    try {
      setLoading(true);
      const res = await api.predictDemand(selectedMedId, horizon);
      setForecastData(res);
    } catch (err) {
      console.error('Failed to load forecast:', err);
    } finally {
      setLoading(false);
    }
  };

  // Combine historical and forecast for composite chart
  const combinedChartData = [];
  if (forecastData) {
    forecastData.history.forEach((h) => {
      combinedChartData.push({
        date: h.date,
        actual: h.actual_demand,
        predicted: null,
        lower: null,
        upper: null
      });
    });

    forecastData.forecast.forEach((f) => {
      combinedChartData.push({
        date: f.date,
        actual: null,
        predicted: f.predicted_demand,
        lower: f.lower_bound,
        upper: f.upper_bound
      });
    });
  }

  return (
    <div>
      <HeroBanner
        tag="MODULE 1 • DEMAND FORECASTING ENGINE"
        title="Machine Learning Demand Forecasting"
        subtitle="Multi-horizon forward demand projections leveraging historical lag dynamics, calendar seasonality, and gradient-boosted decision trees."
      />

      {/* Control Panel in Glass Surface */}
      <div className="glass-panel" style={{ padding: '18px 22px', marginBottom: '22px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '14px' }}>
          <Sliders size={16} color="#38BDF8" />
          <div style={{ fontSize: '0.82rem', fontWeight: 700, color: '#F8FAFC', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Forecast Generation Controls
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr 1fr', gap: '16px', alignItems: 'center' }}>
          <div>
            <label style={{ display: 'block', fontSize: '0.74rem', color: '#94A3B8', marginBottom: '6px', fontWeight: 600 }}>
              Select Medicine / SKU:
            </label>
            <select
              className="input-select"
              style={{ width: '100%' }}
              value={selectedMedId}
              onChange={(e) => setSelectedMedId(e.target.value)}
            >
              {medicines.map((m) => (
                <option key={m.medicine_id} value={m.medicine_id}>
                  {m.medicine_name} ({m.medicine_id}) — {m.category_name}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '0.74rem', color: '#94A3B8', marginBottom: '6px', fontWeight: 600 }}>
              Forecast Horizon: <strong style={{ color: '#38BDF8' }}>{horizon} Days</strong>
            </label>
            <input
              type="range"
              min="7"
              max="180"
              step="7"
              value={horizon}
              onChange={(e) => setHorizon(Number(e.target.value))}
              style={{ width: '100%', accentColor: '#0EA5E9' }}
            />
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '0.74rem', color: '#94A3B8', marginBottom: '6px', fontWeight: 600 }}>
              Active Model Architecture:
            </label>
            <div style={{
              padding: '10px 14px',
              background: 'rgba(14, 165, 233, 0.08)',
              border: '1px solid rgba(14, 165, 233, 0.25)',
              borderRadius: '8px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between'
            }}>
              <div>
                <div style={{ fontSize: '0.86rem', fontWeight: 700, color: '#38BDF8' }}>
                  {forecastData?.champion?.model_name || 'Linear Regression'}
                </div>
                <div style={{ fontSize: '0.68rem', color: '#94A3B8' }}>
                  Lowest Test WAPE (53.24%) • Selected
                </div>
              </div>
              <span className="pill-badge pill-success" style={{ fontSize: '0.68rem', padding: '2px 8px' }}>
                CHAMPION
              </span>
            </div>
          </div>
        </div>
      </div>

      {loading && !forecastData ? (
        <div style={{ padding: '40px', textAlign: 'center', color: '#94A3B8' }}>
          <div className="pulse-dot" style={{ width: '10px', height: '10px', marginBottom: '10px' }}></div>
          <div>Generating recursive forward forecast...</div>
        </div>
      ) : forecastData ? (
        <>
          {/* SKU Economics Banner */}
          <div style={{
            background: 'rgba(15, 23, 42, 0.8)',
            border: '1px solid var(--border-subtle)',
            borderRadius: '10px',
            padding: '14px 20px',
            marginBottom: '20px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between'
          }}>
            <div>
              <span style={{ fontSize: '1.05rem', fontWeight: 700, color: '#F8FAFC' }}>
                {forecastData.medicine.medicine_name}
              </span>
              <span style={{ marginLeft: '10px', fontSize: '0.78rem', color: '#38BDF8', fontFamily: 'monospace' }}>
                {forecastData.medicine.medicine_id}
              </span>
            </div>
            <div style={{ display: 'flex', gap: '20px', fontSize: '0.8rem', color: '#94A3B8' }}>
              <div>Cost: <strong style={{ color: '#F1F5F9' }}>Rs. {forecastData.medicine.unit_cost}</strong></div>
              <div>Price: <strong style={{ color: '#F1F5F9' }}>Rs. {forecastData.medicine.selling_price}</strong></div>
              <div>Gross Margin: <strong style={{ color: '#34D399' }}>{forecastData.medicine.gross_margin_pct}%</strong></div>
              <div>Reorder Threshold: <strong style={{ color: '#FBBF24' }}>{forecastData.medicine.reorder_level} units</strong></div>
            </div>
          </div>

          {/* 4 Summary KPIs */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '14px', marginBottom: '22px' }}>
            <KpiCard
              title={`${horizon}-Day Demand`}
              value={`${forecastData.summary.total_forecast_units.toLocaleString()} units`}
              subtext="Projected total volume"
              icon="📈"
              badge="Demand"
              badgeType="cyan"
            />
            <KpiCard
              title="Daily Run Rate"
              value={`${forecastData.summary.avg_daily_demand} units/day`}
              subtext="Mean velocity"
              icon="⚡"
              badge="Velocity"
              badgeType="neutral"
            />
            <KpiCard
              title="Projected Revenue"
              value={`Rs. ${Math.round(forecastData.summary.estimated_revenue_pkr).toLocaleString()}`}
              subtext="Gross turnover"
              icon="💰"
              badge="Revenue"
              badgeType="success"
            />
            <KpiCard
              title="Safety Stock Buffer"
              value={`${forecastData.summary.safety_stock_recommended} units`}
              subtext="Lead-time buffer"
              icon="🛡️"
              badge="Recommended"
              badgeType="warning"
            />
          </div>

          {/* Forecast Visualization Chart */}
          <div className="glass-panel" style={{ padding: '22px', marginBottom: '24px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
              <div>
                <div style={{ fontSize: '0.94rem', fontWeight: 700, color: '#F8FAFC' }}>
                  Demand History & Forward Forecast Cone (80% Confidence Band)
                </div>
                <div style={{ fontSize: '0.74rem', color: '#94A3B8' }}>
                  Historical 60-day observed actuals paired with {horizon}-day recursive machine learning prediction
                </div>
              </div>
              <span className="pill-badge pill-success">
                {(forecastData?.champion?.model_name || 'LINEAR REGRESSION').toUpperCase()} (CHAMPION)
              </span>
            </div>

            <div style={{ height: '320px', width: '100%' }}>
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={combinedChartData}>
                  <XAxis dataKey="date" stroke="#64748B" fontSize={11} tickLine={false} />
                  <YAxis stroke="#64748B" fontSize={11} tickLine={false} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#111827', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '8px', fontSize: '12px' }}
                  />
                  <Legend verticalAlign="top" height={36} wrapperStyle={{ fontSize: '12px' }} />
                  {/* Historical demand */}
                  <Line type="monotone" dataKey="actual" stroke="#94A3B8" strokeWidth={2} dot={false} name="Actual Historical Sales" />
                  {/* Predicted demand */}
                  <Line type="monotone" dataKey="predicted" stroke="#38BDF8" strokeWidth={2.5} dot={false} name="Forecasted Demand" />
                  {/* Upper confidence bound */}
                  <Area type="monotone" dataKey="upper" stroke="none" fill="#38BDF8" fillOpacity={0.15} name="Upper 80% Ribbon" />
                  {/* Lower confidence bound */}
                  <Line type="monotone" dataKey="lower" stroke="#0284C7" strokeDasharray="3 3" strokeWidth={1} dot={false} name="Lower 80% Ribbon" />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      ) : null}
    </div>
  );
}
