import React, { useState, useEffect } from 'react';
import HeroBanner from '../components/HeroBanner';
import KpiCard from '../components/KpiCard';
import { api } from '../services/api';
import { ShieldAlert, Truck, AlertOctagon, CheckCircle2, Clock } from 'lucide-react';

export default function StockoutView() {
  const [data, setData] = useState(null);
  const [supplierId, setSupplierId] = useState('ALL');
  const [riskLevel, setRiskLevel] = useState('ALL');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadStockout();
  }, [supplierId, riskLevel]);

  const loadStockout = async () => {
    try {
      setLoading(true);
      const res = await api.getStockout(supplierId, riskLevel);
      setData(res);
    } catch (err) {
      console.error('Failed to load stockout data:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <HeroBanner
        tag="MODULE 3 • STOCKOUT INTELLIGENCE"
        title="Stockout Intelligence & Replenishment Pipeline Risk"
        subtitle="Supervised machine learning early-warning engine forecasting usable inventory exhaustion before vendor shipments arrive."
      />

      {/* Filter Panel in Glass Surface */}
      <div className="glass-panel" style={{ padding: '16px 20px', marginBottom: '22px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
          <ShieldAlert size={15} color="#38BDF8" />
          <div style={{ fontSize: '0.78rem', fontWeight: 700, color: '#94A3B8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Stockout Pipeline Filters
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', alignItems: 'center' }}>
          <div>
            <label style={{ display: 'block', fontSize: '0.74rem', color: '#94A3B8', marginBottom: '6px', fontWeight: 600 }}>
              Filter by Vendor / Supplier:
            </label>
            <select
              className="input-select"
              style={{ width: '100%' }}
              value={supplierId}
              onChange={(e) => setSupplierId(e.target.value)}
            >
              <option value="ALL">All 10 Suppliers</option>
              {data && data.suppliers && data.suppliers.map((s) => (
                <option key={s.supplier_id} value={s.supplier_id}>
                  {s.supplier_name} ({s.reliability_pct}% on-time)
                </option>
              ))}
            </select>
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '0.74rem', color: '#94A3B8', marginBottom: '6px', fontWeight: 600 }}>
              Risk Classification:
            </label>
            <select
              className="input-select"
              style={{ width: '100%' }}
              value={riskLevel}
              onChange={(e) => setRiskLevel(e.target.value)}
            >
              <option value="ALL">All Statuses</option>
              <option value="High Risk">High Risk Threats Only</option>
              <option value="Low Risk">Adequate Inventory Only</option>
            </select>
          </div>
        </div>
      </div>

      {loading && !data ? (
        <div style={{ padding: '40px', textAlign: 'center', color: '#94A3B8' }}>
          <div className="pulse-dot" style={{ width: '10px', height: '10px', marginBottom: '10px' }}></div>
          <div>Evaluating inventory burn rates against supplier lead times...</div>
        </div>
      ) : data ? (
        <>
          {/* 4 Summary KPIs */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '14px', marginBottom: '22px' }}>
            <KpiCard
              title="Monitored Pipeline Events"
              value={data.kpis.total_monitored_events}
              subtext="Active SKU lead-time windows"
              icon="🔍"
              badge="Tracking"
              badgeType="neutral"
            />
            <KpiCard
              title="High-Risk Threats"
              value={data.kpis.high_risk_threats}
              subtext="Inventory exhaust before arrival"
              icon="🚨"
              badge="Threats"
              badgeType="critical"
            />
            <KpiCard
              title="Vulnerability Rate"
              value={`${data.kpis.vulnerability_rate_pct}%`}
              subtext="Overall stockout exposure"
              icon="⚡"
              badge="Risk %"
              badgeType="warning"
            />
            <KpiCard
              title="Avg Days of Supply"
              value={`${data.kpis.avg_days_of_supply}d`}
              subtext="Across formulary on-hand"
              icon="📦"
              badge="Buffer"
              badgeType="cyan"
            />
          </div>

          {/* Supplier Reliability & Dynamic Buffer Matrix */}
          <div className="glass-panel" style={{ padding: '20px', marginBottom: '24px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px' }}>
              <div>
                <div style={{ fontSize: '0.92rem', fontWeight: 700, color: '#F8FAFC' }}>
                  Supplier Delivery Reliability & Safety Buffer Multipliers
                </div>
                <div style={{ fontSize: '0.74rem', color: '#94A3B8' }}>
                  Click any supplier card to filter threats and purchase orders (Dynamic buffer adjusted by lead-time volatility)
                </div>
              </div>
              <span className="pill-badge pill-cyan">10 QUALIFIED SUPPLIERS</span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '12px' }}>
              {data.suppliers.map((s, i) => {
                const isSelected = supplierId === s.supplier_id;
                return (
                  <div
                    key={i}
                    onClick={() => setSupplierId(isSelected ? 'ALL' : s.supplier_id)}
                    style={{
                      background: isSelected ? 'rgba(56, 189, 248, 0.12)' : '#0F172A',
                      border: isSelected ? '1px solid #38BDF8' : '1px solid var(--border-subtle)',
                      boxShadow: isSelected ? '0 0 12px rgba(56, 189, 248, 0.25)' : 'none',
                      borderRadius: '8px',
                      padding: '12px',
                      display: 'flex',
                      flexDirection: 'column',
                      justifyContent: 'space-between',
                      cursor: 'pointer',
                      transition: 'all 0.2s ease'
                    }}
                    title={`Click to filter by ${s.supplier_name}`}
                  >
                    <div style={{ fontSize: '0.8rem', fontWeight: 700, color: isSelected ? '#38BDF8' : '#F1F5F9', marginBottom: '4px' }}>
                      {s.supplier_name}
                    </div>
                    <div style={{ fontSize: '0.72rem', color: '#94A3B8', marginBottom: '8px' }}>
                      Lead Time: <strong>{s.lead_time_days} days</strong>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <span style={{ fontSize: '0.82rem', fontWeight: 800, color: s.reliability_pct >= 95 ? '#34D399' : s.reliability_pct >= 90 ? '#38BDF8' : s.reliability_pct >= 85 ? '#FBBF24' : '#F87171' }}>
                        {s.reliability_pct}%
                      </span>
                      <span className={`pill-badge ${isSelected ? 'pill-cyan' : 'pill-neutral'}`}>
                        {s.buffer_multiplier}x Buffer
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* 7-Day Ahead Stockout Threat Register */}
          <div className="glass-panel" style={{ padding: '20px', marginBottom: '24px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px' }}>
              <div>
                <div style={{ fontSize: '0.92rem', fontWeight: 700, color: '#F8FAFC' }}>
                  7-Day Replenishment Threat Register
                </div>
                <div style={{ fontSize: '0.74rem', color: '#94A3B8' }}>
                  Evaluated with XGBoost Classifier (81.48% Recall, 0.897 ROC-AUC)
                </div>
              </div>
              <span className="pill-badge pill-neutral">{data.items.length} ITEMS EVALUATED</span>
            </div>

            <div className="data-table-container">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Medicine</th>
                    <th>Category</th>
                    <th>Current Stock</th>
                    <th>Daily Burn</th>
                    <th>Days Supply</th>
                    <th>Lead Time</th>
                    <th>Supplier</th>
                    <th>Risk Status</th>
                    <th>Action Protocol</th>
                  </tr>
                </thead>
                <tbody>
                  {data.items.slice(0, 50).map((it, i) => (
                    <tr key={i}>
                      <td style={{ fontWeight: 600 }}>{it.medicine_name}</td>
                      <td style={{ color: '#94A3B8' }}>{it.category_name}</td>
                      <td>{it.current_stock}</td>
                      <td>{it.burn_rate} u/d</td>
                      <td>
                        <span className={`pill-badge ${it.days_of_supply <= 5 ? 'pill-critical' : it.days_of_supply <= 10 ? 'pill-warning' : 'pill-success'}`}>
                          {it.days_of_supply}d
                        </span>
                      </td>
                      <td>{it.lead_time_days}d</td>
                      <td style={{ color: '#94A3B8' }}>{it.supplier_name}</td>
                      <td>
                        {it.risk_flag === 'High Risk' ? (
                          <span className="pill-badge pill-critical">HIGH THREAT</span>
                        ) : (
                          <span className="pill-badge pill-success">ADEQUATE</span>
                        )}
                      </td>
                      <td style={{ fontSize: '0.78rem', color: it.risk_flag === 'High Risk' ? '#F87171' : '#38BDF8' }}>
                        {it.recommended_action}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Active Purchase Orders Tracking Ledger */}
          <div className="glass-panel" style={{ padding: '20px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px' }}>
              <div>
                <div style={{ fontSize: '0.92rem', fontWeight: 700, color: '#F8FAFC' }}>
                  Recent Inbound Purchase Orders Ledger
                </div>
                <div style={{ fontSize: '0.74rem', color: '#94A3B8' }}>
                  Inbound shipments tracked against scheduled arrival dates
                </div>
              </div>
              <span className="pill-badge pill-neutral">{data.recent_pos.length} INBOUND ORDERS</span>
            </div>

            <div className="data-table-container">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>PO ID</th>
                    <th>Medicine</th>
                    <th>Supplier</th>
                    <th>Quantity</th>
                    <th>Order Date</th>
                    <th>Expected Delivery</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {data.recent_pos.map((p, i) => (
                    <tr key={i}>
                      <td style={{ fontFamily: 'monospace', color: '#38BDF8' }}>{p.po_id}</td>
                      <td>{p.medicine_name}</td>
                      <td style={{ color: '#94A3B8' }}>{p.supplier_name}</td>
                      <td style={{ fontWeight: 600 }}>{p.quantity.toLocaleString()}</td>
                      <td>{p.order_date}</td>
                      <td>{p.expected_delivery}</td>
                      <td>
                        <span className={`pill-badge ${p.status === 'Delivered' ? 'pill-success' : 'pill-cyan'}`}>
                          {p.status}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      ) : null}
    </div>
  );
}
