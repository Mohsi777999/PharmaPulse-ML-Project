import React, { useState, useEffect } from 'react';
import HeroBanner from '../components/HeroBanner';
import KpiCard from '../components/KpiCard';
import { api } from '../services/api';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend
} from 'recharts';
import { AlertTriangle, Filter, CheckCircle, ArrowRight, ShieldAlert } from 'lucide-react';

export default function ExpiryView() {
  const [data, setData] = useState(null);
  const [category, setCategory] = useState('ALL');
  const [riskLevel, setRiskLevel] = useState('ALL');
  const [maxDays, setMaxDays] = useState(180);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadExpiry();
  }, [category, riskLevel, maxDays]);

  const loadExpiry = async () => {
    try {
      setLoading(true);
      const res = await api.getExpiry(category, riskLevel, maxDays);
      setData(res);
    } catch (err) {
      console.error('Failed to load expiry data:', err);
    } finally {
      setLoading(false);
    }
  };

  const getRiskBadge = (level) => {
    if (level === 'Critical') return <span className="pill-badge pill-critical" style={{ background: 'rgba(239, 68, 68, 0.25)', border: '1px solid #EF4444', color: '#F87171', fontWeight: 700 }}>CRITICAL</span>;
    if (level === 'High') return <span className="pill-badge pill-critical">HIGH RISK</span>;
    if (level === 'Medium') return <span className="pill-badge pill-warning">MEDIUM</span>;
    return <span className="pill-badge pill-success">LOW RISK</span>;
  };

  return (
    <div>
      <HeroBanner
        tag="MODULE 2 • EXPIRY RISK INTELLIGENCE"
        title="Batch-Level Expiry Risk & Capital Exposure"
        subtitle="Dual-task machine learning intelligence predicting unsaleable inventory batches and quantifying potential capital exposure in PKR."
      />

      {/* Filter Panel in Glass Surface */}
      <div className="glass-panel" style={{ padding: '16px 20px', marginBottom: '22px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
          <Filter size={15} color="#38BDF8" />
          <div style={{ fontSize: '0.78rem', fontWeight: 700, color: '#94A3B8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Multi-Criteria Expiry Risk Filters
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '16px', alignItems: 'center' }}>
          <div>
            <label style={{ display: 'block', fontSize: '0.74rem', color: '#94A3B8', marginBottom: '6px', fontWeight: 600 }}>
              Therapeutic Category:
            </label>
            <select
              className="input-select"
              style={{ width: '100%' }}
              value={category}
              onChange={(e) => setCategory(e.target.value)}
            >
              <option value="ALL">All Categories</option>
              <option value="CAT001">Pain Relief & Fever</option>
              <option value="CAT002">Anti-Infectives</option>
              <option value="CAT003">Respiratory & Cold</option>
              <option value="CAT004">Gastrointestinal</option>
              <option value="CAT005">Chronic Care</option>
              <option value="CAT006">Vitamins & Supplements</option>
            </select>
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '0.74rem', color: '#94A3B8', marginBottom: '6px', fontWeight: 600 }}>
              Risk Level:
            </label>
            <select
              className="input-select"
              style={{ width: '100%' }}
              value={riskLevel}
              onChange={(e) => setRiskLevel(e.target.value)}
            >
              <option value="ALL">All Risk Levels</option>
              <option value="Critical">Critical Risk Only (Imminent / High Loss)</option>
              <option value="High">High Risk Only</option>
              <option value="Medium">Medium Risk Only</option>
              <option value="Low">Low Risk Only (Safe)</option>
            </select>
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '0.74rem', color: '#94A3B8', marginBottom: '6px', fontWeight: 600 }}>
              Expiry Horizon: <strong style={{ color: '#38BDF8' }}>≤ {maxDays} Days</strong>
            </label>
            <input
              type="range"
              min="14"
              max="365"
              step="7"
              value={maxDays}
              onChange={(e) => setMaxDays(Number(e.target.value))}
              style={{ width: '100%', accentColor: '#0EA5E9' }}
            />
          </div>
        </div>
      </div>

      {loading && !data ? (
        <div style={{ padding: '40px', textAlign: 'center', color: '#94A3B8' }}>
          <div className="pulse-dot" style={{ width: '10px', height: '10px', marginBottom: '10px' }}></div>
          <div>Analyzing batch expiry risks...</div>
        </div>
      ) : data ? (
        <>
          {/* 4 Summary KPIs */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '14px', marginBottom: '22px' }}>
            <KpiCard
              title="Capital at Risk"
              value={`Rs. ${data.kpis.total_capital_at_risk_k.toLocaleString()}k`}
              subtext="Total potential exposure"
              icon="💰"
              badge="Exposure"
              badgeType="critical"
            />
            <KpiCard
              title="Urgent Batches"
              value={`${data.kpis.urgent_batches} (${data.kpis.critical_risk_batches} Critical)`}
              subtext="Immediate mitigation needed"
              icon="🚨"
              badge="Urgent"
              badgeType="critical"
            />
            <KpiCard
              title="Avg Days to Expiry"
              value={`${data.kpis.avg_days_to_expiry}d`}
              subtext="Across active formulary"
              icon="⏱️"
              badge="Timeline"
              badgeType="warning"
            />
            <KpiCard
              title="Units at Risk"
              value={data.kpis.total_units_at_risk.toLocaleString()}
              subtext="Projected unsold quantity"
              icon="📦"
              badge="Volume"
              badgeType="neutral"
            />
          </div>

          {/* Visualizations Row */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '18px', marginBottom: '24px' }}>
            {/* Risk Distribution Donut */}
            <div className="glass-panel" style={{ padding: '20px' }}>
              <div style={{ fontSize: '0.9rem', fontWeight: 700, color: '#F8FAFC', marginBottom: '4px' }}>
                Risk Classification Share
              </div>
              <div style={{ fontSize: '0.74rem', color: '#94A3B8', marginBottom: '12px' }}>
                Dual-Task ML (XGBoost Classifier + Gradient Boosting Regressor)
              </div>
              <div style={{ height: '220px', width: '100%' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={data.risk_distribution}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      innerRadius={45}
                      outerRadius={75}
                      paddingAngle={4}
                    >
                      {data.risk_distribution.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{ backgroundColor: '#111827', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '8px', fontSize: '12px' }}
                    />
                    <Legend wrapperStyle={{ fontSize: '11px' }} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Category Exposure Bar Chart */}
            <div className="glass-panel" style={{ padding: '20px' }}>
              <div style={{ fontSize: '0.9rem', fontWeight: 700, color: '#F8FAFC', marginBottom: '4px' }}>
                Financial Exposure by Therapeutic Category (PKR)
              </div>
              <div style={{ fontSize: '0.74rem', color: '#94A3B8', marginBottom: '12px' }}>
                Unsold batch value calculated at cost price
              </div>
              <div style={{ height: '220px', width: '100%' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data.category_exposure}>
                    <XAxis dataKey="category_name" stroke="#64748B" fontSize={10} tickLine={false} interval={0} />
                    <YAxis stroke="#64748B" fontSize={11} tickLine={false} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#111827', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '8px', fontSize: '12px' }}
                      formatter={(val) => [`Rs. ${val.toLocaleString()}`, 'Exposure (PKR)']}
                    />
                    <Bar dataKey="exposure_pkr" fill="#38BDF8" radius={[4, 4, 0, 0]} name="Exposure (PKR)" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Formatted Batches Table */}
          <div className="glass-panel" style={{ padding: '20px', marginBottom: '24px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px' }}>
              <div>
                <div style={{ fontSize: '0.92rem', fontWeight: 700, color: '#F8FAFC' }}>
                  Active Batches Expiry Risk Register
                </div>
                <div style={{ fontSize: '0.74rem', color: '#94A3B8' }}>
                  Showing top {data.batches.length} batches sorted by financial exposure
                </div>
              </div>
              <span className="pill-badge pill-neutral">{data.batches.length} BATCHES LISTED</span>
            </div>

            <div className="data-table-container">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Batch ID</th>
                    <th>Medicine</th>
                    <th>Category</th>
                    <th>Days Left</th>
                    <th>On-Hand</th>
                    <th>Est. Unsold</th>
                    <th>Exposure (PKR)</th>
                    <th>Risk Class</th>
                    <th>Recommended Mitigation</th>
                  </tr>
                </thead>
                <tbody>
                  {data.batches.map((b, i) => (
                    <tr key={i}>
                      <td style={{ fontFamily: 'monospace', color: '#38BDF8' }}>{b.batch_id}</td>
                      <td>{b.medicine_name}</td>
                      <td style={{ color: '#94A3B8' }}>{b.category_name}</td>
                      <td>
                        <span className={`pill-badge ${b.days_to_expiry <= 30 ? 'pill-critical' : b.days_to_expiry <= 90 ? 'pill-warning' : 'pill-neutral'}`}>
                          {b.days_to_expiry}d
                        </span>
                      </td>
                      <td>{b.current_quantity}</td>
                      <td style={{ fontWeight: 600 }}>{b.predicted_unsold}</td>
                      <td style={{ color: '#F87171', fontWeight: 700 }}>
                        Rs. {Number(b.financial_exposure).toLocaleString()}
                      </td>
                      <td>{getRiskBadge(b.risk_level)}</td>
                      <td style={{ fontSize: '0.78rem', color: '#38BDF8' }}>{b.recommended_action}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* 3-Tier Mitigation Playbook Guide */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
            <div className="alert-box alert-info">
              <div style={{ fontWeight: 700, fontSize: '0.86rem', color: '#38BDF8', marginBottom: '4px' }}>
                Tier 1: Supplier Return Window (&gt;90 Days)
              </div>
              <div style={{ fontSize: '0.76rem', color: '#94A3B8', lineHeight: '1.4' }}>
                Stock in this window qualifies for contractual credit return or batch exchange with vendor before penalty period.
              </div>
            </div>

            <div className="alert-box alert-warning">
              <div style={{ fontWeight: 700, fontSize: '0.86rem', color: '#FBBF24', marginBottom: '4px' }}>
                Tier 2: Clearance Markdown (30–90 Days)
              </div>
              <div style={{ fontSize: '0.76rem', color: '#94A3B8', lineHeight: '1.4' }}>
                Apply 20%–35% retail markdown, activate bundle promotions, and lock strict FEFO priority at point-of-sale.
              </div>
            </div>

            <div className="alert-box alert-critical">
              <div style={{ fontWeight: 700, fontSize: '0.86rem', color: '#F87171', marginBottom: '4px' }}>
                Tier 3: Inter-Branch Swap (&lt;30 Days)
              </div>
              <div style={{ fontSize: '0.76rem', color: '#94A3B8', lineHeight: '1.4' }}>
                Immediately transfer to high-velocity regional sister branches or hospital wards to ensure 100% patient dispensing.
              </div>
            </div>
          </div>
        </>
      ) : null}
    </div>
  );
}
