import React, { useState, useEffect } from 'react';
import HeroBanner from '../components/HeroBanner';
import KpiCard from '../components/KpiCard';
import { api } from '../services/api';
import {
  ScatterChart, Scatter, XAxis, YAxis, Tooltip, ResponsiveContainer,
  Cell, Legend
} from 'recharts';
import { PieChart, Search, Box, CloudRain, Zap, ShieldCheck } from 'lucide-react';

export default function ClusteringView() {
  const [data, setData] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [activeCohort, setActiveCohort] = useState('ALL');
  const [selectedSku, setSelectedSku] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadClustering();
  }, []);

  const loadClustering = async () => {
    try {
      setLoading(true);
      const res = await api.getClustering();
      setData(res);
      if (res.sku_points && res.sku_points.length > 0) {
        setSelectedSku(res.sku_points[0]);
      }
    } catch (err) {
      console.error('Failed to load clustering data:', err);
    } finally {
      setLoading(false);
    }
  };

  const CLUSTER_COLORS = {
    0: '#38BDF8', // Core Staples (Cyan)
    1: '#34D399', // Seasonal (Emerald)
    2: '#FBBF24'  // Erratic Spikers (Amber)
  };

  const CLUSTER_ICONS = {
    0: <Box size={20} color="#38BDF8" />,
    1: <CloudRain size={20} color="#34D399" />,
    2: <Zap size={20} color="#FBBF24" />
  };

  const filteredSkus = data?.sku_points?.filter((s) => {
    const matchesCohort = activeCohort === 'ALL' || s.cluster_id === activeCohort;
    const matchesQuery = s.medicine_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      s.medicine_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      s.cluster_name.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesCohort && matchesQuery;
  }) || [];

  return (
    <div>
      <HeroBanner
        tag="MODULE 4 • SALES BEHAVIOUR CLUSTERING"
        title="Medicine Sales Behaviour Clustering & Segmentation"
        subtitle="Unsupervised machine learning discovering natural pharmacy consumption archetypes across a 90-day rolling operational window."
      />

      {loading && !data ? (
        <div style={{ padding: '40px', textAlign: 'center', color: '#94A3B8' }}>
          <div className="pulse-dot" style={{ width: '10px', height: '10px', marginBottom: '10px' }}></div>
          <div>Projecting 150 SKUs across principal component space...</div>
        </div>
      ) : data ? (
        <>
          {/* 3 Discovered Behavioural Cohort Cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px', marginBottom: '24px' }}>
            {data.profiles.map((p) => {
              const borderCol = CLUSTER_COLORS[p.cluster_id] || '#38BDF8';
              const isActive = activeCohort === p.cluster_id;
              return (
                <div
                  key={p.cluster_id}
                  onClick={() => setActiveCohort(isActive ? 'ALL' : p.cluster_id)}
                  className="glass-panel"
                  style={{
                    padding: '20px',
                    borderTop: `3px solid ${borderCol}`,
                    border: isActive ? `1px solid ${borderCol}` : undefined,
                    boxShadow: isActive ? `0 0 16px ${borderCol}40` : 'none',
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'space-between',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease'
                  }}
                  title={`Click to filter medicines by ${p.cluster_name}`}
                >
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        {CLUSTER_ICONS[p.cluster_id]}
                        <span style={{ fontSize: '0.98rem', fontWeight: 800, color: isActive ? borderCol : '#F8FAFC' }}>
                          {p.cluster_name}
                        </span>
                      </div>
                      <span className={`pill-badge ${isActive ? 'pill-cyan' : 'pill-neutral'}`}>
                        {isActive ? 'FILTER ACTIVE' : `${p.sku_count} SKUs`}
                      </span>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '8px', margin: '14px 0', fontSize: '0.78rem' }}>
                      <div style={{ background: '#0F172A', padding: '8px', borderRadius: '6px' }}>
                        <div style={{ color: '#94A3B8', fontSize: '0.68rem' }}>Daily Demand</div>
                        <div style={{ fontWeight: 700, color: '#F1F5F9', fontSize: '0.92rem' }}>{p.avg_daily_demand} u/d</div>
                      </div>
                      <div style={{ background: '#0F172A', padding: '8px', borderRadius: '6px' }}>
                        <div style={{ color: '#94A3B8', fontSize: '0.68rem' }}>Volatility (CV)</div>
                        <div style={{ fontWeight: 700, color: borderCol, fontSize: '0.92rem' }}>{p.avg_cv}</div>
                      </div>
                      <div style={{ background: '#0F172A', padding: '8px', borderRadius: '6px' }}>
                        <div style={{ color: '#94A3B8', fontSize: '0.68rem' }}>Margin</div>
                        <div style={{ fontWeight: 700, color: '#34D399', fontSize: '0.92rem' }}>{p.avg_margin_pct}%</div>
                      </div>
                    </div>
                  </div>

                  <div style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: '10px', fontSize: '0.74rem', color: '#94A3B8', lineHeight: '1.4' }}>
                    <strong style={{ color: borderCol }}>Strategy: </strong>
                    {p.recommended_strategy}
                  </div>
                </div>
              );
            })}
          </div>

          {/* PCA 2D Cluster Scatter Plot */}
          <div className="glass-panel" style={{ padding: '22px', marginBottom: '24px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
              <div>
                <div style={{ fontSize: '0.94rem', fontWeight: 700, color: '#F8FAFC' }}>
                  2D Principal Component Projection (150 Formulary Medicines)
                </div>
                <div style={{ fontSize: '0.74rem', color: '#94A3B8' }}>
                  Dimensionality reduced feature space: volume, CV volatility, demand peaks, and margin (Silhouette: {data.kpis.silhouette_score}, Variance: {data.kpis.variance_explained_pct}%)
                </div>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', fontSize: '0.72rem' }}>
                <span onClick={() => setActiveCohort(activeCohort === 0 ? 'ALL' : 0)} style={{ color: '#38BDF8', cursor: 'pointer', fontWeight: activeCohort === 0 ? 800 : 500 }}>● Core Staples (88)</span>
                <span onClick={() => setActiveCohort(activeCohort === 1 ? 'ALL' : 1)} style={{ color: '#34D399', cursor: 'pointer', fontWeight: activeCohort === 1 ? 800 : 500 }}>● Seasonal Waves (32)</span>
                <span onClick={() => setActiveCohort(activeCohort === 2 ? 'ALL' : 2)} style={{ color: '#FBBF24', cursor: 'pointer', fontWeight: activeCohort === 2 ? 800 : 500 }}>● Erratic Spikers (30)</span>
                {activeCohort !== 'ALL' && (
                  <button
                    onClick={() => setActiveCohort('ALL')}
                    className="pill-badge pill-neutral"
                    style={{ cursor: 'pointer', border: '1px solid #38BDF8', color: '#38BDF8' }}
                  >
                    RESET TO ALL (150)
                  </button>
                )}
              </div>
            </div>

            <div style={{ height: '360px', width: '100%' }}>
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
                  <XAxis dataKey="pca_x" name="PCA Axis 1" stroke="#64748B" fontSize={11} tickLine={false} label={{ value: 'Principal Component 1 (Volume & Velocity)', position: 'insideBottom', offset: -10, fill: '#64748B', fontSize: 11 }} />
                  <YAxis dataKey="pca_y" name="PCA Axis 2" stroke="#64748B" fontSize={11} tickLine={false} label={{ value: 'Principal Component 2 (Volatility & Margin)', angle: -90, position: 'insideLeft', fill: '#64748B', fontSize: 11 }} />
                  <Tooltip
                    content={({ payload }) => {
                      if (!payload || !payload.length) return null;
                      const pt = payload[0].payload;
                      return (
                        <div style={{ backgroundColor: '#111827', border: '1px solid rgba(255,255,255,0.15)', borderRadius: '8px', padding: '10px 14px', fontSize: '12px' }}>
                          <div style={{ fontWeight: 700, color: '#F8FAFC', marginBottom: '4px' }}>{pt.medicine_name}</div>
                          <div style={{ color: CLUSTER_COLORS[pt.cluster_id], fontWeight: 600, marginBottom: '6px' }}>{pt.cluster_name}</div>
                          <div style={{ color: '#94A3B8', lineHeight: '1.4' }}>
                            <div>Daily Sales: <strong>{pt.daily_demand} units</strong></div>
                            <div>Volatility (CV): <strong>{pt.volatility_cv}</strong></div>
                            <div>Gross Margin: <strong>{pt.profit_margin_pct}%</strong></div>
                            <div>Selling Price: <strong>Rs. {pt.selling_price}</strong></div>
                          </div>
                        </div>
                      );
                    }}
                  />
                  <Scatter
                    data={data.sku_points}
                    onClick={(entry) => setSelectedSku(entry)}
                  >
                    {data.sku_points.map((entry, index) => {
                      const isHighlighted = activeCohort === 'ALL' || entry.cluster_id === activeCohort;
                      const fill = isHighlighted ? (CLUSTER_COLORS[entry.cluster_id] || '#38BDF8') : 'rgba(100, 116, 139, 0.18)';
                      return (
                        <Cell
                          key={`cell-${index}`}
                          fill={fill}
                          cursor="pointer"
                          opacity={isHighlighted ? 1.0 : 0.25}
                        />
                      );
                    })}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Interactive SKU Deep-Dive Inspector */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '18px' }}>
            {/* SKU Search List */}
            <div className="glass-panel" style={{ padding: '18px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
                <Search size={15} color="#38BDF8" />
                <div style={{ fontSize: '0.86rem', fontWeight: 700, color: '#F8FAFC' }}>
                  Formulary SKU Search
                </div>
              </div>
              <input
                type="text"
                placeholder="Search medicine by name or ID..."
                className="input-text"
                style={{ width: '100%', marginBottom: '12px' }}
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />

              <div style={{ maxHeight: '280px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                {filteredSkus.slice(0, 50).map((s) => {
                  const isSelected = selectedSku?.medicine_id === s.medicine_id;
                  return (
                    <div
                      key={s.medicine_id}
                      onClick={() => setSelectedSku(s)}
                      style={{
                        padding: '8px 12px',
                        borderRadius: '6px',
                        background: isSelected ? 'rgba(14, 165, 233, 0.15)' : 'transparent',
                        border: isSelected ? '1px solid rgba(14, 165, 233, 0.4)' : '1px solid transparent',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        fontSize: '0.8rem'
                      }}
                    >
                      <div>
                        <div style={{ fontWeight: isSelected ? 700 : 500, color: '#F1F5F9' }}>{s.medicine_name}</div>
                        <div style={{ fontSize: '0.7rem', color: '#64748B' }}>{s.medicine_id} • {s.category_name}</div>
                      </div>
                      <span className="pill-badge" style={{ color: CLUSTER_COLORS[s.cluster_id], border: `1px solid ${CLUSTER_COLORS[s.cluster_id]}` }}>
                        {s.cluster_name.split(' ')[0]}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Selected SKU Policy Card */}
            {selectedSku && (
              <div className="glass-panel" style={{ padding: '22px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
                    <div>
                      <div style={{ fontSize: '1.2rem', fontWeight: 800, color: '#F8FAFC' }}>
                        {selectedSku.medicine_name}
                      </div>
                      <div style={{ fontSize: '0.78rem', color: '#38BDF8', fontFamily: 'monospace' }}>
                        {selectedSku.medicine_id} • {selectedSku.category_name}
                      </div>
                    </div>
                    <span className="pill-badge" style={{ color: CLUSTER_COLORS[selectedSku.cluster_id], border: `1px solid ${CLUSTER_COLORS[selectedSku.cluster_id]}`, fontSize: '0.8rem', padding: '5px 12px' }}>
                      {selectedSku.cluster_name}
                    </span>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px', marginBottom: '20px' }}>
                    <div style={{ background: '#0F172A', padding: '12px', borderRadius: '8px' }}>
                      <div style={{ fontSize: '0.7rem', color: '#94A3B8' }}>Daily Demand</div>
                      <div style={{ fontSize: '1.1rem', fontWeight: 800, color: '#F1F5F9' }}>{selectedSku.daily_demand} u/d</div>
                    </div>
                    <div style={{ background: '#0F172A', padding: '12px', borderRadius: '8px' }}>
                      <div style={{ fontSize: '0.7rem', color: '#94A3B8' }}>Volatility (CV)</div>
                      <div style={{ fontSize: '1.1rem', fontWeight: 800, color: CLUSTER_COLORS[selectedSku.cluster_id] }}>{selectedSku.volatility_cv}</div>
                    </div>
                    <div style={{ background: '#0F172A', padding: '12px', borderRadius: '8px' }}>
                      <div style={{ fontSize: '0.7rem', color: '#94A3B8' }}>Gross Margin</div>
                      <div style={{ fontSize: '1.1rem', fontWeight: 800, color: '#34D399' }}>{selectedSku.profit_margin_pct}%</div>
                    </div>
                    <div style={{ background: '#0F172A', padding: '12px', borderRadius: '8px' }}>
                      <div style={{ fontSize: '0.7rem', color: '#94A3B8' }}>Unit Price</div>
                      <div style={{ fontSize: '1.1rem', fontWeight: 800, color: '#F1F5F9' }}>Rs. {selectedSku.selling_price}</div>
                    </div>
                  </div>
                </div>

                <div style={{ background: 'rgba(15, 23, 42, 0.7)', border: '1px solid var(--border-subtle)', borderRadius: '8px', padding: '14px' }}>
                  <div style={{ fontSize: '0.76rem', fontWeight: 700, color: '#38BDF8', textTransform: 'uppercase', marginBottom: '4px' }}>
                    Automated Inventory Replenishment Directive
                  </div>
                  <div style={{ fontSize: '0.82rem', color: '#94A3B8', lineHeight: '1.45' }}>
                    {selectedSku.cluster_id === 0 && 'High daily turnover item: Configure JIT automated daily replenishment orders with minimal safety stock buffers to optimize working capital.'}
                    {selectedSku.cluster_id === 1 && 'Seasonal demand wave: Monitor quarterly weather indicators and initiate pre-season warehouse stock buildup 30 days prior to peak.'}
                    {selectedSku.cluster_id === 2 && 'Erratic high-margin SKU: Maintain higher safety buffer stock to capture sudden customer demand, while strictly enforcing FEFO dispatch to avoid batch expiry write-offs.'}
                  </div>
                </div>
              </div>
            )}
          </div>
        </>
      ) : null}
    </div>
  );
}
