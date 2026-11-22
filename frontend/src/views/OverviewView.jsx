import React, { useState, useEffect } from 'react';
import HeroBanner from '../components/HeroBanner';
import KpiCard from '../components/KpiCard';
import { api } from '../services/api';
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend
} from 'recharts';
import {
  AlertCircle, TrendingUp, TrendingDown, ShieldCheck,
  Filter, RefreshCw, Search, Calendar, DollarSign, Activity, Layers
} from 'lucide-react';

export default function OverviewView() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [timeRange, setTimeRange] = useState('ALL');
  const [selectedCategory, setSelectedCategory] = useState('ALL');
  const [chartMetric, setChartMetric] = useState('units'); // 'units' or 'revenue'
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    loadOverview(timeRange, selectedCategory);
  }, [timeRange, selectedCategory]);

  const loadOverview = async (range, category) => {
    try {
      setLoading(true);
      const res = await api.getOverview(range, category);
      setData(res);
    } catch (err) {
      console.error('Failed to load overview data:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = async () => {
    try {
      setRefreshing(true);
      const res = await api.getOverview(timeRange, selectedCategory);
      setData(res);
    } catch (err) {
      console.error('Failed to refresh overview data:', err);
    } finally {
      setRefreshing(false);
    }
  };

  const handleResetFilters = () => {
    setTimeRange('ALL');
    setSelectedCategory('ALL');
    setSearchQuery('');
  };

  if (loading && !data) {
    return (
      <div style={{ padding: '60px 20px', textAlign: 'center', color: '#94A3B8' }}>
        <div className="pulse-dot" style={{ width: '14px', height: '14px', margin: '0 auto 16px auto' }}></div>
        <div style={{ fontSize: '1.05rem', fontWeight: 600, color: '#F1F5F9' }}>Loading Operations Intelligence...</div>
        <div style={{ fontSize: '0.82rem', marginTop: '6px' }}>Synthesizing multi-horizon demand, expiry, and stockout signals</div>
      </div>
    );
  }

  const {
    kpis = {},
    available_categories = [],
    monthly_velocity = [],
    categories = [],
    critical_expiry_batches = [],
    urgent_stockouts = [],
    top_growing = [],
    top_declining = []
  } = data || {};

  const COLORS = ['#38BDF8', '#34D399', '#818CF8', '#FBBF24', '#F87171', '#A78BFA'];

  // Filter alerts by search query if present
  const query = searchQuery.trim().toLowerCase();
  const filteredExpiryBatches = critical_expiry_batches.filter(b =>
    !query ||
    (b.batch_id && b.batch_id.toLowerCase().includes(query)) ||
    (b.medicine_name && b.medicine_name.toLowerCase().includes(query)) ||
    (b.category_name && b.category_name.toLowerCase().includes(query))
  );

  const filteredStockouts = urgent_stockouts.filter(s =>
    !query ||
    (s.medicine_name && s.medicine_name.toLowerCase().includes(query)) ||
    (s.purchase_order_id && s.purchase_order_id.toLowerCase().includes(query)) ||
    (s.supplier_id && s.supplier_id.toLowerCase().includes(query))
  );

  const timeRangeLabels = {
    ALL: '36 Months',
    '12M': 'Last 12 Months',
    '6M': 'Last 6 Months',
    '30D': 'Last 30 Days'
  };

  const isFiltered = timeRange !== 'ALL' || selectedCategory !== 'ALL' || searchQuery !== '';

  return (
    <div>
      <HeroBanner
        tag="OPERATIONS INTELLIGENCE • BRANCH 1 REGIONAL HUB"
        title="PharmaPulse — Executive Operations Overview"
        subtitle="Dynamic decision support cockpit delivering unified visibility across formulary dispensing velocity, gross revenue margins, batch expiry liabilities, vendor replenishment vulnerabilities, and SKU trajectory dynamics."
      />

      {/* Dynamic Interactive Filter & Control Ribbon */}
      <div
        className="glass-panel"
        style={{
          padding: '16px 20px',
          marginBottom: '20px',
          display: 'flex',
          flexWrap: 'wrap',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '14px'
        }}
      >
        {/* Left side: Time Horizon & Category Selectors */}
        <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '14px' }}>
          {/* Time Range Pills */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Calendar size={15} color="#38BDF8" style={{ marginRight: '4px' }} />
            <span style={{ fontSize: '0.78rem', color: '#94A3B8', fontWeight: 600 }}>Horizon:</span>
            {[
              { id: 'ALL', label: 'All 36M' },
              { id: '12M', label: '12 Months' },
              { id: '6M', label: '6 Months' },
              { id: '30D', label: '30 Days' }
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setTimeRange(tab.id)}
                className={`btn-secondary ${timeRange === tab.id ? 'btn-primary' : ''}`}
                style={{
                  padding: '5px 11px',
                  fontSize: '0.74rem',
                  borderRadius: '6px',
                  fontWeight: timeRange === tab.id ? 700 : 500,
                  cursor: 'pointer'
                }}
              >
                {tab.label}
              </button>
            ))}
          </div>

          <div style={{ width: '1px', height: '22px', backgroundColor: 'rgba(255,255,255,0.1)' }} />

          {/* Category Dropdown */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Filter size={15} color="#34D399" />
            <span style={{ fontSize: '0.78rem', color: '#94A3B8', fontWeight: 600 }}>Category:</span>
            <select
              className="input-select"
              value={selectedCategory}
              onChange={e => setSelectedCategory(e.target.value)}
              style={{
                padding: '5px 10px',
                fontSize: '0.78rem',
                minWidth: '180px',
                borderRadius: '6px'
              }}
            >
              {available_categories.length > 0 ? (
                available_categories.map(cat => (
                  <option key={cat.category_id} value={cat.category_id}>
                    {cat.category_name}
                  </option>
                ))
              ) : (
                <option value="ALL">All Categories</option>
              )}
            </select>
          </div>

          {/* Quick Alert Filter Search */}
          <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
            <Search size={14} color="#64748B" style={{ position: 'absolute', left: '9px' }} />
            <input
              type="text"
              className="input-text"
              placeholder="Search SKU or Batch..."
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              style={{
                paddingLeft: '28px',
                paddingTop: '5px',
                paddingBottom: '5px',
                fontSize: '0.76rem',
                width: '170px',
                borderRadius: '6px'
              }}
            />
          </div>
        </div>

        {/* Right side: Actions & Status Badge */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          {isFiltered && (
            <button
              onClick={handleResetFilters}
              className="btn-secondary"
              style={{
                padding: '5px 10px',
                fontSize: '0.74rem',
                color: '#F87171',
                borderColor: 'rgba(248, 113, 113, 0.3)',
                borderRadius: '6px',
                cursor: 'pointer'
              }}
              title="Reset to default scope"
            >
              Reset Filters
            </button>
          )}

          <button
            onClick={handleRefresh}
            disabled={refreshing}
            className="btn-secondary"
            style={{
              padding: '5px 12px',
              fontSize: '0.76rem',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              borderRadius: '6px',
              cursor: refreshing ? 'not-allowed' : 'pointer'
            }}
            title="Refresh Operations Overview"
          >
            <RefreshCw size={13} className={refreshing ? 'spin-animation' : ''} />
            <span>{refreshing ? 'Updating...' : 'Refresh'}</span>
          </button>

          <span className="pill-badge pill-cyan" style={{ fontSize: '0.72rem', padding: '4px 9px' }}>
            {timeRangeLabels[timeRange]} • {kpis.selected_category_name || 'All Formulary'}
          </span>
        </div>
      </div>

      {/* 5 Dynamic KPI Metric Row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '14px', marginBottom: '24px' }}>
        <KpiCard
          title="Active Formulary"
          value={`${kpis.active_medicines || 0} SKUs`}
          subtext={kpis.selected_category_name ? `${kpis.selected_category_name}` : `Distributed in ${kpis.total_categories || 6} categories`}
          icon="💊"
          badge={kpis.active_medicines === 150 ? '100% Active' : `${kpis.active_medicines} Filtered`}
          badgeType="neutral"
        />
        <KpiCard
          title={`${timeRange === 'ALL' ? '3-Year' : timeRangeLabels[timeRange]} Revenue`}
          value={`Rs. ${kpis.gross_revenue_m || 0}M`}
          subtext={`Gross margin: ${kpis.gross_margin_pct || 0}% (Rs. ${kpis.gross_profit_m || 0}M)`}
          icon="💰"
          badge={kpis.gross_margin_pct >= 30 ? 'Healthy Margin' : 'Moderate'}
          badgeType="success"
        />
        <KpiCard
          title="Dispensed Volume"
          value={(kpis.units_sold || 0).toLocaleString()}
          subtext={`Fulfillment rate: ${kpis.fulfillment_rate_pct || 0}%`}
          icon="📦"
          badge={`${kpis.fulfillment_rate_pct || 0}% Met`}
          badgeType={kpis.fulfillment_rate_pct >= 95 ? 'success' : 'warning'}
        />
        <KpiCard
          title="Expired Batches"
          value={kpis.expired_batches ?? 0}
          subtext={`${(kpis.expired_units || 0).toLocaleString()} units written off`}
          icon="⚠️"
          badge={`${kpis.expired_loss_pct ?? 0}% Loss`}
          badgeType={kpis.expired_batches > 0 ? 'critical' : 'success'}
        />
        <KpiCard
          title="Stockout Incidents"
          value={`${kpis.stockout_pct || 0}%`}
          subtext={`${(kpis.stockout_days || 0).toLocaleString()} medicine-days`}
          icon="⚡"
          badge={`${kpis.stockout_pct || 0}% Rate`}
          badgeType={kpis.stockout_pct > 3.5 ? 'critical' : 'warning'}
        />
      </div>

      {/* Dual Charts Row: Dispensing Velocity & Therapeutic Categories */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '18px', marginBottom: '24px' }}>
        {/* Dispensing Velocity Trend */}
        <div className="glass-panel" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
            <div>
              <div style={{ fontSize: '0.92rem', fontWeight: 700, color: '#F8FAFC' }}>
                {timeRange === '30D' ? 'Daily Dispensing Trajectory (December 2026)' : 'Monthly Dispensing Velocity Trajectory'}
              </div>
              <div style={{ fontSize: '0.74rem', color: '#94A3B8' }}>
                {timeRangeLabels[timeRange]} {selectedCategory !== 'ALL' ? `• ${kpis.selected_category_name}` : 'aggregate trajectory'}
              </div>
            </div>

            {/* Metric Toggle */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div style={{ display: 'flex', backgroundColor: 'rgba(255,255,255,0.06)', borderRadius: '6px', padding: '2px' }}>
                <button
                  onClick={() => setChartMetric('units')}
                  style={{
                    padding: '3px 8px',
                    fontSize: '0.7rem',
                    border: 'none',
                    borderRadius: '4px',
                    background: chartMetric === 'units' ? '#38BDF8' : 'transparent',
                    color: chartMetric === 'units' ? '#0F172A' : '#94A3B8',
                    fontWeight: chartMetric === 'units' ? 700 : 500,
                    cursor: 'pointer'
                  }}
                >
                  Units
                </button>
                <button
                  onClick={() => setChartMetric('revenue')}
                  style={{
                    padding: '3px 8px',
                    fontSize: '0.7rem',
                    border: 'none',
                    borderRadius: '4px',
                    background: chartMetric === 'revenue' ? '#34D399' : 'transparent',
                    color: chartMetric === 'revenue' ? '#0F172A' : '#94A3B8',
                    fontWeight: chartMetric === 'revenue' ? 700 : 500,
                    cursor: 'pointer'
                  }}
                >
                  Revenue (PKR k)
                </button>
              </div>
              <span className="pill-badge pill-cyan" style={{ fontSize: '0.7rem' }}>
                {monthly_velocity.length} DATA POINTS
              </span>
            </div>
          </div>

          <div style={{ height: '240px', width: '100%' }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={monthly_velocity}>
                <defs>
                  <linearGradient id="colorUnits" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={chartMetric === 'units' ? '#38BDF8' : '#34D399'} stopOpacity={0.35}/>
                    <stop offset="95%" stopColor={chartMetric === 'units' ? '#38BDF8' : '#34D399'} stopOpacity={0.0}/>
                  </linearGradient>
                </defs>
                <XAxis dataKey="month" stroke="#64748B" fontSize={11} tickLine={false} />
                <YAxis
                  stroke="#64748B"
                  fontSize={11}
                  tickLine={false}
                  tickFormatter={val => chartMetric === 'revenue' ? `Rs.${val}k` : val.toLocaleString()}
                />
                <Tooltip
                  contentStyle={{ backgroundColor: '#111827', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '8px', fontSize: '12px' }}
                  labelStyle={{ color: '#F1F5F9', fontWeight: 600 }}
                  formatter={(val, name) => [
                    name === 'Revenue (PKR k)' ? `Rs. ${Number(val).toLocaleString()}k` : Number(val).toLocaleString(),
                    name
                  ]}
                />
                <Area
                  type="monotone"
                  dataKey={chartMetric === 'units' ? 'units' : 'revenue_k'}
                  stroke={chartMetric === 'units' ? '#38BDF8' : '#34D399'}
                  strokeWidth={2}
                  fillOpacity={1}
                  fill="url(#colorUnits)"
                  name={chartMetric === 'units' ? 'Dispensed Units' : 'Revenue (PKR k)'}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Therapeutic Categories / Top Medicines Breakdown */}
        <div className="glass-panel" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '4px' }}>
            <div style={{ fontSize: '0.92rem', fontWeight: 700, color: '#F8FAFC' }}>
              {selectedCategory === 'ALL' ? 'Category Volume Share' : `Top SKUs in Category`}
            </div>
            <span className="pill-badge pill-neutral" style={{ fontSize: '0.68rem' }}>
              {selectedCategory === 'ALL' ? '6 Cohorts' : `${categories.length} SKUs`}
            </span>
          </div>
          <div style={{ fontSize: '0.74rem', color: '#94A3B8', marginBottom: '14px' }}>
            {selectedCategory === 'ALL' ? 'Distribution across therapeutic classes' : `${kpis.selected_category_name} SKU contribution`}
          </div>

          <div style={{ height: '220px', width: '100%' }}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={categories}
                  dataKey="units_sold"
                  nameKey="category_name"
                  cx="50%"
                  cy="50%"
                  outerRadius={75}
                  innerRadius={44}
                  paddingAngle={3}
                >
                  {categories.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ backgroundColor: '#111827', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '8px', fontSize: '12px' }}
                  formatter={(val, name, props) => [
                    `${Number(val).toLocaleString()} units (${props.payload.share_pct || 0}%) • Rs. ${props.payload.revenue_m || 0}M`,
                    name
                  ]}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Critical AI Alerts Row */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '18px', marginBottom: '24px' }}>
        {/* Critical Batches Approaching Expiry */}
        <div className="glass-panel" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <AlertCircle size={18} color="#F87171" />
              <div style={{ fontSize: '0.9rem', fontWeight: 700, color: '#F8FAFC' }}>
                Critical Batches Approaching Expiry
              </div>
            </div>
            <span className="pill-badge pill-critical">
              {filteredExpiryBatches.length} ACTION REQUIRED
            </span>
          </div>
          <div style={{ fontSize: '0.76rem', color: '#94A3B8', marginBottom: '14px' }}>
            Active perishable batches with &le; 30 days to expiry and projected unsold financial exposure.
          </div>

          <div className="data-table-container" style={{ maxHeight: '260px', overflowY: 'auto' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Batch ID</th>
                  <th>Medicine</th>
                  <th>Days Left</th>
                  <th>Units</th>
                  <th>Exposure (PKR)</th>
                </tr>
              </thead>
              <tbody>
                {filteredExpiryBatches.length > 0 ? (
                  filteredExpiryBatches.slice(0, 6).map((b, i) => (
                    <tr key={i}>
                      <td style={{ fontFamily: 'monospace', color: '#38BDF8' }}>{b.batch_id}</td>
                      <td style={{ maxWidth: '170px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }} title={b.medicine_name}>
                        {b.medicine_name}
                      </td>
                      <td>
                        <span className={`pill-badge ${b.days_to_expiry <= 7 ? 'pill-critical' : 'pill-warning'}`}>
                          {b.days_to_expiry}d
                        </span>
                      </td>
                      <td>{Number(b.remaining_batch_quantity || 0).toLocaleString()}</td>
                      <td style={{ color: '#F87171', fontWeight: 600 }}>
                        Rs. {Number(b.financial_exposure || 0).toLocaleString()}
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan="5" style={{ textAlign: 'center', color: '#64748B', padding: '24px' }}>
                      No critical expiry risk batches detected for the current filter scope.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Urgent Stockout Replenishment Alerts */}
        <div className="glass-panel" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <ShieldCheck size={18} color="#FBBF24" />
              <div style={{ fontSize: '0.9rem', fontWeight: 700, color: '#F8FAFC' }}>
                Stockout Replenishment Alerts
              </div>
            </div>
            <span className="pill-badge pill-warning">
              {filteredStockouts.length} EARLY WARNINGS
            </span>
          </div>
          <div style={{ fontSize: '0.76rem', color: '#94A3B8', marginBottom: '14px' }}>
            Medicines where usable stock runs out before expected supplier delivery arrives.
          </div>

          <div className="data-table-container" style={{ maxHeight: '260px', overflowY: 'auto' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>PO ID</th>
                  <th>Medicine</th>
                  <th>Supply Days</th>
                  <th>Lead Time</th>
                  <th>Supplier</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {filteredStockouts.length > 0 ? (
                  filteredStockouts.slice(0, 6).map((s, i) => (
                    <tr key={i}>
                      <td style={{ fontFamily: 'monospace', color: '#38BDF8', fontSize: '0.74rem' }}>
                        {s.purchase_order_id}
                      </td>
                      <td style={{ maxWidth: '160px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }} title={s.medicine_name}>
                        {s.medicine_name}
                      </td>
                      <td>
                        <span className={`pill-badge ${s.days_of_supply <= 2 ? 'pill-critical' : 'pill-warning'}`}>
                          {s.days_of_supply}d
                        </span>
                      </td>
                      <td>{s.expected_lead_time_days}d</td>
                      <td style={{ color: '#94A3B8', fontSize: '0.76rem' }}>{s.supplier_id}</td>
                      <td>
                        <span className={`pill-badge ${s.urgency_status === 'CRITICAL' ? 'pill-critical' : 'pill-warning'}`}>
                          {s.urgency_status || 'EXPEDITE PO'}
                        </span>
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan="6" style={{ textAlign: 'center', color: '#64748B', padding: '24px' }}>
                      No active replenishment stockout risks identified for the current filter scope.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Growth & Decline Dynamics */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '18px' }}>
        <div className="glass-panel" style={{ padding: '18px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <TrendingUp size={16} color="#34D399" />
              <div style={{ fontSize: '0.86rem', fontWeight: 700, color: '#F8FAFC' }}>
                Top Demand Velocity Gainers (Last 30 Days)
              </div>
            </div>
            <span className="pill-badge pill-success">{top_growing.length} SKUs</span>
          </div>
          <div className="data-table-container">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Medicine</th>
                  <th>Growth %</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {top_growing && top_growing.length > 0 ? (
                  top_growing.map((m, i) => (
                    <tr key={i}>
                      <td style={{ maxWidth: '240px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {m.medicine_name}
                      </td>
                      <td style={{ color: '#34D399', fontWeight: 700 }}>+{m.growth_pct}%</td>
                      <td><span className="pill-badge pill-success">EXPANDING</span></td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan="3" style={{ textAlign: 'center', color: '#64748B', padding: '18px' }}>
                      No expanding SKUs identified for this selection.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '18px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <TrendingDown size={16} color="#F87171" />
              <div style={{ fontSize: '0.86rem', fontWeight: 700, color: '#F8FAFC' }}>
                Demand Contraction Watchlist (Last 30 Days)
              </div>
            </div>
            <span className="pill-badge pill-neutral">{top_declining.length} SKUs</span>
          </div>
          <div className="data-table-container">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Medicine</th>
                  <th>Decline %</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {top_declining && top_declining.length > 0 ? (
                  top_declining.map((m, i) => (
                    <tr key={i}>
                      <td style={{ maxWidth: '240px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {m.medicine_name}
                      </td>
                      <td style={{ color: '#F87171', fontWeight: 700 }}>{m.decline_pct}%</td>
                      <td><span className="pill-badge pill-neutral">COOLING</span></td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan="3" style={{ textAlign: 'center', color: '#64748B', padding: '18px' }}>
                      No contracting SKUs identified for this selection.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
