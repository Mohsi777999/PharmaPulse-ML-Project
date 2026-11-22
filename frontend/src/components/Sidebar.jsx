import React from 'react';
import { 
  LayoutDashboard, 
  TrendingUp, 
  AlertTriangle, 
  ShieldAlert, 
  PieChart 
} from 'lucide-react';

export default function Sidebar({ activeView, setActiveView, systemStatus }) {
  const navItems = [
    { id: 'overview', label: 'Operations Overview', icon: LayoutDashboard },
    { id: 'forecasting', label: 'Demand Forecasting', icon: TrendingUp },
    { id: 'expiry', label: 'Expiry Intelligence', icon: AlertTriangle },
    { id: 'stockout', label: 'Stockout Intelligence', icon: ShieldAlert },
    { id: 'clustering', label: 'Sales Behaviour', icon: PieChart },
  ];

  return (
    <aside style={{
      width: '260px',
      backgroundColor: 'var(--bg-sidebar)',
      borderRight: '1px solid var(--border-subtle)',
      display: 'flex',
      flexDirection: 'column',
      height: '100vh',
      position: 'sticky',
      top: 0,
      flexShrink: 0
    }}>
      {/* Brand Header */}
      <div style={{ padding: '24px 20px', borderBottom: '1px solid var(--border-subtle)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            width: '38px',
            height: '38px',
            borderRadius: '10px',
            background: 'linear-gradient(135deg, #0EA5E9 0%, #0369A1 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '1.2rem',
            boxShadow: '0 0 16px rgba(14, 165, 233, 0.4)'
          }}>
            💊
          </div>
          <div>
            <div style={{ fontWeight: 800, fontSize: '1.1rem', letterSpacing: '-0.02em', color: '#F8FAFC' }}>
              PHARMAPULSE
            </div>
            <div style={{ fontSize: '0.68rem', color: '#38BDF8', fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase' }}>
              Operations Intelligence
            </div>
          </div>
        </div>
      </div>

      {/* Navigation List */}
      <nav style={{ padding: '16px 12px', flex: 1, overflowY: 'auto' }}>
        <div style={{ fontSize: '0.68rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', padding: '0 8px 8px 8px' }}>
          Operations Modules
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeView === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveView(item.id)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px',
                  padding: '10px 14px',
                  borderRadius: '8px',
                  border: 'none',
                  background: isActive ? 'linear-gradient(90deg, rgba(14, 165, 233, 0.15) 0%, rgba(14, 165, 233, 0.05) 100%)' : 'transparent',
                  color: isActive ? '#38BDF8' : '#94A3B8',
                  fontWeight: isActive ? 700 : 500,
                  fontSize: '0.84rem',
                  cursor: 'pointer',
                  textAlign: 'left',
                  transition: 'all 0.15s ease',
                  borderLeft: isActive ? '3px solid #38BDF8' : '3px solid transparent'
                }}
                onMouseEnter={(e) => {
                  if (!isActive) e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.04)';
                }}
                onMouseLeave={(e) => {
                  if (!isActive) e.currentTarget.style.backgroundColor = 'transparent';
                }}
              >
                <Icon size={17} color={isActive ? '#38BDF8' : '#64748B'} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </div>
      </nav>

      {/* Operational Parameters & Reconciled Badge */}
      <div style={{ padding: '16px', borderTop: '1px solid var(--border-subtle)', background: 'rgba(0,0,0,0.2)' }}>
        <div style={{
          background: 'rgba(16, 185, 129, 0.1)',
          border: '1px solid rgba(16, 185, 129, 0.25)',
          borderRadius: '8px',
          padding: '10px',
          marginBottom: '12px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.74rem', fontWeight: 700, color: '#34D399' }}>
            <span className="pulse-dot"></span>
            RECONCILIATION: 100% PASS
          </div>
          <div style={{ fontSize: '0.68rem', color: '#94A3B8', marginTop: '2px' }}>
            0 balance discrepancies across 164,250 medicine-days.
          </div>
        </div>

        <div style={{ fontSize: '0.7rem', color: '#64748B', lineHeight: '1.4' }}>
          <div><strong>Facility:</strong> Branch 1 Regional Hub</div>
          <div><strong>Dispensing:</strong> Batch FEFO</div>
        </div>
      </div>
    </aside>
  );
}
