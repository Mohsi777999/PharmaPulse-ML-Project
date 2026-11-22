import React from 'react';

export default function KpiCard({ title, value, subtext, icon, badge, badgeType = 'neutral' }) {
  const badgeClasses = {
    critical: 'pill-critical',
    warning: 'pill-warning',
    success: 'pill-success',
    cyan: 'pill-cyan',
    neutral: 'pill-neutral'
  };

  return (
    <div className="kpi-card">
      <div className="kpi-header">
        <span className="kpi-title">{title}</span>
        {icon && <span className="kpi-icon">{icon}</span>}
      </div>
      <div className="kpi-value">{value}</div>
      <div className="kpi-subtext">
        {badge && (
          <span className={`pill-badge ${badgeClasses[badgeType] || 'pill-neutral'}`}>
            {badge}
          </span>
        )}
        <span>{subtext}</span>
      </div>
    </div>
  );
}
