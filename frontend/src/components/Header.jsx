import React from 'react';

export default function Header({ activeView }) {
  const titles = {
    overview: 'Operations Overview & Intelligence Portal',
    forecasting: 'Module 1 • Demand Forecasting Engine',
    expiry: 'Module 2 • Batch Expiry Risk & Capital Exposure',
    stockout: 'Module 3 • Stockout & Replenishment Pipeline Risk',
    clustering: 'Module 4 • Medicine Sales Behaviour Clustering'
  };

  return (
    <header style={{
      height: '64px',
      backgroundColor: 'var(--bg-canvas)',
      borderBottom: '1px solid var(--border-subtle)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '0 28px',
      position: 'sticky',
      top: 0,
      zIndex: 10
    }}>
      <div>
        <h2 style={{ fontSize: '1.05rem', fontWeight: 700, color: '#F8FAFC' }}>
          {titles[activeView] || 'Intelligence Portal'}
        </h2>
      </div>
    </header>
  );
}
