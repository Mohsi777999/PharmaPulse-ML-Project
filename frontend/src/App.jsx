import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import OverviewView from './views/OverviewView';
import ForecastingView from './views/ForecastingView';
import ExpiryView from './views/ExpiryView';
import StockoutView from './views/StockoutView';
import ClusteringView from './views/ClusteringView';

export default function App() {
  const [activeView, setActiveView] = useState('overview');

  const renderActiveView = () => {
    switch (activeView) {
      case 'overview':
        return <OverviewView />;
      case 'forecasting':
        return <ForecastingView />;
      case 'expiry':
        return <ExpiryView />;
      case 'stockout':
        return <StockoutView />;
      case 'clustering':
        return <ClusteringView />;
      default:
        return <OverviewView />;
    }
  };

  return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: 'var(--bg-canvas)' }}>
      {/* Sidebar Navigation */}
      <Sidebar
        activeView={activeView}
        setActiveView={setActiveView}
      />

      {/* Main Content Area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        <Header activeView={activeView} />

        <main style={{ flex: 1, padding: '24px 28px', maxWidth: '1440px', width: '100%', margin: '0 auto' }}>
          {renderActiveView()}
        </main>
      </div>
    </div>
  );
}
