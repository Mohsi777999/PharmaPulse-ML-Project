import React from 'react';

export default function HeroBanner({ tag, title, subtitle }) {
  return (
    <div className="hero-banner">
      {tag && (
        <div className="hero-tag">
          <span>●</span>
          <span>{tag}</span>
        </div>
      )}
      <h1 className="hero-title">{title}</h1>
      <p className="hero-subtitle">{subtitle}</p>
    </div>
  );
}
