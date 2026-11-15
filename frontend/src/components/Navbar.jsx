import React from 'react';
import { Database, MessageSquare, Files, BarChart3, CheckCircle, AlertCircle } from 'lucide-react';

export default function Navbar({ activeTab, setActiveTab, healthStatus }) {
  const tabs = [
    { id: 'chat', label: 'Q&A Chat', icon: MessageSquare },
    { id: 'documents', label: 'Documents', icon: Files },
    { id: 'evaluation', label: 'RAG Evaluation', icon: BarChart3 },
  ];

  return (
    <header style={{
      borderBottom: '1px solid var(--border-color)',
      backgroundColor: 'rgba(17, 24, 39, 0.8)',
      backdropFilter: 'blur(12px)',
      position: 'sticky',
      top: 0,
      zIndex: 50,
      padding: '0.75rem 0'
    }}>
      <div className="container" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        {/* Brand */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div style={{
            width: '38px',
            height: '38px',
            borderRadius: '10px',
            background: 'linear-gradient(135deg, #6366f1 0%, #a855f7 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#fff',
            boxShadow: '0 4px 12px rgba(99, 102, 241, 0.3)'
          }}>
            <Database size={20} />
          </div>
          <div>
            <h1 style={{ fontSize: '1.25rem', fontWeight: 700, letterSpacing: '-0.02em' }}>
              Docu<span style={{ color: 'var(--primary)' }}>Mind</span>
            </h1>
            <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Enterprise Hybrid RAG Platform</p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav style={{ display: 'flex', gap: '0.5rem', background: 'var(--bg-secondary)', padding: '0.3rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  padding: '0.5rem 1rem',
                  border: 'none',
                  borderRadius: 'var(--radius-sm)',
                  fontSize: '0.85rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                  background: isActive ? 'var(--primary)' : 'transparent',
                  color: isActive ? '#fff' : 'var(--text-muted)'
                }}
              >
                <Icon size={16} />
                {tab.label}
              </button>
            );
          })}
        </nav>

        {/* System Health Indicator */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          {healthStatus === 'connected' ? (
            <span className="badge badge-green">
              <CheckCircle size={12} />
              pgvector & Gemini Ready
            </span>
          ) : (
            <span className="badge badge-amber">
              <AlertCircle size={12} />
              Connecting to Backend...
            </span>
          )}
        </div>
      </div>
    </header>
  );
}
