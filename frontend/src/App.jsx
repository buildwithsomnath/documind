import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import ChatInterface from './components/ChatInterface';
import DocumentUpload from './components/DocumentUpload';
import DocumentList from './components/DocumentList';
import EvaluationDashboard from './components/EvaluationDashboard';
import { api } from './services/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('chat');
  const [healthStatus, setHealthStatus] = useState('connecting');
  const [documents, setDocuments] = useState([]);
  const [loadingDocs, setLoadingDocs] = useState(false);

  const fetchHealth = async () => {
    try {
      const data = await api.getHealth();
      if (data.status === 'ok') {
        setHealthStatus('connected');
      } else {
        setHealthStatus('error');
      }
    } catch (err) {
      setHealthStatus('error');
    }
  };

  const fetchDocuments = async () => {
    setLoadingDocs(true);
    try {
      const data = await api.getDocuments();
      setDocuments(data.documents || []);
    } catch (err) {
      console.error('Failed to fetch documents:', err);
    } finally {
      setLoadingDocs(false);
    }
  };

  useEffect(() => {
    fetchHealth();
    fetchDocuments();

    // Periodic health poll every 30 seconds
    const interval = setInterval(fetchHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        healthStatus={healthStatus}
      />

      <main className="container" style={{ flex: 1, padding: '1.5rem 1.5rem' }}>
        {activeTab === 'chat' && (
          <ChatInterface documents={documents} />
        )}

        {activeTab === 'documents' && (
          <div>
            <DocumentUpload onUploadSuccess={fetchDocuments} />
            <DocumentList documents={documents} onRefresh={fetchDocuments} />
          </div>
        )}

        {activeTab === 'evaluation' && (
          <EvaluationDashboard />
        )}
      </main>
    </div>
  );
}
