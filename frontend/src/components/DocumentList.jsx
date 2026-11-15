import React, { useState } from 'react';
import { FileText, Trash2, Eye, X, Layers, Calendar } from 'lucide-react';
import { api } from '../services/api';

export default function DocumentList({ documents, onRefresh }) {
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [chunksLoading, setChunksLoading] = useState(false);
  const [docChunks, setDocChunks] = useState([]);
  const [deletingId, setDeletingId] = useState(null);

  const handleViewChunks = async (doc) => {
    setSelectedDoc(doc);
    setChunksLoading(true);
    try {
      const data = await api.getDocument(doc.id);
      setDocChunks(data.chunks || []);
    } catch (err) {
      console.error('Failed to load chunks:', err);
    } finally {
      setChunksLoading(false);
    }
  };

  const handleDelete = async (docId, filename) => {
    if (!window.confirm(`Are you sure you want to delete "${filename}" and its vector embeddings?`)) {
      return;
    }
    setDeletingId(docId);
    try {
      await api.deleteDocument(docId);
      if (onRefresh) onRefresh();
    } catch (err) {
      alert(`Delete failed: ${err.message}`);
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
        <div>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 600 }}>Indexed Knowledge Base</h3>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            {documents.length} document{documents.length !== 1 ? 's' : ''} stored in PostgreSQL
          </p>
        </div>
        <button onClick={onRefresh} className="btn-secondary" style={{ padding: '0.4rem 0.8rem', fontSize: '0.8rem' }}>
          Refresh
        </button>
      </div>

      {documents.length === 0 ? (
        <div style={{ padding: '3rem 1rem', textAlign: 'center', color: 'var(--text-muted)' }}>
          <FileText size={40} style={{ opacity: 0.3, marginBottom: '0.75rem' }} />
          <p style={{ fontWeight: 500 }}>No documents ingested yet.</p>
          <p style={{ fontSize: '0.8rem', marginTop: '0.25rem' }}>Upload a PDF or DOCX file to begin.</p>
        </div>
      ) : (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.85rem' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border-color)', color: 'var(--text-muted)' }}>
                <th style={{ padding: '0.75rem 0.5rem' }}>Document</th>
                <th style={{ padding: '0.75rem 0.5rem' }}>Format</th>
                <th style={{ padding: '0.75rem 0.5rem' }}>Pages</th>
                <th style={{ padding: '0.75rem 0.5rem' }}>Chunks</th>
                <th style={{ padding: '0.75rem 0.5rem' }}>Status</th>
                <th style={{ padding: '0.75rem 0.5rem', textAlign: 'right' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {documents.map((doc) => (
                <tr key={doc.id} style={{ borderBottom: '1px solid var(--border-subtle)', transition: 'background 0.2s' }}>
                  <td style={{ padding: '0.75rem 0.5rem', fontWeight: 500 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <FileText size={16} color="var(--primary)" />
                      <span>{doc.filename}</span>
                    </div>
                  </td>
                  <td style={{ padding: '0.75rem 0.5rem' }}>
                    <span className="badge badge-indigo" style={{ textTransform: 'uppercase' }}>
                      {doc.file_type}
                    </span>
                  </td>
                  <td style={{ padding: '0.75rem 0.5rem' }}>{doc.total_pages || '-'}</td>
                  <td style={{ padding: '0.75rem 0.5rem' }}>{doc.total_chunks}</td>
                  <td style={{ padding: '0.75rem 0.5rem' }}>
                    <span className={`badge ${doc.status === 'completed' ? 'badge-green' : 'badge-amber'}`}>
                      {doc.status}
                    </span>
                  </td>
                  <td style={{ padding: '0.75rem 0.5rem', textAlign: 'right' }}>
                    <div style={{ display: 'inline-flex', gap: '0.5rem' }}>
                      <button
                        onClick={() => handleViewChunks(doc)}
                        className="btn-secondary"
                        style={{ padding: '0.35rem 0.6rem', fontSize: '0.75rem' }}
                        title="View Chunks"
                      >
                        <Eye size={14} /> Chunks
                      </button>
                      <button
                        onClick={() => handleDelete(doc.id, doc.filename)}
                        disabled={deletingId === doc.id}
                        className="btn-secondary"
                        style={{ padding: '0.35rem 0.6rem', fontSize: '0.75rem', color: '#f87171' }}
                        title="Delete Document"
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Chunk Viewer Modal */}
      {selectedDoc && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0, 0, 0, 0.75)',
          backdropFilter: 'blur(4px)',
          zIndex: 100,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '1.5rem',
        }}>
          <div style={{
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border-color)',
            borderRadius: 'var(--radius-lg)',
            width: '100%',
            maxWidth: '800px',
            maxHeight: '85vh',
            display: 'flex',
            flexDirection: 'column',
          }}>
            <div style={{
              padding: '1.25rem',
              borderBottom: '1px solid var(--border-color)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}>
              <div>
                <h4 style={{ fontWeight: 600, fontSize: '1rem' }}>{selectedDoc.filename}</h4>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  Total {docChunks.length} chunks • 384-dimensional pgvector embeddings
                </p>
              </div>
              <button
                onClick={() => setSelectedDoc(null)}
                style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}
              >
                <X size={20} />
              </button>
            </div>

            <div style={{ padding: '1.25rem', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              {chunksLoading ? (
                <p style={{ textAlign: 'center', color: 'var(--text-muted)' }}>Loading chunks...</p>
              ) : (
                docChunks.map((c) => (
                  <div key={c.id} style={{
                    background: 'var(--bg-card)',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: 'var(--radius-md)',
                    padding: '1rem',
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                      <span className="badge badge-indigo">
                        Chunk #{c.chunk_index}
                      </span>
                      <span className="badge badge-cyan">
                        Page {c.page_number}
                      </span>
                    </div>
                    <p style={{ fontSize: '0.85rem', color: 'var(--text-main)', whiteSpace: 'pre-wrap' }}>
                      {c.content}
                    </p>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
