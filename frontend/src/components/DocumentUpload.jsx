import React, { useState, useRef } from 'react';
import { UploadCloud, FileText, CheckCircle2, AlertTriangle, Loader2 } from 'lucide-react';
import { api } from '../services/api';

export default function DocumentUpload({ onUploadSuccess }) {
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleFile = async (file) => {
    if (!file) return;

    const ext = file.name.split('.').pop().toLowerCase();
    if (!['pdf', 'docx'].includes(ext)) {
      setError('Only PDF (.pdf) and Word (.docx) files are supported.');
      return;
    }

    setUploading(true);
    setError(null);
    setUploadResult(null);

    try {
      const response = await api.uploadDocument(file);
      setUploadResult(response.document);
      if (onUploadSuccess) onUploadSuccess();
    } catch (err) {
      setError(err.message || 'Failed to ingest document');
    } finally {
      setUploading(false);
    }
  };

  const onDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const onDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="card" style={{ marginBottom: '2rem' }}>
      <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <UploadCloud size={20} color="var(--primary)" />
        Ingest Documents (PDF & DOCX)
      </h3>
      <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1.25rem' }}>
        Extracts page-by-page text, segments into 500-character overlapping chunks, computes 384-d vector embeddings, and builds HNSW indexes in PostgreSQL.
      </p>

      {/* Drop Zone */}
      <div
        onDragEnter={onDrag}
        onDragLeave={onDrag}
        onDragOver={onDrag}
        onDrop={onDrop}
        onClick={() => fileInputRef.current?.click()}
        style={{
          border: `2px dashed ${dragActive ? 'var(--primary)' : 'var(--border-color)'}`,
          backgroundColor: dragActive ? 'var(--primary-glow)' : 'rgba(31, 41, 55, 0.4)',
          borderRadius: 'var(--radius-md)',
          padding: '2.5rem 1.5rem',
          textAlign: 'center',
          cursor: uploading ? 'wait' : 'pointer',
          transition: 'all 0.2s',
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.docx"
          style={{ display: 'none' }}
          onChange={(e) => handleFile(e.target.files?.[0])}
          disabled={uploading}
        />

        {uploading ? (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.75rem' }}>
            <Loader2 size={32} className="spin" color="var(--primary)" style={{ animation: 'spin 1s linear infinite' }} />
            <p style={{ fontWeight: 600, fontSize: '0.95rem' }}>Processing & Embedding Document...</p>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Extracting pages → Page-aware chunking → all-MiniLM-L6-v2 embedding</p>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{
              width: '48px',
              height: '48px',
              borderRadius: '50%',
              background: 'var(--bg-card)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'var(--primary)',
            }}>
              <FileText size={24} />
            </div>
            <div>
              <p style={{ fontWeight: 600, fontSize: '0.95rem' }}>
                Click to upload or drag and drop
              </p>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                PDF or DOCX (Max 16 MB)
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Success Notification */}
      {uploadResult && (
        <div style={{
          marginTop: '1rem',
          padding: '1rem',
          background: 'rgba(16, 185, 129, 0.1)',
          border: '1px solid rgba(16, 185, 129, 0.3)',
          borderRadius: 'var(--radius-md)',
          display: 'flex',
          alignItems: 'flex-start',
          gap: '0.75rem',
        }}>
          <CheckCircle2 size={20} color="#34d399" style={{ marginTop: '0.1rem', flexShrink: 0 }} />
          <div>
            <p style={{ fontWeight: 600, color: '#34d399', fontSize: '0.9rem' }}>
              Successfully Ingested: {uploadResult.filename}
            </p>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
              Generated <strong>{uploadResult.total_chunks} chunks</strong> across <strong>{uploadResult.total_pages} page(s)</strong> with 384-dimensional pgvector embeddings.
            </p>
          </div>
        </div>
      )}

      {/* Error Notification */}
      {error && (
        <div style={{
          marginTop: '1rem',
          padding: '1rem',
          background: 'rgba(239, 68, 68, 0.1)',
          border: '1px solid rgba(239, 68, 68, 0.3)',
          borderRadius: 'var(--radius-md)',
          display: 'flex',
          alignItems: 'flex-start',
          gap: '0.75rem',
        }}>
          <AlertTriangle size={20} color="#f87171" style={{ marginTop: '0.1rem', flexShrink: 0 }} />
          <div>
            <p style={{ fontWeight: 600, color: '#f87171', fontSize: '0.9rem' }}>Ingestion Failed</p>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>{error}</p>
          </div>
        </div>
      )}

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
