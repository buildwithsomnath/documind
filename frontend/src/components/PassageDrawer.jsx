import React from 'react';
import { Layers, CheckCircle2, Bookmark } from 'lucide-react';

export default function PassageDrawer({ passages, highlightedChunkId }) {
  if (!passages || passages.length === 0) {
    return (
      <div className="card" style={{ height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', textAlign: 'center', color: 'var(--text-muted)' }}>
        <Layers size={32} style={{ opacity: 0.3, marginBottom: '0.5rem' }} />
        <p style={{ fontWeight: 500, fontSize: '0.9rem' }}>Retrieved Passages</p>
        <p style={{ fontSize: '0.75rem', marginTop: '0.25rem' }}>Ask a question to inspect the retrieved grounding chunks & RRF/reranker provenance.</p>
      </div>
    );
  }

  return (
    <div className="card" style={{ height: '100%', overflowY: 'auto', display: 'flex', flexDirection: 'column' }}>
      <div style={{ paddingBottom: '0.75rem', borderBottom: '1px solid var(--border-color)', marginBottom: '1rem' }}>
        <h4 style={{ fontWeight: 600, fontSize: '0.95rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Layers size={18} color="var(--primary)" />
          Retrieved Grounding Passages ({passages.length})
        </h4>
        <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          Two-stage retrieved passages with cross-encoder relevance scores
        </p>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
        {passages.map((p, idx) => {
          const isHighlighted = highlightedChunkId === p.chunk_id;
          const rrf = p.metadata?.rrf_provenance;
          const rerank = p.metadata?.rerank_provenance;

          return (
            <div
              key={p.chunk_id || idx}
              style={{
                background: isHighlighted ? 'rgba(99, 102, 241, 0.15)' : 'var(--bg-card)',
                border: `1px solid ${isHighlighted ? 'var(--primary)' : 'var(--border-subtle)'}`,
                borderRadius: 'var(--radius-md)',
                padding: '0.85rem',
                transition: 'all 0.2s',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
                <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                  <Bookmark size={13} color="var(--primary)" />
                  {p.filename}
                </span>
                <span className="badge badge-cyan" style={{ fontSize: '0.7rem' }}>
                  Page {p.page_number}
                </span>
              </div>

              <p style={{ fontSize: '0.8rem', color: 'var(--text-main)', lineHeight: 1.45, marginBottom: '0.5rem' }}>
                {p.content}
              </p>

              {/* Provenance Pills */}
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem', paddingTop: '0.4rem', borderTop: '1px solid rgba(255, 255, 255, 0.05)' }}>
                <span className="badge badge-indigo" style={{ fontSize: '0.65rem' }}>
                  Score: {p.score}
                </span>

                {rrf && rrf.found_in_both && (
                  <span className="badge badge-green" style={{ fontSize: '0.65rem' }}>
                    <CheckCircle2 size={10} />
                    Dual Vector + FTS Match
                  </span>
                )}

                {rrf && (
                  <span className="badge badge-amber" style={{ fontSize: '0.65rem' }}>
                    RRF Rank #{rrf.vector_rank || '-'}/#{rrf.keyword_rank || '-'}
                  </span>
                )}

                {rerank && (
                  <span className="badge badge-cyan" style={{ fontSize: '0.65rem' }}>
                    Logit: {rerank.raw_logit}
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
