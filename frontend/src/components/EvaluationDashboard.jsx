import React, { useState } from 'react';
import { BarChart3, Play, Loader2, CheckCircle2, ShieldAlert, Award } from 'lucide-react';
import { api } from '../services/api';

export default function EvaluationDashboard() {
  const [loading, setLoading] = useState(false);
  const [skipLlm, setSkipLlm] = useState(true);
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);

  const runBenchmark = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.runEvaluation(skipLlm);
      setReport(data.benchmark_results);
    } catch (err) {
      setError(err.message || 'Benchmark evaluation failed');
    } finally {
      setLoading(false);
    }
  };

  const retrievalData = report?.retrieval_comparison;
  const genData = report?.generation_metrics;

  return (
    <div style={{ maxWidth: '1100px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Header Card */}
      <div className="card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <h3 style={{ fontSize: '1.2rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <BarChart3 size={22} color="var(--primary)" />
            RAG Evaluation & Benchmark Suite
          </h3>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
            Evaluates Recall@K, MRR, nDCG, Citation Accuracy, and Faithfulness across all retrieval strategies.
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.85rem', color: 'var(--text-muted)', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={!skipLlm}
              onChange={(e) => setSkipLlm(!e.target.checked)}
              style={{ accentColor: 'var(--primary)' }}
            />
            Include LLM Generation Evals
          </label>

          <button
            onClick={runBenchmark}
            disabled={loading}
            className="btn-primary"
          >
            {loading ? (
              <>
                <Loader2 size={16} style={{ animation: 'spin 1s linear infinite' }} />
                Evaluating...
              </>
            ) : (
              <>
                <Play size={16} />
                Run Benchmark
              </>
            )}
          </button>
        </div>
      </div>

      {error && (
        <div style={{ padding: '1rem', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: 'var(--radius-md)', color: '#f87171', fontSize: '0.9rem' }}>
          {error}
        </div>
      )}

      {/* Results Section */}
      {retrievalData && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {/* Retrieval Comparison Table */}
          <div className="card">
            <h4 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Award size={18} color="var(--primary)" />
              Retrieval Strategy Comparison Scorecard
            </h4>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
              Comparative benchmark proving why two-stage Hybrid RRF + Reranker outperforms standalone methods.
            </p>

            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.85rem' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border-color)', color: 'var(--text-muted)' }}>
                    <th style={{ padding: '0.75rem 0.5rem' }}>Strategy</th>
                    <th style={{ padding: '0.75rem 0.5rem' }}>Recall@1</th>
                    <th style={{ padding: '0.75rem 0.5rem' }}>Recall@3</th>
                    <th style={{ padding: '0.75rem 0.5rem' }}>Recall@5</th>
                    <th style={{ padding: '0.75rem 0.5rem' }}>MRR</th>
                    <th style={{ padding: '0.75rem 0.5rem' }}>nDCG@3</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(retrievalData).map(([strat, metrics]) => {
                    const isHybrid = strat === 'hybrid_reranked';
                    return (
                      <tr
                        key={strat}
                        style={{
                          borderBottom: '1px solid var(--border-subtle)',
                          background: isHybrid ? 'rgba(99, 102, 241, 0.08)' : 'transparent',
                          fontWeight: isHybrid ? 600 : 400,
                        }}
                      >
                        <td style={{ padding: '0.75rem 0.5rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                          {isHybrid && <span className="badge badge-green" style={{ fontSize: '0.65rem' }}>Best</span>}
                          <span>{strat.replace('_', ' ').toUpperCase()}</span>
                        </td>
                        <td style={{ padding: '0.75rem 0.5rem' }}>{metrics['recall@1']}</td>
                        <td style={{ padding: '0.75rem 0.5rem' }}>{metrics['recall@3']}</td>
                        <td style={{ padding: '0.75rem 0.5rem' }}>{metrics['recall@5']}</td>
                        <td style={{ padding: '0.75rem 0.5rem' }}>
                          <span className={isHybrid ? 'badge badge-indigo' : ''}>
                            {metrics['mrr']}
                          </span>
                        </td>
                        <td style={{ padding: '0.75rem 0.5rem' }}>
                          <span className={isHybrid ? 'badge badge-green' : ''}>
                            {metrics['ndcg@3']}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Generation & Citation Metrics */}
          {genData && Object.keys(genData).length > 0 && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '1rem' }}>
              <div className="card">
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.35rem' }}>Citation Accuracy</p>
                <h3 style={{ fontSize: '1.8rem', fontWeight: 700, color: '#34d399' }}>
                  {genData.citation_accuracy_percentage}%
                </h3>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                  Verified citations matching genuine retrieved chunks
                </p>
              </div>

              <div className="card">
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.35rem' }}>Mean Faithfulness Score</p>
                <h3 style={{ fontSize: '1.8rem', fontWeight: 700, color: '#a5b4fc' }}>
                  {genData.mean_faithfulness_score} / 1.00
                </h3>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                  Sentence-level context entailment (anti-hallucination)
                </p>
              </div>

              <div className="card">
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.35rem' }}>Anti-Hallucination Guardrail</p>
                <h3 style={{ fontSize: '1.8rem', fontWeight: 700, color: '#67e8f9' }}>
                  {genData.anti_hallucination_block_rate}%
                </h3>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                  Unanswerable questions correctly deflected
                </p>
              </div>
            </div>
          )}
        </div>
      )}

      {!retrievalData && !loading && (
        <div className="card" style={{ padding: '3.5rem 1.5rem', textAlign: 'center', color: 'var(--text-muted)' }}>
          <BarChart3 size={48} style={{ opacity: 0.25, marginBottom: '0.75rem' }} />
          <p style={{ fontWeight: 600, fontSize: '1rem' }}>Ready to Run Benchmark</p>
          <p style={{ fontSize: '0.85rem', marginTop: '0.35rem' }}>
            Click <strong>"Run Benchmark"</strong> to evaluate Vector Search vs Keyword Search vs Hybrid Reranked on Recall@K, MRR, and nDCG.
          </p>
        </div>
      )}
    </div>
  );
}
