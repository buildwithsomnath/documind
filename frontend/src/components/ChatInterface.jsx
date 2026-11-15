import React, { useState } from 'react';
import { Send, Bot, User, ShieldCheck, Sparkles, Filter, Sliders, Loader2, Bookmark } from 'lucide-react';
import { api } from '../services/api';
import PassageDrawer from './PassageDrawer';

export default function ChatInterface({ documents }) {
  const [messages, setMessages] = useState([
    {
      id: 'welcome',
      role: 'assistant',
      content: 'Welcome to DocuMind! Ask any question about your indexed documents. Answers are strictly grounded in retrieved passages with verified page-level citations.',
      citations: [],
      retrieved_passages: [],
      accuracy: 100,
    }
  ]);
  const [inputQuery, setInputQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [topK, setTopK] = useState(3);
  const [selectedDocId, setSelectedDocId] = useState('');
  const [enableRerank, setEnableRerank] = useState(true);
  const [currentPassages, setCurrentPassages] = useState([]);
  const [highlightedChunkId, setHighlightedChunkId] = useState(null);

  const handleSend = async (e) => {
    e?.preventDefault();
    const query = inputQuery.trim();
    if (!query || loading) return;

    const userMessage = {
      id: `u-${Date.now()}`,
      role: 'user',
      content: query,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputQuery('');
    setLoading(true);

    try {
      const resp = await api.askQuestion(
        query,
        topK,
        selectedDocId || null,
        enableRerank
      );

      const assistantMessage = {
        id: `a-${Date.now()}`,
        role: 'assistant',
        content: resp.answer,
        citations: resp.citations || [],
        sources_used: resp.sources_used || [],
        retrieved_passages: resp.retrieved_passages || [],
        accuracy: resp.citation_accuracy_percentage,
      };

      setMessages((prev) => [...prev, assistantMessage]);
      setCurrentPassages(resp.retrieved_passages || []);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: `err-${Date.now()}`,
          role: 'assistant',
          content: `Error generating response: ${err.message}`,
          isError: true,
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.8fr) minmax(320px, 1.2fr)', gap: '1.5rem', height: 'calc(100vh - 120px)' }}>
      {/* Left Column: Chat Conversation */}
      <div style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0 }}>
        {/* Controls Bar */}
        <div className="card" style={{ padding: '0.75rem 1rem', marginBottom: '1rem', display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            <Filter size={14} color="var(--primary)" />
            <span>Document Scope:</span>
            <select
              value={selectedDocId}
              onChange={(e) => setSelectedDocId(e.target.value)}
              style={{
                background: 'var(--bg-card)',
                color: 'var(--text-main)',
                border: '1px solid var(--border-color)',
                borderRadius: 'var(--radius-sm)',
                padding: '0.3rem 0.6rem',
                fontSize: '0.8rem',
              }}
            >
              <option value="">All Documents</option>
              {documents.map((d) => (
                <option key={d.id} value={d.id}>{d.filename}</option>
              ))}
            </select>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            <Sliders size={14} color="var(--primary)" />
            <span>Top-K:</span>
            <select
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              style={{
                background: 'var(--bg-card)',
                color: 'var(--text-main)',
                border: '1px solid var(--border-color)',
                borderRadius: 'var(--radius-sm)',
                padding: '0.3rem 0.5rem',
                fontSize: '0.8rem',
              }}
            >
              <option value={2}>2 Passages</option>
              <option value={3}>3 Passages</option>
              <option value={5}>5 Passages</option>
            </select>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8rem', marginLeft: 'auto' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', cursor: 'pointer', color: enableRerank ? '#a5b4fc' : 'var(--text-muted)' }}>
              <input
                type="checkbox"
                checked={enableRerank}
                onChange={(e) => setEnableRerank(e.target.checked)}
                style={{ accentColor: 'var(--primary)' }}
              />
              <Sparkles size={14} />
              Cross-Encoder Reranker
            </label>
          </div>
        </div>

        {/* Message Stream */}
        <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '1rem', paddingRight: '0.5rem' }}>
          {messages.map((msg) => (
            <div
              key={msg.id}
              style={{
                display: 'flex',
                gap: '0.75rem',
                alignItems: 'flex-start',
                alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
                maxWidth: msg.role === 'user' ? '85%' : '95%',
              }}
            >
              {msg.role === 'assistant' && (
                <div style={{
                  width: '32px',
                  height: '32px',
                  borderRadius: '50%',
                  background: 'var(--primary)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#fff',
                  flexShrink: 0,
                  marginTop: '0.2rem',
                }}>
                  <Bot size={18} />
                </div>
              )}

              <div style={{
                background: msg.role === 'user' ? 'var(--primary)' : 'var(--bg-secondary)',
                border: `1px solid ${msg.role === 'user' ? 'var(--primary)' : 'var(--border-color)'}`,
                color: msg.role === 'user' ? '#fff' : 'var(--text-main)',
                borderRadius: 'var(--radius-md)',
                padding: '1rem 1.25rem',
                fontSize: '0.9rem',
                lineHeight: 1.6,
                boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
              }}>
                <p style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</p>

                {/* Grounded Citations Bar */}
                {msg.citations && msg.citations.length > 0 && (
                  <div style={{ marginTop: '0.75rem', paddingTop: '0.75rem', borderTop: '1px solid var(--border-subtle)', display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '0.5rem' }}>
                    <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                      <Bookmark size={13} color="var(--primary)" />
                      Verified Citations:
                    </span>
                    {msg.citations.map((c, cIdx) => (
                      <button
                        key={cIdx}
                        onClick={() => {
                          setCurrentPassages(msg.retrieved_passages || []);
                          setHighlightedChunkId(c.chunk_id);
                        }}
                        className={`badge ${c.verified ? 'badge-indigo' : 'badge-amber'}`}
                        style={{ border: 'none', cursor: 'pointer', padding: '0.2rem 0.5rem' }}
                        title="Click to view passage source"
                      >
                        {c.filename}, Page {c.page_number}
                      </button>
                    ))}
                    {msg.accuracy !== undefined && (
                      <span className="badge badge-green" style={{ marginLeft: 'auto' }}>
                        <ShieldCheck size={12} />
                        {msg.accuracy}% Citation Accuracy
                      </span>
                    )}
                  </div>
                )}
              </div>

              {msg.role === 'user' && (
                <div style={{
                  width: '32px',
                  height: '32px',
                  borderRadius: '50%',
                  background: 'var(--bg-card)',
                  border: '1px solid var(--border-color)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'var(--text-muted)',
                  flexShrink: 0,
                  marginTop: '0.2rem',
                }}>
                  <User size={18} />
                </div>
              )}
            </div>
          ))}

          {loading && (
            <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
              <div style={{
                width: '32px',
                height: '32px',
                borderRadius: '50%',
                background: 'var(--primary)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#fff',
              }}>
                <Bot size={18} />
              </div>
              <div className="card" style={{ padding: '0.75rem 1.25rem', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <Loader2 size={18} style={{ animation: 'spin 1s linear infinite' }} color="var(--primary)" />
                <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                  Hybrid searching & generating grounded answer with Gemini...
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Input Form */}
        <form onSubmit={handleSend} style={{ marginTop: '1rem', display: 'flex', gap: '0.75rem' }}>
          <input
            type="text"
            value={inputQuery}
            onChange={(e) => setInputQuery(e.target.value)}
            placeholder="Ask a question about your documents (e.g. 'What is the chunk size and token limit?')"
            disabled={loading}
            style={{
              flex: 1,
              background: 'var(--bg-secondary)',
              border: '1px solid var(--border-color)',
              color: 'var(--text-main)',
              borderRadius: 'var(--radius-md)',
              padding: '0.85rem 1.25rem',
              fontSize: '0.9rem',
              outline: 'none',
              transition: 'border 0.2s',
            }}
          />
          <button type="submit" disabled={!inputQuery.trim() || loading} className="btn-primary">
            <Send size={16} />
            Ask
          </button>
        </form>
      </div>

      {/* Right Column: Retrieved Grounding Passages Drawer */}
      <div style={{ height: '100%', minHeight: 0 }}>
        <PassageDrawer
          passages={currentPassages}
          highlightedChunkId={highlightedChunkId}
        />
      </div>
    </div>
  );
}
