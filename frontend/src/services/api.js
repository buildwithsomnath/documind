const BASE_URL = '/api';

async function request(endpoint, options = {}) {
  const url = `${BASE_URL}${endpoint}`;
  const response = await fetch(url, options);

  let data;
  const contentType = response.headers.get('content-type');
  if (contentType && contentType.includes('application/json')) {
    data = await response.json();
  } else {
    data = await response.text();
  }

  if (!response.ok) {
    const errorMsg = data && data.message ? data.message : `HTTP error ${response.status}`;
    throw new Error(errorMsg);
  }

  return data;
}

export const api = {
  // System Health
  async getHealth() {
    return request('/health');
  },

  // Document Management
  async getDocuments() {
    return request('/documents');
  },

  async getDocument(documentId) {
    return request(`/documents/${documentId}?include_chunks=true`);
  },

  async uploadDocument(file, chunkSize, chunkOverlap) {
    const formData = new FormData();
    formData.append('file', file);
    if (chunkSize) formData.append('chunk_size', chunkSize);
    if (chunkOverlap) formData.append('chunk_overlap', chunkOverlap);

    return request('/documents/upload', {
      method: 'POST',
      body: formData,
    });
  },

  async deleteDocument(documentId) {
    return request(`/documents/${documentId}`, {
      method: 'DELETE',
    });
  },

  // Search
  async searchHybrid(query, topK = 5, documentId = null, enableRerank = true) {
    return request('/search/hybrid', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        top_k: topK,
        document_id: documentId || undefined,
        enable_rerank: enableRerank,
      }),
    });
  },

  // Grounded Q&A Chat
  async askQuestion(question, topK = 3, documentId = null, enableRerank = true) {
    return request('/chat/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        top_k: topK,
        document_id: documentId || undefined,
        enable_rerank: enableRerank,
      }),
    });
  },

  // RAG Evaluation
  async runEvaluation(skipLlm = true) {
    return request('/evaluation/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ skip_llm: skipLlm }),
    });
  },
};
