/**
 * CogRepo v2 API Client
 *
 * Provides access to v2 features:
 * - Fast database-backed search
 * - Semantic search with embeddings
 * - Artifact browsing
 * - Project/chain visualization
 */

const API_V2_BASE = '/api/v2';

// =============================================================================
// Search
// =============================================================================

/**
 * Search conversations using v2 API
 * @param {Object} params - Search parameters
 * @param {string} params.query - Search query
 * @param {string} params.mode - Search mode (bm25, semantic, hybrid)
 * @param {string} params.source - Filter by source
 * @param {string} params.tag - Filter by tag
 * @param {boolean} params.hasCode - Filter for conversations with code
 * @param {number} params.minScore - Minimum quality score
 * @param {number} params.page - Page number
 * @param {number} params.limit - Results per page
 */
async function searchV2(params = {}) {
  const queryParams = new URLSearchParams();

  if (params.query) queryParams.set('q', params.query);
  if (params.mode) queryParams.set('mode', params.mode);
  if (params.source) queryParams.set('source', params.source);
  if (params.tag) queryParams.set('tag', params.tag);
  if (params.hasCode) queryParams.set('has_code', 'true');
  if (params.minScore) queryParams.set('min_score', params.minScore);
  if (params.page) queryParams.set('page', params.page);
  if (params.limit) queryParams.set('limit', params.limit);

  const response = await fetch(`${API_V2_BASE}/search?${queryParams}`);
  return response.json();
}

/**
 * Semantic search using embeddings
 * @param {string} query - Search query
 * @param {number} limit - Number of results
 */
async function semanticSearch(query, limit = 10) {
  const response = await fetch(
    `${API_V2_BASE}/semantic_search?q=${encodeURIComponent(query)}&limit=${limit}`
  );
  return response.json();
}

// =============================================================================
// Artifacts
// =============================================================================

/**
 * List artifacts with optional filtering
 * @param {Object} params - Filter parameters
 */
async function listArtifacts(params = {}) {
  const queryParams = new URLSearchParams();

  if (params.type) queryParams.set('type', params.type);
  if (params.language) queryParams.set('language', params.language);
  if (params.convoId) queryParams.set('convo_id', params.convoId);
  if (params.page) queryParams.set('page', params.page);
  if (params.limit) queryParams.set('limit', params.limit);

  const response = await fetch(`${API_V2_BASE}/artifacts?${queryParams}`);
  return response.json();
}

/**
 * Get artifacts for a specific conversation
 * @param {string} convoId - Conversation ID
 */
async function getConversationArtifacts(convoId) {
  const response = await fetch(`${API_V2_BASE}/artifacts/${encodeURIComponent(convoId)}`);
  return response.json();
}

/**
 * Get artifact type counts
 */
async function getArtifactTypes() {
  const response = await fetch(`${API_V2_BASE}/artifact_types`);
  return response.json();
}

/**
 * Get programming language counts
 */
async function getLanguages() {
  const response = await fetch(`${API_V2_BASE}/languages`);
  return response.json();
}

// =============================================================================
// Context (Projects & Chains)
// =============================================================================

/**
 * List all detected projects
 */
async function listProjects() {
  const response = await fetch(`${API_V2_BASE}/projects`);
  return response.json();
}

/**
 * Get conversations for a project
 * @param {string} projectName - Project name
 */
async function getProjectConversations(projectName) {
  const response = await fetch(
    `${API_V2_BASE}/projects/${encodeURIComponent(projectName)}/conversations`
  );
  return response.json();
}

/**
 * List all conversation chains
 */
async function listChains() {
  const response = await fetch(`${API_V2_BASE}/chains`);
  return response.json();
}

/**
 * Get conversations in a chain
 * @param {string} chainId - Chain ID
 */
async function getChainConversations(chainId) {
  const response = await fetch(
    `${API_V2_BASE}/chains/${encodeURIComponent(chainId)}/conversations`
  );
  return response.json();
}

// =============================================================================
// Stats & Metadata
// =============================================================================

/**
 * Get comprehensive statistics
 */
async function getStatsV2() {
  const response = await fetch(`${API_V2_BASE}/stats`);
  return response.json();
}

/**
 * Get tag cloud
 * @param {number} limit - Maximum tags to return
 */
async function getTagsV2(limit = 50) {
  const response = await fetch(`${API_V2_BASE}/tags?limit=${limit}`);
  return response.json();
}

/**
 * Get full conversation with enrichments
 * @param {string} convoId - Conversation ID
 */
async function getConversationV2(convoId) {
  const response = await fetch(`${API_V2_BASE}/conversation/${encodeURIComponent(convoId)}`);
  return response.json();
}

/**
 * Check v2 API health and capabilities
 */
async function checkV2Health() {
  try {
    const response = await fetch(`${API_V2_BASE}/health`);
    return response.json();
  } catch (e) {
    return { status: 'error', error: e.message };
  }
}

// =============================================================================
// UI Helpers
// =============================================================================

/**
 * Highlight code with syntax highlighting
 * Uses highlight.js if available, otherwise returns escaped HTML
 * @param {string} code - Code to highlight
 * @param {string} language - Programming language
 */
function highlightCode(code, language = '') {
  // Escape HTML
  const escaped = code
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');

  // If highlight.js is available, use it
  if (typeof hljs !== 'undefined') {
    try {
      if (language && hljs.getLanguage(language)) {
        return hljs.highlight(code, { language }).value;
      }
      return hljs.highlightAuto(code).value;
    } catch (e) {
      return escaped;
    }
  }

  return escaped;
}

/**
 * Create artifact HTML element
 * @param {Object} artifact - Artifact data
 */
function createArtifactElement(artifact) {
  const container = document.createElement('div');
  container.className = 'artifact-card';
  container.dataset.type = artifact.artifact_type;

  // Type badge
  const typeBadge = document.createElement('span');
  typeBadge.className = `artifact-type-badge type-${artifact.artifact_type}`;
  typeBadge.textContent = formatArtifactType(artifact.artifact_type);

  // Language badge (if applicable)
  let langBadge = '';
  if (artifact.language) {
    langBadge = `<span class="artifact-lang-badge">${artifact.language}</span>`;
  }

  // Content
  const content = artifact.content || '';
  const highlighted = highlightCode(content, artifact.language);

  // Description
  const description = artifact.description
    ? `<p class="artifact-description">${escapeHtml(artifact.description)}</p>`
    : '';

  container.innerHTML = `
    <div class="artifact-header">
      ${typeBadge.outerHTML}
      ${langBadge}
      <button class="artifact-copy-btn" onclick="copyArtifact(this)" title="Copy to clipboard">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
        </svg>
      </button>
    </div>
    ${description}
    <pre class="artifact-code"><code class="language-${artifact.language || 'plaintext'}">${highlighted}</code></pre>
  `;

  return container;
}

/**
 * Format artifact type for display
 */
function formatArtifactType(type) {
  const typeMap = {
    'code_snippet': 'Code',
    'shell_command': 'Command',
    'configuration': 'Config',
    'error_solution': 'Solution',
    'best_practice': 'Best Practice',
    'api_example': 'API Example'
  };
  return typeMap[type] || type;
}

/**
 * Escape HTML entities
 */
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

/**
 * Copy artifact content to clipboard
 */
async function copyArtifact(button) {
  const card = button.closest('.artifact-card');
  const code = card.querySelector('.artifact-code code');

  if (code) {
    try {
      await navigator.clipboard.writeText(code.textContent);
      button.classList.add('copied');
      setTimeout(() => button.classList.remove('copied'), 2000);
    } catch (e) {
      console.error('Failed to copy:', e);
    }
  }
}

/**
 * Create project card element
 */
function createProjectCard(project) {
  const container = document.createElement('div');
  container.className = 'project-card';

  const techBadges = (project.technologies || [])
    .slice(0, 5)
    .map(t => `<span class="tech-badge">${escapeHtml(t)}</span>`)
    .join('');

  container.innerHTML = `
    <div class="project-header">
      <h3 class="project-name">${escapeHtml(project.name)}</h3>
      <span class="project-count">${project.conversation_ids?.length || 0} conversations</span>
    </div>
    <div class="project-tech">${techBadges}</div>
    <div class="project-confidence">
      <span class="confidence-bar" style="width: ${(project.confidence || 0) * 100}%"></span>
    </div>
  `;

  container.onclick = () => showProjectDetail(project.name);

  return container;
}

/**
 * Create chain card element
 */
function createChainCard(chain) {
  const container = document.createElement('div');
  container.className = 'chain-card';

  const typeColors = {
    'continuation': '#4CAF50',
    'follow_up': '#2196F3',
    'debug_session': '#FF9800',
    'semantic_link': '#9C27B0'
  };

  const color = typeColors[chain.chain_type] || '#666';

  container.innerHTML = `
    <div class="chain-header" style="border-left: 3px solid ${color}">
      <span class="chain-type">${chain.chain_type.replace('_', ' ')}</span>
      <span class="chain-count">${chain.conversation_ids?.length || 0} conversations</span>
    </div>
    <div class="chain-entities">
      ${(chain.shared_entities || []).slice(0, 3).map(e => `<span class="entity-badge">${escapeHtml(e)}</span>`).join('')}
    </div>
  `;

  container.onclick = () => showChainDetail(chain.chain_id);

  return container;
}

// Export for use in other scripts
window.CogRepoV2 = {
  // Search
  search: searchV2,
  semanticSearch,

  // Artifacts
  listArtifacts,
  getConversationArtifacts,
  getArtifactTypes,
  getLanguages,

  // Context
  listProjects,
  getProjectConversations,
  listChains,
  getChainConversations,

  // Stats
  getStats: getStatsV2,
  getTags: getTagsV2,
  getConversation: getConversationV2,
  checkHealth: checkV2Health,

  // UI Helpers
  highlightCode,
  createArtifactElement,
  createProjectCard,
  createChainCard
};
