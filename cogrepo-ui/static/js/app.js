/**
 * CogRepo Main Application
 *
 * Main application logic for the conversation explorer.
 * Integrates API, UI, search, and state management.
 */

// =============================================================================
// State Management
// =============================================================================

class Store {
  constructor() {
    this.state = {
      conversations: [],
      filteredResults: [],
      searchQuery: '',
      filters: {
        source: '',
        dateFrom: '',
        dateTo: '',
        minScore: 0
      },
      pagination: {
        currentPage: 1,
        itemsPerPage: 25,
        totalItems: 0
      },
      tags: new Map(),
      savedSearches: [],
      isLoading: false,
      error: null,
      stats: null
    };

    this.listeners = new Set();
    this._loadSavedSearches();
  }

  /**
   * Get current state
   */
  getState() {
    return { ...this.state };
  }

  /**
   * Update state and notify listeners
   */
  setState(updates) {
    this.state = { ...this.state, ...updates };
    this._notify();
  }

  /**
   * Subscribe to state changes
   */
  subscribe(listener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  _notify() {
    for (const listener of this.listeners) {
      listener(this.state);
    }
  }

  // Saved searches persistence
  _loadSavedSearches() {
    try {
      const saved = localStorage.getItem('cogrepo_saved_searches');
      if (saved) {
        this.state.savedSearches = JSON.parse(saved);
      }
    } catch (e) {
      console.warn('Failed to load saved searches:', e);
    }
  }

  saveSavedSearches() {
    try {
      localStorage.setItem('cogrepo_saved_searches', JSON.stringify(this.state.savedSearches));
    } catch (e) {
      console.warn('Failed to save searches:', e);
    }
  }

  addSavedSearch(search) {
    if (this.state.savedSearches.some(s => s.query === search.query)) return;
    this.state.savedSearches.unshift(search);
    if (this.state.savedSearches.length > 10) {
      this.state.savedSearches = this.state.savedSearches.slice(0, 10);
    }
    this.saveSavedSearches();
    this._notify();
  }

  removeSavedSearch(query) {
    this.state.savedSearches = this.state.savedSearches.filter(s => s.query !== query);
    this.saveSavedSearches();
    this._notify();
  }
}

// =============================================================================
// Search Highlighting
// =============================================================================

function highlightText(text, query) {
  if (!query || !text) return CogRepoUI.escapeHTML(text);

  const escaped = CogRepoUI.escapeHTML(text);
  const terms = query.split(/\s+/).filter(t => t.length > 1);

  if (terms.length === 0) return escaped;

  // Create regex that matches any term
  const pattern = terms.map(t => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|');
  const regex = new RegExp(`(${pattern})`, 'gi');

  return escaped.replace(regex, '<mark class="highlight">$1</mark>');
}

// =============================================================================
// Components
// =============================================================================

/**
 * Render a conversation card
 */
function renderConversationCard(conversation, query = '') {
  const {
    convo_id,
    external_id,
    generated_title,
    title,
    summary_abstractive,
    source,
    create_time,
    timestamp,
    tags = [],
    score,
    advanced_score,
    relevance
  } = conversation;

  // Use create_time or timestamp (fallback for older data)
  const dateValue = create_time || timestamp;
  const displayTitle = generated_title || title || 'Untitled Conversation';
  const displayDate = CogRepoUI.formatDate(dateValue);
  const relativeDate = CogRepoUI.formatRelativeTime(dateValue);

  const card = CogRepoUI.createElement('article', {
    className: 'card card-interactive conversation-card stagger-item',
    dataset: { convoId: convo_id || external_id },
    tabindex: '0',
    role: 'button',
    'aria-label': `View conversation: ${displayTitle}`
  });

  card.innerHTML = `
    <div class="conversation-card-header">
      <div class="conversation-meta">
        <time class="conversation-date" datetime="${dateValue}" title="${displayDate}">
          ${relativeDate || displayDate}
        </time>
        <div class="conversation-source">
          <span class="source-badge" data-source="${source}">${source}</span>
        </div>
      </div>
      <div style="display: flex; align-items: center; gap: 0.5rem;">
        ${advanced_score ? `
          <div class="conversation-score advanced-score">
            <span class="score-value">${advanced_score.overall}</span>
            <span class="score-grade grade-${advanced_score.grade.replace('+', 'plus').replace('-', 'minus')}">${advanced_score.grade}</span>
          </div>
        ` : score ? `
          <div class="conversation-score legacy-score">
            <span class="score-value">${score}</span>
            <span class="score-label">Score</span>
          </div>
        ` : ''}
        <button class="enrich-btn" data-conversation-id="${convo_id || external_id}"
                title="Enrich with SOTA AI" aria-label="Enrich conversation">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 3v18m9-9H3"/>
            <circle cx="12" cy="12" r="3"/>
          </svg>
          <span>Enrich</span>
        </button>
      </div>
    </div>
    <div class="conversation-card-body">
      <h3 class="conversation-title">${highlightText(displayTitle, query)}</h3>
      ${summary_abstractive ? `
        <p class="conversation-summary line-clamp-3">
          ${highlightText(summary_abstractive, query)}
        </p>
      ` : ''}
      <div class="conversation-tags">
        ${(tags || []).slice(0, 5).map(tag =>
          `<span class="tag tag-interactive" data-tag="${CogRepoUI.escapeHTML(tag)}">${CogRepoUI.escapeHTML(tag)}</span>`
        ).join('')}
        ${tags && tags.length > 5 ? `<span class="tag">+${tags.length - 5} more</span>` : ''}
      </div>
    </div>
  `;

  // Click handler
  card.addEventListener('click', (e) => {
    // Don't open modal if clicking on a tag
    if (e.target.classList.contains('tag-interactive')) {
      const tagValue = e.target.dataset.tag;
      if (tagValue) {
        document.getElementById('searchInput').value = `tag:${tagValue}`;
        window.app.performSearch(`tag:${tagValue}`);
      }
      return;
    }

    // Don't open modal if clicking on enrich button or its children
    if (e.target.classList.contains('enrich-btn') ||
        e.target.closest('.enrich-btn')) {
      return;
    }

    window.app.viewConversation(convo_id || external_id);
  });

  // Keyboard handler
  card.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      window.app.viewConversation(convo_id || external_id);
    }
  });

  return card;
}

/**
 * SVG illustrations for empty states
 */
const emptyStateIllustrations = {
  search: `<svg viewBox="0 0 120 120" fill="none" xmlns="http://www.w3.org/2000/svg">
    <circle cx="50" cy="50" r="30" stroke="currentColor" stroke-width="4" opacity="0.3"/>
    <circle cx="50" cy="50" r="30" stroke="url(#searchGrad)" stroke-width="4" stroke-dasharray="8 4"/>
    <line x1="72" y1="72" x2="95" y2="95" stroke="currentColor" stroke-width="4" stroke-linecap="round" opacity="0.5"/>
    <circle cx="95" cy="25" r="3" fill="currentColor" opacity="0.4">
      <animate attributeName="opacity" values="0.4;1;0.4" dur="2s" repeatCount="indefinite"/>
    </circle>
    <circle cx="25" cy="85" r="2" fill="currentColor" opacity="0.3">
      <animate attributeName="opacity" values="0.3;0.8;0.3" dur="2.5s" repeatCount="indefinite"/>
    </circle>
    <circle cx="105" cy="75" r="2.5" fill="currentColor" opacity="0.35">
      <animate attributeName="opacity" values="0.35;0.9;0.35" dur="1.8s" repeatCount="indefinite"/>
    </circle>
    <defs>
      <linearGradient id="searchGrad" x1="20" y1="20" x2="80" y2="80">
        <stop offset="0%" stop-color="#667eea"/>
        <stop offset="100%" stop-color="#764ba2"/>
      </linearGradient>
    </defs>
  </svg>`,
  initial: `<svg viewBox="0 0 120 120" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect x="20" y="35" width="50" height="35" rx="8" stroke="currentColor" stroke-width="3" opacity="0.3"/>
    <rect x="50" y="50" width="50" height="35" rx="8" stroke="url(#chatGrad)" stroke-width="3"/>
    <circle cx="35" cy="52" r="3" fill="currentColor" opacity="0.4"/>
    <circle cx="47" cy="52" r="3" fill="currentColor" opacity="0.4"/>
    <circle cx="59" cy="52" r="3" fill="currentColor" opacity="0.4">
      <animate attributeName="opacity" values="0.4;1;0.4" dur="1.5s" repeatCount="indefinite"/>
    </circle>
    <circle cx="67" cy="67" r="2.5" fill="currentColor" opacity="0.5"/>
    <circle cx="79" cy="67" r="2.5" fill="currentColor" opacity="0.5"/>
    <circle cx="91" cy="67" r="2.5" fill="currentColor" opacity="0.5">
      <animate attributeName="opacity" values="0.5;1;0.5" dur="1.5s" repeatCount="indefinite" begin="0.5s"/>
    </circle>
    <circle cx="100" cy="25" r="4" fill="currentColor" opacity="0.2">
      <animate attributeName="r" values="4;6;4" dur="3s" repeatCount="indefinite"/>
    </circle>
    <circle cx="15" cy="75" r="3" fill="currentColor" opacity="0.15">
      <animate attributeName="r" values="3;5;3" dur="3.5s" repeatCount="indefinite"/>
    </circle>
    <defs>
      <linearGradient id="chatGrad" x1="50" y1="50" x2="100" y2="85">
        <stop offset="0%" stop-color="#667eea"/>
        <stop offset="100%" stop-color="#764ba2"/>
      </linearGradient>
    </defs>
  </svg>`,
  error: `<svg viewBox="0 0 120 120" fill="none" xmlns="http://www.w3.org/2000/svg">
    <circle cx="60" cy="60" r="35" stroke="currentColor" stroke-width="3" opacity="0.2"/>
    <circle cx="60" cy="60" r="35" stroke="url(#errorGrad)" stroke-width="3" stroke-dasharray="12 8"/>
    <line x1="45" y1="45" x2="75" y2="75" stroke="currentColor" stroke-width="4" stroke-linecap="round" opacity="0.5"/>
    <line x1="75" y1="45" x2="45" y2="75" stroke="currentColor" stroke-width="4" stroke-linecap="round" opacity="0.5"/>
    <circle cx="95" cy="30" r="2" fill="currentColor" opacity="0.3">
      <animate attributeName="opacity" values="0.3;0.6;0.3" dur="2s" repeatCount="indefinite"/>
    </circle>
    <circle cx="25" cy="90" r="2.5" fill="currentColor" opacity="0.25">
      <animate attributeName="opacity" values="0.25;0.5;0.25" dur="2.5s" repeatCount="indefinite"/>
    </circle>
    <defs>
      <linearGradient id="errorGrad" x1="25" y1="25" x2="95" y2="95">
        <stop offset="0%" stop-color="#ef4444"/>
        <stop offset="100%" stop-color="#f97316"/>
      </linearGradient>
    </defs>
  </svg>`
};

/**
 * Render empty state
 */
function renderEmptyState(type = 'search') {
  const messages = {
    search: {
      illustration: emptyStateIllustrations.search,
      title: 'No results found',
      desc: 'Try adjusting your search terms or filters'
    },
    initial: {
      illustration: emptyStateIllustrations.initial,
      title: 'Search your conversations',
      desc: 'Enter a query above to explore your AI conversation archive'
    },
    error: {
      illustration: emptyStateIllustrations.error,
      title: 'Something went wrong',
      desc: 'Please try again or check your connection'
    }
  };

  const { illustration, title, desc } = messages[type] || messages.search;

  const container = CogRepoUI.createElement('div', { className: 'empty-state' });

  const iconDiv = CogRepoUI.createElement('div', {
    className: 'empty-state-illustration',
    innerHTML: illustration
  });

  container.appendChild(iconDiv);
  container.appendChild(CogRepoUI.createElement('h2', { className: 'empty-state-title' }, [title]));
  container.appendChild(CogRepoUI.createElement('p', { className: 'empty-state-desc' }, [desc]));

  return container;
}

/**
 * Render stats bar
 */
function renderStatsBar(stats) {
  if (!stats) return null;

  return `
    <div class="stats-bar">
      <div class="stat-item">
        <div class="stat-icon">💬</div>
        <div class="stat-content">
          <span class="stat-value">${stats.total_conversations?.toLocaleString() || 0}</span>
          <span class="stat-label">Conversations</span>
        </div>
      </div>
      <div class="stat-item">
        <div class="stat-icon">📅</div>
        <div class="stat-content">
          <span class="stat-value">${stats.date_range?.earliest ? CogRepoUI.formatDate(stats.date_range.earliest).split(',')[0] : 'N/A'}</span>
          <span class="stat-label">Earliest</span>
        </div>
      </div>
      <div class="stat-item">
        <div class="stat-icon">⭐</div>
        <div class="stat-content">
          <span class="stat-value">${stats.avg_score?.toFixed(1) || 'N/A'}</span>
          <span class="stat-label">Avg Score</span>
        </div>
      </div>
    </div>
  `;
}

/**
 * Render tags cloud
 */
function renderTagsCloud(tags) {
  if (!tags || tags.size === 0) return '';

  // Filter and sort tags
  // - Filter out malformed tags (with brackets, quotes, etc.)
  // - Only show tags with at least 5 occurrences
  // - Take top 50 tags by frequency
  const sortedTags = [...tags.entries()]
    .filter(([tag, count]) => {
      // Filter out malformed tags
      if (!tag || tag.length < 2) return false;
      if (tag.includes('[') || tag.includes(']') || tag.includes('"') || tag.includes("'")) return false;
      // Filter by minimum occurrence
      return count >= 5;
    })
    .sort((a, b) => b[1] - a[1])
    .slice(0, 50);

  if (sortedTags.length === 0) return '';

  const maxCount = sortedTags[0][1];

  return `
    <div class="tags-cloud">
      <div class="tags-cloud-header">
        <h3 class="tags-cloud-title">Popular Tags</h3>
      </div>
      <div class="tags-cloud-list">
        ${sortedTags.map(([tag, count]) => {
          // Scale size from 1-5 based on frequency
          const size = Math.min(Math.ceil((count / maxCount) * 5), 5);
          return `<span class="tag tag-interactive cloud-tag" data-size="${size}" data-tag="${CogRepoUI.escapeHTML(tag)}">${CogRepoUI.escapeHTML(tag)} <span class="tag-count">(${count})</span></span>`;
        }).join('')}
      </div>
    </div>
  `;
}

// =============================================================================
// Main Application Class
// =============================================================================

class CogRepoApp {
  constructor() {
    this.store = new Store();
    this.pagination = new CogRepoUI.Pagination({
      itemsPerPage: 25,
      onChange: (page) => this.handlePageChange(page)
    });

    // Search state management
    this.currentSearchAbort = null;
    this.lastSearchQuery = null;

    // Bind methods
    this.performSearch = this.performSearch.bind(this);
    this.handlePageChange = this.handlePageChange.bind(this);
  }

  /**
   * Initialize the application
   */
  async init() {
    console.log('🚀 CogRepo initializing...');

    // Setup UI components
    this._setupSearch();
    this._setupKeyboardShortcuts();
    this._setupFilters();
    this._setupSavedSearches();
    this._updatePlatformShortcuts();

    // Mount pagination
    this.pagination.mount('pagination');

    // Subscribe to state changes
    this.store.subscribe((state) => this._render(state));

    // Load initial data
    await this._loadInitialData();

    console.log('✅ CogRepo ready');
  }

  /**
   * Update keyboard shortcuts display for current platform
   */
  _updatePlatformShortcuts() {
    const modKey = CogRepoUI.getModifierKey();

    // Update shortcuts in the static shortcuts panel
    document.querySelectorAll('.shortcut-key kbd').forEach(kbd => {
      if (kbd.textContent === '⌘' || kbd.textContent === 'Ctrl') {
        kbd.textContent = modKey;
      }
    });
  }

  /**
   * Load initial data
   */
  async _loadInitialData() {
    this.store.setState({ isLoading: true });

    try {
      // Load stats
      const stats = await CogRepoAPI.getStats();
      this.store.setState({ stats });

      // Load all conversations for tag building
      const data = await CogRepoAPI.getConversations({ limit: 1000 });
      const conversations = Array.isArray(data) ? data : (data.conversations || []);

      // Build tags map
      const tags = new Map();
      for (const conv of conversations) {
        for (const tag of (conv.tags || [])) {
          tags.set(tag, (tags.get(tag) || 0) + 1);
        }
      }

      this.store.setState({
        conversations,
        tags,
        isLoading: false
      });

      // Display initial state (no search yet)
      this._renderInitialState();

    } catch (error) {
      console.error('Failed to load initial data:', error);
      this.store.setState({ isLoading: false, error: error.message });
      CogRepoUI.toast.error('Failed to load conversations');
    }
  }

  /**
   * Perform search
   */
  async performSearch(query, options = {}) {
    const { semantic = false } = options;
    const { filters } = this.store.getState();

    // Cancel previous search if still running
    if (this.currentSearchAbort) {
      this.currentSearchAbort.abort();
    }

    // Create new abort controller for this search
    this.currentSearchAbort = new AbortController();
    const signal = this.currentSearchAbort.signal;

    // Debounce: skip if this is the same query as last search
    if (query === this.lastSearchQuery && !semantic) {
      return;
    }
    this.lastSearchQuery = query;

    this.store.setState({
      searchQuery: query,
      isLoading: true,
      pagination: { ...this.store.state.pagination, currentPage: 1 }
    });

    const resultsContainer = document.getElementById('results');
    if (resultsContainer) {
      resultsContainer.innerHTML = '<div class="loading-overlay"><div class="spinner"></div></div>';
    }

    try {
      // Check if request was cancelled
      if (signal.aborted) return;
      // Build filter params
      const filterParams = {};
      if (filters.source) filterParams.source = filters.source;
      if (filters.dateFrom) filterParams.date_from = filters.dateFrom;
      if (filters.dateTo) filterParams.date_to = filters.dateTo;
      if (filters.minScore > 0) filterParams.min_score = filters.minScore;

      // Parse tag: syntax
      let searchQuery = query;
      const tagMatch = query.match(/^tag:(.+)$/i);
      if (tagMatch) {
        filterParams.tag = tagMatch[1].trim();
        searchQuery = ''; // Clear query so it only filters by tag
      }

      let results;

      if (searchQuery.trim() || filterParams.tag) {
        // API search
        const response = await CogRepoAPI.search(searchQuery, filterParams, {
          semantic,
          page: 1,
          limit: 100
        });
        results = response.results || [];
      } else {
        // Filter local data
        results = this._filterLocalResults(filterParams);
      }

      this.store.setState({
        filteredResults: results,
        isLoading: false,
        pagination: {
          ...this.store.state.pagination,
          totalItems: results.length,
          currentPage: 1
        }
      });

      this.pagination.update(results.length, 1);
      this._renderResults(results, query);

    } catch (error) {
      console.error('Search failed:', error);
      this.store.setState({ isLoading: false, error: error.message });
      CogRepoUI.toast.error('Search failed. Please try again.');
      this._renderResults([], query);
    }
  }

  /**
   * Filter local results when no query
   */
  _filterLocalResults(filters) {
    let results = [...this.store.state.conversations];

    if (filters.source) {
      results = results.filter(c => c.source === filters.source);
    }
    if (filters.date_from) {
      const from = new Date(filters.date_from);
      results = results.filter(c => new Date(c.create_time) >= from);
    }
    if (filters.date_to) {
      const to = new Date(filters.date_to);
      results = results.filter(c => new Date(c.create_time) <= to);
    }
    if (filters.min_score > 0) {
      results = results.filter(c => (c.score || 0) >= filters.min_score);
    }

    // Sort by date descending
    results.sort((a, b) => new Date(b.create_time) - new Date(a.create_time));

    return results;
  }

  /**
   * Handle page change
   */
  handlePageChange(page) {
    this.store.setState({
      pagination: { ...this.store.state.pagination, currentPage: page }
    });

    const { filteredResults, searchQuery } = this.store.getState();
    this._renderResults(filteredResults, searchQuery);

    // Scroll to top of results
    document.getElementById('results')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  /**
   * View conversation details
   */
  async viewConversation(id) {
    try {
      const conversation = await CogRepoAPI.getConversation(id);
      this._renderConversationModal(conversation);
      CogRepoUI.modal.open('conversationModal');
    } catch (error) {
      console.error('Failed to load conversation:', error);
      CogRepoUI.toast.error('Failed to load conversation');
    }
  }

  /**
   * Save current search
   */
  saveCurrentSearch() {
    const query = document.getElementById('searchInput')?.value;
    if (!query?.trim()) {
      CogRepoUI.toast.warning('Enter a search query first');
      return;
    }

    this.store.addSavedSearch({
      query: query.trim(),
      timestamp: new Date().toISOString()
    });

    this._renderSavedSearches();
    CogRepoUI.toast.success('Search saved!');
  }

  /**
   * Export results
   */
  async exportResults(format = 'json') {
    const { filteredResults } = this.store.getState();

    if (filteredResults.length === 0) {
      CogRepoUI.toast.warning('No results to export');
      return;
    }

    try {
      CogRepoUI.toast.info('Preparing export...');

      const ids = filteredResults.map(c => c.convo_id || c.external_id);
      const response = await CogRepoAPI.exportConversations(ids, format);

      // Create download
      const blob = new Blob([JSON.stringify(response.data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `cogrepo-export-${Date.now()}.json`;
      a.click();
      URL.revokeObjectURL(url);

      CogRepoUI.toast.success(`Exported ${ids.length} conversations`);

    } catch (error) {
      console.error('Export failed:', error);
      CogRepoUI.toast.error('Export failed');
    }
  }

  // =========================================================================
  // Setup Methods
  // =========================================================================

  _setupSearch() {
    const searchInput = document.getElementById('searchInput');
    const searchBtn = document.getElementById('searchBtn');
    const semanticBtn = document.getElementById('semanticSearchBtn');

    if (searchInput) {
      // Update placeholder with platform-specific shortcut
      const modKey = CogRepoUI.getModifierKey();
      searchInput.placeholder = `Search conversations... (${modKey}K)`;

      // Debounced real-time search
      const debouncedSearch = CogRepoUI.debounce((query) => {
        // Only search if query is 3+ chars or empty (to clear results)
        if (query.length >= 3 || query.length === 0) {
          this.performSearch(query);
        }
      }, 600); // Increased from 400ms to 600ms for better typing experience

      searchInput.addEventListener('input', (e) => {
        debouncedSearch(e.target.value);
      });

      // Enter key
      searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          this.performSearch(e.target.value);
        }
      });
    }

    if (searchBtn) {
      searchBtn.addEventListener('click', () => {
        this.performSearch(searchInput?.value || '');
      });
    }

    if (semanticBtn) {
      semanticBtn.addEventListener('click', () => {
        this.performSearch(searchInput?.value || '', { semantic: true });
      });
    }
  }

  _setupKeyboardShortcuts() {
    // Search focus (Cmd/Ctrl + K)
    CogRepoUI.shortcuts.register('mod+k', () => {
      document.getElementById('searchInput')?.focus();
    }, 'Focus search');

    // Save search (Cmd/Ctrl + S)
    CogRepoUI.shortcuts.register('mod+s', () => {
      this.saveCurrentSearch();
    }, 'Save search');

    // Export (Cmd/Ctrl + E)
    CogRepoUI.shortcuts.register('mod+e', () => {
      this.exportResults();
    }, 'Export results');

    // Clear search (Escape in search)
    CogRepoUI.shortcuts.register('escape', () => {
      const searchInput = document.getElementById('searchInput');
      if (document.activeElement === searchInput) {
        searchInput.value = '';
        this.performSearch('');
        searchInput.blur();
      }
    }, 'Clear search');

    // Navigate results (j/k vim-style)
    CogRepoUI.shortcuts.register('j', () => {
      this._navigateResults(1);
    }, 'Next result');

    CogRepoUI.shortcuts.register('k', () => {
      this._navigateResults(-1);
    }, 'Previous result');

    // Help (?)
    CogRepoUI.shortcuts.register('shift+/', () => {
      this._showKeyboardHelp();
    }, 'Show shortcuts');
  }

  _setupFilters() {
    const sourceFilter = document.getElementById('sourceFilter');
    const dateFrom = document.getElementById('dateFrom');
    const dateTo = document.getElementById('dateTo');
    const scoreFilter = document.getElementById('scoreFilter');

    const applyFilters = () => {
      this.store.setState({
        filters: {
          source: sourceFilter?.value || '',
          dateFrom: dateFrom?.value || '',
          dateTo: dateTo?.value || '',
          minScore: parseInt(scoreFilter?.value) || 0
        }
      });
      this.performSearch(this.store.state.searchQuery);
    };

    [sourceFilter, dateFrom, dateTo, scoreFilter].forEach(el => {
      if (el) el.addEventListener('change', applyFilters);
    });

    // Quick filter buttons
    document.querySelectorAll('[data-quick-filter]').forEach(btn => {
      btn.addEventListener('click', () => {
        const query = btn.dataset.quickFilter;
        document.getElementById('searchInput').value = query;
        this.performSearch(query);
      });
    });
  }

  _setupSavedSearches() {
    this._renderSavedSearches();

    // Save button
    document.getElementById('saveSearchBtn')?.addEventListener('click', () => {
      this.saveCurrentSearch();
    });
  }

  // =========================================================================
  // Render Methods
  // =========================================================================

  _render(state) {
    // State change handler - can be extended for reactive updates
  }

  _renderInitialState() {
    const resultsContainer = document.getElementById('results');
    if (!resultsContainer) return;

    const { stats, tags } = this.store.getState();

    resultsContainer.innerHTML = `
      ${renderStatsBar(stats) || ''}
      ${renderTagsCloud(tags)}
      <div class="initial-state">
        ${renderEmptyState('initial').outerHTML}
      </div>
    `;

    // Add click handlers to cloud tags
    resultsContainer.querySelectorAll('.cloud-tag').forEach(tag => {
      tag.addEventListener('click', () => {
        const tagValue = tag.dataset.tag;
        document.getElementById('searchInput').value = `tag:${tagValue}`;
        this.performSearch(`tag:${tagValue}`);
      });
    });
  }

  _renderResults(results, query) {
    const resultsContainer = document.getElementById('results');
    const resultsHeader = document.getElementById('resultsHeader');

    if (!resultsContainer) return;

    const { pagination } = this.store.getState();
    const startIndex = (pagination.currentPage - 1) * pagination.itemsPerPage;
    const endIndex = startIndex + pagination.itemsPerPage;
    const pageResults = results.slice(startIndex, endIndex);

    // Update header - show it when we have results or a query
    if (resultsHeader) {
      resultsHeader.style.display = (results.length > 0 || query) ? 'flex' : 'none';
      resultsHeader.innerHTML = `
        <div class="results-count">
          <span>${results.length.toLocaleString()}</span> results
          ${query ? `for "${CogRepoUI.escapeHTML(query)}"` : ''}
        </div>
        <div class="results-actions">
          <button class="btn btn-sm btn-secondary" id="saveSearchBtn" aria-label="Save this search">
            💾 Save
          </button>
          <button class="btn btn-sm btn-success" id="exportBtn" aria-label="Export results">
            📥 Export
          </button>
        </div>
      `;

      document.getElementById('saveSearchBtn')?.addEventListener('click', () => this.saveCurrentSearch());
      document.getElementById('exportBtn')?.addEventListener('click', () => this.exportResults());
    }

    // Render results
    if (pageResults.length === 0) {
      resultsContainer.innerHTML = renderEmptyState(query ? 'search' : 'initial').outerHTML;
      return;
    }

    const grid = CogRepoUI.createElement('div', { className: 'results-grid', role: 'list' });

    for (const result of pageResults) {
      grid.appendChild(renderConversationCard(result, query));
    }

    resultsContainer.innerHTML = '';
    resultsContainer.appendChild(grid);

    // Update pagination
    this.pagination.update(results.length, pagination.currentPage);
  }

  _renderSavedSearches() {
    const container = document.getElementById('savedSearches');
    if (!container) return;

    const { savedSearches } = this.store.getState();

    if (savedSearches.length === 0) {
      container.innerHTML = '';
      container.style.display = 'none';
      return;
    }

    container.style.display = 'block';
    container.innerHTML = `
      <div class="saved-searches-header">
        <span class="saved-searches-title">Saved Searches</span>
      </div>
      <div class="saved-search-list">
        ${savedSearches.map(s => `
          <div class="saved-search-item" data-query="${CogRepoUI.escapeHTML(s.query)}">
            <span>${CogRepoUI.escapeHTML(s.query)}</span>
            <button class="saved-search-delete" aria-label="Remove saved search" data-delete="${CogRepoUI.escapeHTML(s.query)}">×</button>
          </div>
        `).join('')}
      </div>
    `;

    // Click handlers
    container.querySelectorAll('.saved-search-item').forEach(item => {
      item.addEventListener('click', (e) => {
        if (e.target.classList.contains('saved-search-delete')) {
          e.stopPropagation();
          this.store.removeSavedSearch(e.target.dataset.delete);
          this._renderSavedSearches();
          return;
        }
        const query = item.dataset.query;
        document.getElementById('searchInput').value = query;
        this.performSearch(query);
      });
    });
  }

  _renderConversationModal(conversation) {
    const modalBody = document.querySelector('#conversationModal .modal-body');
    const modalTitle = document.querySelector('#conversationModal .modal-title');

    if (!modalBody || !modalTitle) return;

    const title = conversation.generated_title || conversation.title || 'Untitled';
    modalTitle.textContent = title;

    const messages = conversation.messages || [];
    const rawText = conversation.raw_text || '';

    modalBody.innerHTML = `
      <div class="conversation-modal-meta">
        <div class="meta-item">
          <span class="meta-label">Source</span>
          <span class="meta-value">${conversation.source || 'Unknown'}</span>
        </div>
        <div class="meta-item">
          <span class="meta-label">Date</span>
          <span class="meta-value">${CogRepoUI.formatDate(conversation.create_time)}</span>
        </div>
        ${conversation.score ? `
          <div class="meta-item">
            <span class="meta-label">Score</span>
            <span class="meta-value">${conversation.score}/10</span>
          </div>
        ` : ''}
        ${conversation.word_count ? `
          <div class="meta-item">
            <span class="meta-label">Words</span>
            <span class="meta-value">${conversation.word_count.toLocaleString()}</span>
          </div>
        ` : ''}
      </div>

      ${conversation.summary_abstractive ? `
        <div class="ai-insights">
          <h4 class="ai-insights-title">✨ AI Summary</h4>
          <p>${CogRepoUI.escapeHTML(conversation.summary_abstractive)}</p>
        </div>
      ` : ''}

      ${conversation.tags?.length ? `
        <div class="conversation-tags mb-6">
          ${conversation.tags.map(tag => `<span class="tag">${CogRepoUI.escapeHTML(tag)}</span>`).join('')}
        </div>
      ` : ''}

      ${conversation.advanced_score ? `
        <div class="ai-insights">
          <h4 class="ai-insights-title">📊 Advanced Score</h4>
          <div class="insights-grid">
            <div class="insight-item">
              <div class="insight-label">Overall Score</div>
              <div class="insight-value">${conversation.advanced_score.overall}/100 (${conversation.advanced_score.grade})</div>
            </div>
            ${conversation.advanced_score.dimensions ? Object.entries(conversation.advanced_score.dimensions).map(([dim, data]) => `
              <div class="insight-item">
                <div class="insight-label">${dim.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</div>
                <div class="insight-value">${data.value}/100</div>
              </div>
            `).join('') : ''}
          </div>
        </div>
      ` : ''}

      ${conversation.chains?.length ? `
        <div class="ai-insights">
          <h4 class="ai-insights-title">🔗 Related Conversations</h4>
          <div class="chains-list">
            ${conversation.chains.map(chain => `
              <a href="#" class="chain-link" data-id="${chain.id}">
                ${CogRepoUI.escapeHTML(chain.title || chain.id)}
              </a>
            `).join('')}
          </div>
        </div>
      ` : ''}

      ${conversation.projects?.length ? `
        <div class="ai-insights">
          <h4 class="ai-insights-title">📂 Projects</h4>
          <div class="projects-list">
            ${conversation.projects.map(proj => `
              <span class="tag tag-primary">${CogRepoUI.escapeHTML(proj.name || proj)}</span>
            `).join('')}
          </div>
        </div>
      ` : ''}

      <div class="messages-section">
        <h4 class="messages-title">Conversation</h4>
        <div class="messages-list">
          ${messages.length > 0 ? messages.map(msg => `
            <div class="message ${msg.role === 'user' || msg.role === 'human' ? 'user' : 'assistant'}">
              <div class="message-avatar ${msg.role === 'user' || msg.role === 'human' ? 'user' : 'assistant'}">
                ${msg.role === 'user' || msg.role === 'human' ? '👤' : '🤖'}
              </div>
              <div class="message-content">
                <div class="message-role">${CogRepoUI.escapeHTML(msg.role || 'Unknown')}</div>
                <div class="message-text">${CogRepoUI.escapeHTML(msg.content || '')}</div>
              </div>
            </div>
          `).join('') : `
            <div class="raw-text-container">
              <pre style="white-space: pre-wrap; font-family: inherit; font-size: var(--font-size-sm);">${CogRepoUI.escapeHTML(rawText || 'No content available')}</pre>
            </div>
          `}
        </div>
      </div>
    `;
  }

  _navigateResults(direction) {
    const cards = document.querySelectorAll('.conversation-card');
    if (cards.length === 0) return;

    const focused = document.activeElement;
    const currentIndex = Array.from(cards).indexOf(focused);

    let nextIndex;
    if (currentIndex === -1) {
      nextIndex = direction > 0 ? 0 : cards.length - 1;
    } else {
      nextIndex = currentIndex + direction;
      if (nextIndex < 0) nextIndex = cards.length - 1;
      if (nextIndex >= cards.length) nextIndex = 0;
    }

    cards[nextIndex].focus();
  }

  _showKeyboardHelp() {
    const shortcuts = CogRepoUI.shortcuts.getShortcutsList();

    const modal = document.getElementById('keyboardHelpModal');
    if (!modal) return;

    const body = modal.querySelector('.modal-body');
    if (body) {
      body.innerHTML = `
        <div class="shortcuts-list">
          ${shortcuts.map(s => `
            <div class="shortcut-item">
              <span class="shortcut-desc">${s.description}</span>
              <span class="shortcut-key">
                ${s.shortcut.split(' + ').map(k => `<kbd>${k}</kbd>`).join('')}
              </span>
            </div>
          `).join('')}
        </div>
      `;
    }

    CogRepoUI.modal.open('keyboardHelpModal');
  }
}

// =============================================================================
// Initialize Application
// =============================================================================

const app = new CogRepoApp();

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => app.init());
} else {
  app.init();
}

// Expose globally
window.app = app;

// =============================================================================
// SOTA Enrichment System Integration
// =============================================================================

import { EnrichmentManager, EnrichmentButton, EnrichmentPreviewModal } from './enrichment.js';

// Initialize enrichment system when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  // Initialize enrichment manager
  const enrichmentManager = new EnrichmentManager();

  // Set up event delegation for enrich buttons
  document.addEventListener('click', async (e) => {
    const enrichBtn = e.target.closest('.enrich-btn');
    if (!enrichBtn) return;

    e.stopPropagation();
    const conversationId = enrichBtn.dataset.conversationId;

    if (!conversationId) {
      console.error('No conversation ID found on enrich button');
      return;
    }

    try {
      // Disable button during enrichment
      enrichBtn.disabled = true;
      enrichBtn.classList.add('loading');

      // Update button text to show progress
      const originalHTML = enrichBtn.innerHTML;
      enrichBtn.innerHTML = '<span class="spinner"></span> Enriching...';

      // Start enrichment with streaming
      await enrichmentManager.enrichSingle(conversationId, ['title', 'summary', 'tags'], {
        onProgress: (update) => {
          // Show field being enriched
          if (!update.partial) {
            console.log(`✓ Enriched ${update.field}`);
          }
        },
        onComplete: async (result) => {
          enrichBtn.innerHTML = '✓ Enriched';
          enrichBtn.classList.remove('loading');
          enrichBtn.classList.add('success');

          // Get preview and show modal
          const preview = await enrichmentManager.getPreview(conversationId);
          if (preview) {
            const modal = new EnrichmentPreviewModal(conversationId, preview, enrichmentManager);
            modal.show();
          }

          // Re-enable button after a delay
          setTimeout(() => {
            enrichBtn.innerHTML = originalHTML;
            enrichBtn.disabled = false;
            enrichBtn.classList.remove('success');
          }, 2000);
        },
        onError: (error) => {
          enrichBtn.innerHTML = '❌ Failed';
          enrichBtn.classList.remove('loading');
          enrichBtn.classList.add('error');
          CogRepoUI.toast.error(`Enrichment failed: ${error.message}`);

          // Reset button after delay
          setTimeout(() => {
            enrichBtn.innerHTML = originalHTML;
            enrichBtn.disabled = false;
            enrichBtn.classList.remove('error');
          }, 3000);
        }
      });
    } catch (error) {
      console.error('Enrichment error:', error);
      CogRepoUI.toast.error(`Failed to enrich: ${error.message}`);
      enrichBtn.disabled = false;
      enrichBtn.classList.remove('loading');
    }
  });

  // Expose enrichment manager globally for debugging
  window.enrichmentManager = enrichmentManager;
});

export { CogRepoApp, Store, highlightText, renderConversationCard };
