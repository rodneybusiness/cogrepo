/**
 * CogRepo API Module
 *
 * Handles all API communication with proper error handling,
 * request cancellation, and retry logic.
 */

class APIError extends Error {
  constructor(message, status, data) {
    super(message);
    this.name = 'APIError';
    this.status = status;
    this.data = data;
  }
}

class API {
  constructor(baseURL = null) {
    // Auto-detect base URL from current location
    this.baseURL = baseURL || this._detectBaseURL();
    this.defaultTimeout = 30000; // 30 seconds
    this.activeRequests = new Map();
  }

  _detectBaseURL() {
    // Try to use same origin, fallback to localhost
    if (window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
      return window.location.origin;
    }
    // Check if we're running from file or need different port
    const port = window.location.port || '8000';
    return `http://localhost:${port}`;
  }

  /**
   * Make an API request with automatic retries and cancellation support
   */
  async request(endpoint, options = {}) {
    const {
      method = 'GET',
      body = null,
      headers = {},
      timeout = this.defaultTimeout,
      retries = 3,
      retryDelay = 1000,
      signal = null,
      requestId = null
    } = options;

    // Create abort controller for timeout and cancellation
    const controller = new AbortController();
    const combinedSignal = signal || controller.signal;

    // Store active request for cancellation
    if (requestId) {
      // Cancel existing request with same ID
      this.cancelRequest(requestId);
      this.activeRequests.set(requestId, controller);
    }

    // Set timeout
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    const url = `${this.baseURL}${endpoint}`;
    const config = {
      method,
      headers: {
        'Content-Type': 'application/json',
        ...headers
      },
      signal: combinedSignal
    };

    if (body && method !== 'GET') {
      config.body = JSON.stringify(body);
    }

    let lastError;

    for (let attempt = 0; attempt < retries; attempt++) {
      try {
        const response = await fetch(url, config);

        clearTimeout(timeoutId);

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new APIError(
            errorData.message || `HTTP ${response.status}: ${response.statusText}`,
            response.status,
            errorData
          );
        }

        const data = await response.json();

        // Clean up active request
        if (requestId) {
          this.activeRequests.delete(requestId);
        }

        return data;
      } catch (error) {
        lastError = error;

        // Don't retry on abort or client errors
        if (error.name === 'AbortError' || (error.status && error.status < 500)) {
          break;
        }

        // Wait before retry with exponential backoff
        if (attempt < retries - 1) {
          await new Promise(resolve => setTimeout(resolve, retryDelay * Math.pow(2, attempt)));
        }
      }
    }

    clearTimeout(timeoutId);
    if (requestId) {
      this.activeRequests.delete(requestId);
    }

    throw lastError;
  }

  /**
   * Cancel an active request by ID
   */
  cancelRequest(requestId) {
    const controller = this.activeRequests.get(requestId);
    if (controller) {
      controller.abort();
      this.activeRequests.delete(requestId);
    }
  }

  /**
   * Cancel all active requests
   */
  cancelAllRequests() {
    for (const [id, controller] of this.activeRequests) {
      controller.abort();
    }
    this.activeRequests.clear();
  }

  // =========================================================================
  // Conversation Endpoints
  // =========================================================================

  /**
   * Get all conversations with optional pagination
   */
  async getConversations({ page = 1, limit = 50, sortBy = 'date', sortOrder = 'desc' } = {}) {
    const params = new URLSearchParams({ page, limit, sort_by: sortBy, sort_order: sortOrder });
    return this.request(`/api/conversations?${params}`);
  }

  /**
   * Get a single conversation by ID
   */
  async getConversation(id) {
    return this.request(`/api/conversation/${encodeURIComponent(id)}`);
  }

  /**
   * Search conversations
   */
  async search(query, filters = {}, options = {}) {
    const {
      page = 1,
      limit = 25,
      semantic = false
    } = options;

    const endpoint = semantic ? '/api/semantic_search' : '/api/search';
    const params = new URLSearchParams({
      q: query,
      page,
      limit,
      ...filters
    });

    return this.request(`${endpoint}?${params}`, {
      requestId: 'search', // Allow cancellation of previous searches
      timeout: semantic ? 60000 : 30000 // Semantic search may take longer
    });
  }

  /**
   * Get search suggestions/autocomplete
   */
  async getSuggestions(query) {
    const params = new URLSearchParams({ q: query, limit: 10 });
    return this.request(`/api/suggestions?${params}`, {
      requestId: 'suggestions',
      retries: 1
    });
  }

  // =========================================================================
  // Stats & Metadata
  // =========================================================================

  /**
   * Get repository statistics
   */
  async getStats() {
    return this.request('/api/stats', {
      requestId: 'stats'
    });
  }

  /**
   * Get all tags with counts
   */
  async getTags() {
    return this.request('/api/tags');
  }

  /**
   * Get available sources
   */
  async getSources() {
    return this.request('/api/sources');
  }

  // =========================================================================
  // Export
  // =========================================================================

  /**
   * Export conversations
   */
  async exportConversations(ids, format = 'json') {
    return this.request('/api/export', {
      method: 'POST',
      body: { conversation_ids: ids, format },
      timeout: 120000 // 2 minutes for large exports
    });
  }

  /**
   * Download export file
   */
  getExportDownloadURL(exportId) {
    return `${this.baseURL}/api/export/download/${exportId}`;
  }

  // =========================================================================
  // Import
  // =========================================================================

  /**
   * Upload file for import
   */
  async uploadFile(file, onProgress = null) {
    const formData = new FormData();
    formData.append('file', file);

    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();

      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable && onProgress) {
          onProgress(Math.round((e.loaded / e.total) * 100));
        }
      });

      xhr.addEventListener('load', () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText));
          } catch {
            resolve({ success: true });
          }
        } else {
          reject(new APIError('Upload failed', xhr.status, xhr.responseText));
        }
      });

      xhr.addEventListener('error', () => {
        reject(new APIError('Network error during upload', 0));
      });

      xhr.open('POST', `${this.baseURL}/api/upload`);
      xhr.send(formData);
    });
  }

  /**
   * Start import process
   */
  async startImport(options = {}) {
    return this.request('/api/import', {
      method: 'POST',
      body: options
    });
  }
}

// Create singleton instance
const api = new API();

// Export for ES modules
export { API, APIError, api };

// Also expose globally for non-module scripts
if (typeof window !== 'undefined') {
  window.CogRepoAPI = api;
}
