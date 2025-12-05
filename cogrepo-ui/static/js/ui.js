/**
 * CogRepo UI Module
 *
 * Core UI utilities: modals, toasts, loading states, keyboard shortcuts.
 * Optimized for macOS with Cmd-key shortcuts.
 */

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Safely escape HTML to prevent XSS
 */
function escapeHTML(str) {
  if (!str) return '';
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

/**
 * Create element with attributes and children
 */
function createElement(tag, attrs = {}, children = []) {
  const el = document.createElement(tag);

  for (const [key, value] of Object.entries(attrs)) {
    if (key === 'className') {
      el.className = value;
    } else if (key === 'dataset') {
      Object.assign(el.dataset, value);
    } else if (key.startsWith('on') && typeof value === 'function') {
      el.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (key === 'innerHTML') {
      el.innerHTML = value;
    } else {
      el.setAttribute(key, value);
    }
  }

  for (const child of children) {
    if (typeof child === 'string') {
      el.appendChild(document.createTextNode(child));
    } else if (child instanceof Node) {
      el.appendChild(child);
    }
  }

  return el;
}

/**
 * Debounce function calls
 */
function debounce(fn, delay = 300) {
  let timeoutId;
  return function (...args) {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn.apply(this, args), delay);
  };
}

/**
 * Throttle function calls
 */
function throttle(fn, limit = 100) {
  let inThrottle;
  return function (...args) {
    if (!inThrottle) {
      fn.apply(this, args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  };
}

/**
 * Format date for display
 */
function formatDate(dateStr) {
  if (!dateStr) return 'Unknown';
  const date = new Date(dateStr);
  if (isNaN(date.getTime())) return dateStr;

  return new Intl.DateTimeFormat('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  }).format(date);
}

/**
 * Format relative time
 */
function formatRelativeTime(dateStr) {
  if (!dateStr) return '';
  const date = new Date(dateStr);
  if (isNaN(date.getTime())) return '';

  const rtf = new Intl.RelativeTimeFormat('en', { numeric: 'auto' });
  const diff = date - new Date();
  const days = Math.round(diff / (1000 * 60 * 60 * 24));

  if (Math.abs(days) < 1) {
    const hours = Math.round(diff / (1000 * 60 * 60));
    if (Math.abs(hours) < 1) {
      const minutes = Math.round(diff / (1000 * 60));
      return rtf.format(minutes, 'minute');
    }
    return rtf.format(hours, 'hour');
  } else if (Math.abs(days) < 30) {
    return rtf.format(days, 'day');
  } else if (Math.abs(days) < 365) {
    return rtf.format(Math.round(days / 30), 'month');
  }
  return rtf.format(Math.round(days / 365), 'year');
}

/**
 * Detect if running on macOS
 */
function isMac() {
  return navigator.platform.toUpperCase().indexOf('MAC') >= 0 ||
         navigator.userAgent.toUpperCase().indexOf('MAC') >= 0;
}

/**
 * Get modifier key name (Cmd for Mac, Ctrl for others)
 */
function getModifierKey() {
  return isMac() ? '⌘' : 'Ctrl';
}

// =============================================================================
// Modal Manager
// =============================================================================

class ModalManager {
  constructor() {
    this.activeModals = [];
    this.focusTrapElements = new WeakMap();
    this._setupGlobalListeners();
  }

  _setupGlobalListeners() {
    // Close on Escape
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && this.activeModals.length > 0) {
        e.preventDefault();
        this.close(this.activeModals[this.activeModals.length - 1]);
      }
    });

    // Close on backdrop click
    document.addEventListener('click', (e) => {
      if (e.target.classList.contains('modal-backdrop') && e.target.classList.contains('active')) {
        const modalId = e.target.dataset.modalId;
        if (modalId) {
          this.close(modalId);
        }
      }
    });
  }

  /**
   * Open a modal by ID
   */
  open(modalId, options = {}) {
    const modal = document.getElementById(modalId);
    const backdrop = document.getElementById(`${modalId}-backdrop`) ||
                     document.querySelector(`[data-modal-id="${modalId}"]`);

    if (!modal) {
      console.error(`Modal ${modalId} not found`);
      return;
    }

    // Store previously focused element
    this.focusTrapElements.set(modal, document.activeElement);

    // Show backdrop and modal
    if (backdrop) {
      backdrop.classList.add('active');
    }
    modal.classList.add('active');

    // Prevent body scroll
    document.body.style.overflow = 'hidden';

    // Focus first focusable element
    requestAnimationFrame(() => {
      const focusable = modal.querySelector(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      );
      if (focusable) focusable.focus();
    });

    // Setup focus trap
    this._setupFocusTrap(modal);

    this.activeModals.push(modalId);

    // Call onOpen callback
    if (options.onOpen) options.onOpen(modal);

    return modal;
  }

  /**
   * Close a modal by ID
   */
  close(modalId) {
    const modal = document.getElementById(modalId);
    const backdrop = document.getElementById(`${modalId}-backdrop`) ||
                     document.querySelector(`[data-modal-id="${modalId}"]`);

    if (!modal) return;

    // Hide backdrop and modal
    if (backdrop) {
      backdrop.classList.remove('active');
    }
    modal.classList.remove('active');

    // Restore body scroll if no more modals
    const index = this.activeModals.indexOf(modalId);
    if (index > -1) {
      this.activeModals.splice(index, 1);
    }

    if (this.activeModals.length === 0) {
      document.body.style.overflow = '';
    }

    // Restore focus
    const previousFocus = this.focusTrapElements.get(modal);
    if (previousFocus && previousFocus.focus) {
      previousFocus.focus();
    }

    this._removeFocusTrap(modal);
  }

  /**
   * Close all modals
   */
  closeAll() {
    [...this.activeModals].forEach(id => this.close(id));
  }

  _setupFocusTrap(modal) {
    const focusableElements = modal.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );

    if (focusableElements.length === 0) return;

    const firstFocusable = focusableElements[0];
    const lastFocusable = focusableElements[focusableElements.length - 1];

    const trapHandler = (e) => {
      if (e.key !== 'Tab') return;

      if (e.shiftKey && document.activeElement === firstFocusable) {
        e.preventDefault();
        lastFocusable.focus();
      } else if (!e.shiftKey && document.activeElement === lastFocusable) {
        e.preventDefault();
        firstFocusable.focus();
      }
    };

    modal._focusTrapHandler = trapHandler;
    modal.addEventListener('keydown', trapHandler);
  }

  _removeFocusTrap(modal) {
    if (modal._focusTrapHandler) {
      modal.removeEventListener('keydown', modal._focusTrapHandler);
      delete modal._focusTrapHandler;
    }
  }
}

// =============================================================================
// Toast Notifications
// =============================================================================

class ToastManager {
  constructor() {
    this.container = null;
    this.toasts = [];
    this._createContainer();
  }

  _createContainer() {
    this.container = document.getElementById('toast-container');
    if (!this.container) {
      this.container = createElement('div', {
        id: 'toast-container',
        className: 'toast-container',
        'aria-live': 'polite',
        'aria-atomic': 'true'
      });
      document.body.appendChild(this.container);
    }
  }

  /**
   * Show a toast notification
   */
  show(message, options = {}) {
    const {
      type = 'info', // info, success, warning, danger
      duration = 4000,
      action = null, // { label: 'Undo', onClick: () => {} }
      icon = null
    } = options;

    const icons = {
      info: '💡',
      success: '✓',
      warning: '⚠',
      danger: '✕'
    };

    const toast = createElement('div', {
      className: `toast toast-${type}`,
      role: 'alert'
    }, [
      createElement('span', { className: 'toast-icon' }, [icon || icons[type] || '']),
      createElement('span', { className: 'toast-message' }, [message])
    ]);

    if (action) {
      const actionBtn = createElement('button', {
        className: 'toast-action btn btn-sm btn-ghost',
        onClick: () => {
          action.onClick();
          this.dismiss(toast);
        }
      }, [action.label]);
      toast.appendChild(actionBtn);
    }

    const closeBtn = createElement('button', {
      className: 'toast-close',
      'aria-label': 'Dismiss notification',
      onClick: () => this.dismiss(toast)
    }, ['×']);
    toast.appendChild(closeBtn);

    this.container.appendChild(toast);
    this.toasts.push(toast);

    // Auto dismiss
    if (duration > 0) {
      setTimeout(() => this.dismiss(toast), duration);
    }

    return toast;
  }

  /**
   * Dismiss a toast
   */
  dismiss(toast) {
    if (!toast || !toast.parentNode) return;

    toast.style.animation = 'slide-out 0.3s ease-out forwards';
    setTimeout(() => {
      toast.remove();
      const index = this.toasts.indexOf(toast);
      if (index > -1) this.toasts.splice(index, 1);
    }, 300);
  }

  /**
   * Dismiss all toasts
   */
  dismissAll() {
    [...this.toasts].forEach(toast => this.dismiss(toast));
  }

  // Convenience methods
  info(message, options = {}) {
    return this.show(message, { ...options, type: 'info' });
  }

  success(message, options = {}) {
    return this.show(message, { ...options, type: 'success' });
  }

  warning(message, options = {}) {
    return this.show(message, { ...options, type: 'warning' });
  }

  error(message, options = {}) {
    return this.show(message, { ...options, type: 'danger' });
  }
}

// =============================================================================
// Loading States
// =============================================================================

class LoadingManager {
  /**
   * Show loading overlay on element
   */
  showOverlay(element, message = 'Loading...') {
    const overlay = createElement('div', { className: 'loading-overlay' }, [
      createElement('div', { className: 'loading-content' }, [
        createElement('div', { className: 'spinner' }),
        message ? createElement('p', { className: 'loading-message' }, [message]) : null
      ].filter(Boolean))
    ]);

    element.style.position = 'relative';
    element.appendChild(overlay);
    return overlay;
  }

  /**
   * Hide loading overlay
   */
  hideOverlay(element) {
    const overlay = element.querySelector('.loading-overlay');
    if (overlay) {
      overlay.remove();
    }
  }

  /**
   * Create skeleton placeholders
   */
  createSkeleton(type = 'card', count = 1) {
    const skeletons = [];

    for (let i = 0; i < count; i++) {
      let skeleton;

      switch (type) {
        case 'card':
          skeleton = createElement('div', { className: 'card skeleton-card' }, [
            createElement('div', { className: 'card-header' }, [
              createElement('div', { className: 'skeleton skeleton-text', style: 'width: 40%' }),
              createElement('div', { className: 'skeleton skeleton-text', style: 'width: 20%' })
            ]),
            createElement('div', { className: 'card-body' }, [
              createElement('div', { className: 'skeleton skeleton-title' }),
              createElement('div', { className: 'skeleton skeleton-text' }),
              createElement('div', { className: 'skeleton skeleton-text', style: 'width: 80%' }),
              createElement('div', { className: 'skeleton skeleton-text', style: 'width: 60%' })
            ])
          ]);
          break;

        case 'text':
          skeleton = createElement('div', { className: 'skeleton skeleton-text' });
          break;

        case 'title':
          skeleton = createElement('div', { className: 'skeleton skeleton-title' });
          break;

        default:
          skeleton = createElement('div', { className: 'skeleton', style: `height: ${type}` });
      }

      skeletons.push(skeleton);
    }

    return skeletons;
  }
}

// =============================================================================
// Keyboard Shortcuts
// =============================================================================

class KeyboardShortcuts {
  constructor() {
    this.shortcuts = new Map();
    this.enabled = true;
    this._setupListener();
  }

  _setupListener() {
    document.addEventListener('keydown', (e) => {
      if (!this.enabled) return;

      // Don't trigger shortcuts when typing in inputs
      if (e.target.matches('input, textarea, select, [contenteditable]')) {
        // Allow Escape to blur inputs
        if (e.key === 'Escape') {
          e.target.blur();
        }
        return;
      }

      const key = this._normalizeKey(e);
      const handler = this.shortcuts.get(key);

      if (handler) {
        e.preventDefault();
        handler(e);
      }
    });
  }

  _normalizeKey(e) {
    const parts = [];

    // Use metaKey for Cmd on Mac, ctrlKey for Ctrl on others
    if (e.metaKey || e.ctrlKey) parts.push('mod');
    if (e.shiftKey) parts.push('shift');
    if (e.altKey) parts.push('alt');

    // Normalize key name
    let key = e.key.toLowerCase();
    if (key === ' ') key = 'space';

    parts.push(key);
    return parts.join('+');
  }

  /**
   * Register a keyboard shortcut
   */
  register(shortcut, handler, description = '') {
    // Normalize shortcut string
    const normalized = shortcut.toLowerCase()
      .replace('cmd', 'mod')
      .replace('ctrl', 'mod')
      .replace(/\s+/g, '');

    this.shortcuts.set(normalized, handler);

    // Store description for help display
    if (description) {
      handler._description = description;
      handler._shortcut = shortcut;
    }
  }

  /**
   * Unregister a keyboard shortcut
   */
  unregister(shortcut) {
    const normalized = shortcut.toLowerCase()
      .replace('cmd', 'mod')
      .replace('ctrl', 'mod')
      .replace(/\s+/g, '');
    this.shortcuts.delete(normalized);
  }

  /**
   * Get all registered shortcuts with descriptions
   */
  getShortcutsList() {
    const list = [];
    for (const [key, handler] of this.shortcuts) {
      if (handler._description) {
        // Format shortcut for display
        const displayKey = key
          .replace('mod', getModifierKey())
          .replace('shift', '⇧')
          .replace('alt', isMac() ? '⌥' : 'Alt')
          .split('+')
          .map(k => k.charAt(0).toUpperCase() + k.slice(1))
          .join(' + ');

        list.push({
          shortcut: displayKey,
          description: handler._description
        });
      }
    }
    return list;
  }

  /**
   * Enable/disable shortcuts
   */
  setEnabled(enabled) {
    this.enabled = enabled;
  }
}

// =============================================================================
// Pagination
// =============================================================================

class Pagination {
  constructor(options = {}) {
    this.currentPage = 1;
    this.totalPages = 1;
    this.itemsPerPage = options.itemsPerPage || 25;
    this.totalItems = 0;
    this.onChange = options.onChange || (() => {});
    this.container = null;
  }

  /**
   * Update pagination state
   */
  update(totalItems, currentPage = 1) {
    this.totalItems = totalItems;
    this.totalPages = Math.ceil(totalItems / this.itemsPerPage);
    this.currentPage = Math.min(currentPage, this.totalPages) || 1;
    this.render();
  }

  /**
   * Go to specific page
   */
  goToPage(page) {
    if (page < 1 || page > this.totalPages) return;
    this.currentPage = page;
    this.render();
    this.onChange(page);
  }

  /**
   * Render pagination controls
   */
  render() {
    if (!this.container) return;

    this.container.innerHTML = '';

    if (this.totalPages <= 1) {
      this.container.style.display = 'none';
      return;
    }

    this.container.style.display = 'flex';

    // Previous button
    const prevBtn = createElement('button', {
      className: 'pagination-btn',
      disabled: this.currentPage === 1,
      'aria-label': 'Previous page',
      onClick: () => this.goToPage(this.currentPage - 1)
    }, ['←']);
    this.container.appendChild(prevBtn);

    // Page numbers
    const pages = this._getPageNumbers();
    for (const page of pages) {
      if (page === '...') {
        this.container.appendChild(
          createElement('span', { className: 'pagination-ellipsis' }, ['...'])
        );
      } else {
        const pageBtn = createElement('button', {
          className: `pagination-btn ${page === this.currentPage ? 'active' : ''}`,
          'aria-label': `Page ${page}`,
          'aria-current': page === this.currentPage ? 'page' : null,
          onClick: () => this.goToPage(page)
        }, [String(page)]);
        this.container.appendChild(pageBtn);
      }
    }

    // Next button
    const nextBtn = createElement('button', {
      className: 'pagination-btn',
      disabled: this.currentPage === this.totalPages,
      'aria-label': 'Next page',
      onClick: () => this.goToPage(this.currentPage + 1)
    }, ['→']);
    this.container.appendChild(nextBtn);
  }

  _getPageNumbers() {
    const pages = [];
    const { currentPage, totalPages } = this;

    if (totalPages <= 7) {
      for (let i = 1; i <= totalPages; i++) pages.push(i);
    } else {
      pages.push(1);

      if (currentPage > 3) pages.push('...');

      const start = Math.max(2, currentPage - 1);
      const end = Math.min(totalPages - 1, currentPage + 1);

      for (let i = start; i <= end; i++) pages.push(i);

      if (currentPage < totalPages - 2) pages.push('...');

      pages.push(totalPages);
    }

    return pages;
  }

  /**
   * Mount pagination to container
   */
  mount(container) {
    this.container = typeof container === 'string'
      ? document.getElementById(container)
      : container;
    this.render();
  }
}

// =============================================================================
// Initialize Singletons
// =============================================================================

const modal = new ModalManager();
const toast = new ToastManager();
const loading = new LoadingManager();
const shortcuts = new KeyboardShortcuts();

// =============================================================================
// Exports
// =============================================================================

export {
  // Utilities
  escapeHTML,
  createElement,
  debounce,
  throttle,
  formatDate,
  formatRelativeTime,
  isMac,
  getModifierKey,

  // Classes
  ModalManager,
  ToastManager,
  LoadingManager,
  KeyboardShortcuts,
  Pagination,

  // Singleton instances
  modal,
  toast,
  loading,
  shortcuts
};

// Global exposure for non-module usage
if (typeof window !== 'undefined') {
  window.CogRepoUI = {
    escapeHTML,
    createElement,
    debounce,
    throttle,
    formatDate,
    formatRelativeTime,
    isMac,
    getModifierKey,
    modal,
    toast,
    loading,
    shortcuts,
    Pagination
  };
}
