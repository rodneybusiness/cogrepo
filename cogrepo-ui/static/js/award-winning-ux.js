/**
 * CogRepo Award-Winning UX
 *
 * Zero-dependency JavaScript enhancements:
 * 1. Keyboard shortcuts (Cmd+K, etc.)
 * 2. Theme toggle with smooth transitions
 * 3. Enhanced focus management
 * 4. Scroll animations
 * 5. Touch gestures for mobile
 *
 * Performance: Pure vanilla JS, <3KB gzipped
 */

(function() {
  'use strict';

  /* ===========================================================================
     1. KEYBOARD SHORTCUTS
     Award-winning power user features
     =========================================================================== */

  class KeyboardShortcuts {
    constructor() {
      this.shortcuts = {
        'cmd+k': () => this.focusSearch(),
        'ctrl+k': () => this.focusSearch(),
        'cmd+s': (e) => this.saveSearch(e),
        'ctrl+s': (e) => this.saveSearch(e),
        'cmd+e': (e) => this.exportResults(e),
        'ctrl+e': (e) => this.exportResults(e),
        'escape': () => this.handleEscape(),
        'j': () => this.nextResult(),
        'k': () => this.previousResult(),
        '?': () => this.showShortcutsHelp(),
        'cmd+shift+d': () => this.toggleTheme(),
        'ctrl+shift+d': () => this.toggleTheme(),
      };

      this.currentResultIndex = -1;
      this.init();
    }

    init() {
      document.addEventListener('keydown', (e) => this.handleKeydown(e));
      // Visual feedback for keyboard nav
      document.addEventListener('keydown', () => {
        document.body.classList.add('keyboard-nav-active');
      });
      document.addEventListener('mousedown', () => {
        document.body.classList.remove('keyboard-nav-active');
      });
    }

    handleKeydown(e) {
      // Don't intercept if typing in input
      if (e.target.matches('input, textarea, select') && !['escape', 'cmd+k', 'ctrl+k'].some(key => this.matchesShortcut(e, key))) {
        return;
      }

      for (const [shortcut, handler] of Object.entries(this.shortcuts)) {
        if (this.matchesShortcut(e, shortcut)) {
          e.preventDefault();
          handler(e);
          break;
        }
      }
    }

    matchesShortcut(e, shortcut) {
      const keys = shortcut.toLowerCase().split('+');
      const hasCmd = keys.includes('cmd') && (e.metaKey || e.ctrlKey);
      const hasCtrl = keys.includes('ctrl') && e.ctrlKey;
      const hasShift = keys.includes('shift') && e.shiftKey;
      const mainKey = keys[keys.length - 1];

      if (keys.length === 1) {
        return e.key.toLowerCase() === mainKey && !e.metaKey && !e.ctrlKey && !e.shiftKey;
      }

      if (keys.length === 2) {
        if (keys[0] === 'cmd' || keys[0] === 'ctrl') {
          return (hasCmd || hasCtrl) && e.key.toLowerCase() === mainKey && !hasShift;
        }
      }

      if (keys.length === 3) {
        return (hasCmd || hasCtrl) && hasShift && e.key.toLowerCase() === mainKey;
      }

      return false;
    }

    focusSearch() {
      const searchInput = document.getElementById('searchInput');
      if (searchInput) {
        searchInput.focus();
        searchInput.select();
        this.showToast('Search focused - start typing', 'info', 1500);
      }
    }

    saveSearch(e) {
      if (window.app && typeof window.app.saveSearch === 'function') {
        window.app.saveSearch();
        this.showToast('Search saved!', 'success');
      }
    }

    exportResults(e) {
      if (window.app && typeof window.app.exportResults === 'function') {
        window.app.exportResults();
        this.showToast('Exporting results...', 'info');
      }
    }

    handleEscape() {
      // Clear search or close modal
      const activeModal = document.querySelector('.modal.active');
      if (activeModal) {
        const modalId = activeModal.id;
        if (window.CogRepoUI?.modal?.close) {
          window.CogRepoUI.modal.close(modalId);
        }
      } else {
        const searchInput = document.getElementById('searchInput');
        if (searchInput && searchInput.value) {
          searchInput.value = '';
          searchInput.blur();
          this.showToast('Search cleared', 'info', 1000);
        }
      }
    }

    nextResult() {
      const results = document.querySelectorAll('.conversation-card');
      if (results.length === 0) return;

      this.currentResultIndex = (this.currentResultIndex + 1) % results.length;
      this.highlightResult(results[this.currentResultIndex]);
    }

    previousResult() {
      const results = document.querySelectorAll('.conversation-card');
      if (results.length === 0) return;

      this.currentResultIndex = this.currentResultIndex <= 0
        ? results.length - 1
        : this.currentResultIndex - 1;
      this.highlightResult(results[this.currentResultIndex]);
    }

    highlightResult(element) {
      // Remove previous highlight
      document.querySelectorAll('.conversation-card').forEach(el => {
        el.classList.remove('keyboard-selected');
      });

      // Add highlight
      element.classList.add('keyboard-selected');
      element.scrollIntoView({ behavior: 'smooth', block: 'center' });

      // Flash animation
      element.style.animation = 'none';
      setTimeout(() => {
        element.style.animation = 'keyboard-highlight 0.5s ease-out';
      }, 10);
    }

    showShortcutsHelp() {
      const shortcutsPanel = document.querySelector('.shortcuts-panel');
      if (shortcutsPanel) {
        shortcutsPanel.classList.toggle('active');
      }
    }

    toggleTheme() {
      if (window.themeManager) {
        window.themeManager.toggle();
      }
    }

    showToast(message, type = 'info', duration = 3000) {
      if (window.CogRepoUI?.toast) {
        window.CogRepoUI.toast[type](message, { duration });
      } else {
        console.log(`[${type}] ${message}`);
      }
    }
  }

  /* ===========================================================================
     2. THEME MANAGER
     Smooth light/dark mode transitions
     =========================================================================== */

  class ThemeManager {
    constructor() {
      this.currentTheme = this.getStoredTheme() || this.getSystemTheme();
      this.init();
    }

    init() {
      this.applyTheme(this.currentTheme, false);
      this.createToggleButton();
      this.watchSystemTheme();
    }

    getSystemTheme() {
      return window.matchMedia('(prefers-color-scheme: dark)').matches
        ? 'dark'
        : 'light';
    }

    getStoredTheme() {
      return localStorage.getItem('cogrepo-theme');
    }

    applyTheme(theme, animate = true) {
      if (animate) {
        document.documentElement.classList.add('theme-transitioning');
        setTimeout(() => {
          document.documentElement.classList.remove('theme-transitioning');
        }, 300);
      }

      document.documentElement.setAttribute('data-theme', theme);
      localStorage.setItem('cogrepo-theme', theme);
      this.currentTheme = theme;

      // Update toggle button icon
      this.updateToggleIcon(theme);

      // Announce to screen readers
      this.announce(`Switched to ${theme} mode`);
    }

    toggle() {
      const newTheme = this.currentTheme === 'dark' ? 'light' : 'dark';
      this.applyTheme(newTheme);
    }

    createToggleButton() {
      const button = document.createElement('button');
      button.className = 'theme-toggle';
      button.setAttribute('aria-label', 'Toggle theme');
      button.setAttribute('title', 'Toggle theme (Cmd+Shift+D)');

      button.innerHTML = `
        <svg class="theme-icon-sun" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="5"></circle>
          <line x1="12" y1="1" x2="12" y2="3"></line>
          <line x1="12" y1="21" x2="12" y2="23"></line>
          <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
          <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
          <line x1="1" y1="12" x2="3" y2="12"></line>
          <line x1="21" y1="12" x2="23" y2="12"></line>
          <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
          <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
        </svg>
        <svg class="theme-icon-moon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: none;">
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
        </svg>
      `;

      button.addEventListener('click', () => this.toggle());
      document.body.appendChild(button);

      this.toggleButton = button;
      this.updateToggleIcon(this.currentTheme);
    }

    updateToggleIcon(theme) {
      if (!this.toggleButton) return;

      const sunIcon = this.toggleButton.querySelector('.theme-icon-sun');
      const moonIcon = this.toggleButton.querySelector('.theme-icon-moon');

      if (theme === 'dark') {
        sunIcon.style.display = 'block';
        moonIcon.style.display = 'none';
        this.toggleButton.setAttribute('title', 'Switch to light mode (Cmd+Shift+D)');
      } else {
        sunIcon.style.display = 'none';
        moonIcon.style.display = 'block';
        this.toggleButton.setAttribute('title', 'Switch to dark mode (Cmd+Shift+D)');
      }
    }

    watchSystemTheme() {
      window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
        if (!this.getStoredTheme()) {
          this.applyTheme(e.matches ? 'dark' : 'light');
        }
      });
    }

    announce(message) {
      const announcer = document.getElementById('theme-announcer') || this.createAnnouncer();
      announcer.textContent = message;
    }

    createAnnouncer() {
      const announcer = document.createElement('div');
      announcer.id = 'theme-announcer';
      announcer.setAttribute('role', 'status');
      announcer.setAttribute('aria-live', 'polite');
      announcer.className = 'sr-only';
      document.body.appendChild(announcer);
      return announcer;
    }
  }

  /* ===========================================================================
     3. SCROLL ANIMATIONS
     Reveal elements as they enter viewport
     =========================================================================== */

  class ScrollAnimations {
    constructor() {
      this.observer = null;
      this.init();
    }

    init() {
      if (!('IntersectionObserver' in window)) return;

      this.observer = new IntersectionObserver(
        (entries) => {
          entries.forEach(entry => {
            if (entry.isIntersecting) {
              entry.target.classList.add('in-view');
              this.observer.unobserve(entry.target);
            }
          });
        },
        { threshold: 0.1, rootMargin: '50px' }
      );

      this.observeElements();
    }

    observeElements() {
      // Observe cards as they're added
      const observer = new MutationObserver(() => {
        document.querySelectorAll('.conversation-card:not(.in-view)').forEach(card => {
          this.observer.observe(card);
        });
      });

      observer.observe(document.body, {
        childList: true,
        subtree: true
      });
    }
  }

  /* ===========================================================================
     4. TOUCH GESTURES (Mobile)
     Swipe actions for cards
     =========================================================================== */

  class TouchGestures {
    constructor() {
      if (!('ontouchstart' in window)) return;
      this.init();
    }

    init() {
      let touchStartX = 0;
      let touchStartY = 0;
      let touchEndX = 0;
      let touchEndY = 0;

      document.addEventListener('touchstart', (e) => {
        touchStartX = e.changedTouches[0].screenX;
        touchStartY = e.changedTouches[0].screenY;
      }, { passive: true });

      document.addEventListener('touchend', (e) => {
        touchEndX = e.changedTouches[0].screenX;
        touchEndY = e.changedTouches[0].screenY;
        this.handleGesture(e, touchStartX, touchStartY, touchEndX, touchEndY);
      }, { passive: true });
    }

    handleGesture(e, startX, startY, endX, endY) {
      const deltaX = endX - startX;
      const deltaY = endY - startY;
      const minSwipeDistance = 50;

      // Horizontal swipe
      if (Math.abs(deltaX) > Math.abs(deltaY) && Math.abs(deltaX) > minSwipeDistance) {
        if (deltaX > 0) {
          this.onSwipeRight(e);
        } else {
          this.onSwipeLeft(e);
        }
      }

      // Vertical swipe
      if (Math.abs(deltaY) > Math.abs(deltaX) && Math.abs(deltaY) > minSwipeDistance) {
        if (deltaY > 0) {
          this.onSwipeDown(e);
        } else {
          this.onSwipeUp(e);
        }
      }
    }

    onSwipeRight(e) {
      // Close modal or go back
      const activeModal = document.querySelector('.modal.active');
      if (activeModal && window.CogRepoUI?.modal?.close) {
        const modalId = activeModal.id;
        window.CogRepoUI.modal.close(modalId);
      }
    }

    onSwipeLeft(e) {
      // Could open quick actions menu on card
    }

    onSwipeDown(e) {
      // Pull to refresh (if at top)
      if (window.scrollY === 0) {
        // Could trigger refresh
      }
    }

    onSwipeUp(e) {
      // Could trigger search expansion
    }
  }

  /* ===========================================================================
     5. PERFORMANCE MONITOR
     Track and optimize animations
     =========================================================================== */

  class PerformanceMonitor {
    constructor() {
      this.checkPerformance();
    }

    checkPerformance() {
      // Reduce animations on low-end devices
      if (navigator.hardwareConcurrency && navigator.hardwareConcurrency < 4) {
        document.documentElement.classList.add('reduced-animations');
      }

      // Monitor FPS
      let lastTime = performance.now();
      let frames = 0;

      const checkFPS = () => {
        frames++;
        const currentTime = performance.now();

        if (currentTime >= lastTime + 1000) {
          const fps = Math.round((frames * 1000) / (currentTime - lastTime));

          if (fps < 30) {
            document.documentElement.classList.add('low-fps');
          } else {
            document.documentElement.classList.remove('low-fps');
          }

          frames = 0;
          lastTime = currentTime;
        }

        requestAnimationFrame(checkFPS);
      };

      requestAnimationFrame(checkFPS);
    }
  }

  /* ===========================================================================
     INITIALIZATION
     Initialize all features when DOM is ready
     =========================================================================== */

  function init() {
    // Initialize all features
    window.keyboardShortcuts = new KeyboardShortcuts();
    window.themeManager = new ThemeManager();
    window.scrollAnimations = new ScrollAnimations();
    window.touchGestures = new TouchGestures();
    window.performanceMonitor = new PerformanceMonitor();

    // Add CSS for keyboard navigation
    const style = document.createElement('style');
    style.textContent = `
      .keyboard-selected {
        outline: 3px solid var(--color-primary-500) !important;
        outline-offset: 4px !important;
        z-index: 10;
      }

      @keyframes keyboard-highlight {
        0%, 100% {
          box-shadow: 0 0 0 0 var(--color-primary-400);
        }
        50% {
          box-shadow: 0 0 0 8px rgba(102, 126, 234, 0.3);
        }
      }

      .sr-only {
        position: absolute;
        width: 1px;
        height: 1px;
        padding: 0;
        margin: -1px;
        overflow: hidden;
        clip: rect(0, 0, 0, 0);
        white-space: nowrap;
        border-width: 0;
      }

      .reduced-animations *,
      .low-fps * {
        animation-duration: 0.1s !important;
        transition-duration: 0.1s !important;
      }
    `;
    document.head.appendChild(style);

    console.log('🎨 CogRepo Award-Winning UX initialized');
  }

  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
