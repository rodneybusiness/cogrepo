/**
 * CogRepo Spectacular Interactions
 *
 * Award-winning interactive features custom-designed for AI conversation intelligence
 * Zero dependencies, pure vanilla JavaScript
 */

(function() {
  'use strict';

  /* ===========================================================================
     1. INTELLIGENCE SCORE VISUALIZATION
     Animated circular progress indicators
     =========================================================================== */

  class ScoreVisualizer {
    constructor() {
      this.init();
    }

    init() {
      // Observe score badges and animate them when they appear
      if ('IntersectionObserver' in window) {
        const observer = new IntersectionObserver(
          (entries) => {
            entries.forEach(entry => {
              if (entry.isIntersecting) {
                this.animateScore(entry.target);
                observer.unobserve(entry.target);
              }
            });
          },
          { threshold: 0.5 }
        );

        document.querySelectorAll('.intelligence-score-badge').forEach(badge => {
          observer.observe(badge);
        });
      }
    }

    animateScore(badge) {
      const scoreElement = badge.querySelector('.score-number');
      if (!scoreElement) return;

      const targetScore = parseInt(scoreElement.textContent);
      const circle = badge.querySelector('.score-circle-progress');

      if (!circle) return;

      // Calculate circumference
      const radius = circle.r.baseVal.value;
      const circumference = 2 * Math.PI * radius;

      // Set initial state
      circle.style.strokeDasharray = `${circumference} ${circumference}`;
      circle.style.strokeDashoffset = circumference;

      // Determine color based on score
      let scoreLevel = 'low';
      if (targetScore >= 80) scoreLevel = 'exceptional';
      else if (targetScore >= 60) scoreLevel = 'high';
      else if (targetScore >= 40) scoreLevel = 'medium';

      circle.setAttribute('data-score', scoreLevel);

      // Animate the circle
      setTimeout(() => {
        const offset = circumference - (targetScore / 100) * circumference;
        circle.style.strokeDashoffset = offset;
      }, 100);

      // Animate the number
      this.animateNumber(scoreElement, 0, targetScore, 1000);
    }

    animateNumber(element, start, end, duration) {
      const startTime = performance.now();

      const updateNumber = (currentTime) => {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Easing function
        const easeOut = 1 - Math.pow(1 - progress, 3);
        const current = Math.floor(start + (end - start) * easeOut);

        element.textContent = current;

        if (progress < 1) {
          requestAnimationFrame(updateNumber);
        }
      };

      requestAnimationFrame(updateNumber);
    }

    createScoreBadge(score) {
      const circumference = 2 * Math.PI * 28; // radius of 28

      return `
        <div class="intelligence-score-badge">
          <svg class="score-circle" viewBox="0 0 64 64">
            <circle class="score-circle-bg" cx="32" cy="32" r="28"></circle>
            <circle class="score-circle-progress" cx="32" cy="32" r="28"
                    stroke-dasharray="${circumference} ${circumference}"
                    stroke-dashoffset="${circumference}">
            </circle>
          </svg>
          <div class="score-number">${score}</div>
        </div>
      `;
    }
  }

  /* ===========================================================================
     2. HERO SEARCH ENHANCEMENT
     Magic search experience with live suggestions
     =========================================================================== */

  class HeroSearch {
    constructor() {
      this.searchInput = document.querySelector('.hero-search-input');
      if (!this.searchInput) return;

      this.init();
    }

    init() {
      // Add search suggestions
      this.searchInput.addEventListener('input', (e) => {
        this.handleSearch(e.target.value);
      });

      // Add keyboard shortcuts
      document.addEventListener('keydown', (e) => {
        if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
          e.preventDefault();
          this.searchInput.focus();
          this.searchInput.select();
        }
      });

      // Add sparkle effect on focus
      this.searchInput.addEventListener('focus', () => {
        this.addSparkles();
      });
    }

    handleSearch(query) {
      if (query.length < 2) return;

      // Trigger search with debouncing
      clearTimeout(this.searchTimeout);
      this.searchTimeout = setTimeout(() => {
        this.performSearch(query);
      }, 300);
    }

    performSearch(query) {
      // Integrate with existing search functionality
      if (window.app && typeof window.app.performSearch === 'function') {
        window.app.performSearch(query);
      }
    }

    addSparkles() {
      const container = this.searchInput.parentElement;

      // Create sparkle effect
      for (let i = 0; i < 5; i++) {
        setTimeout(() => {
          this.createSparkle(container);
        }, i * 100);
      }
    }

    createSparkle(container) {
      const sparkle = document.createElement('div');
      sparkle.style.cssText = `
        position: absolute;
        width: 4px;
        height: 4px;
        background: white;
        border-radius: 50%;
        pointer-events: none;
        animation: sparkle-fade 1s ease-out forwards;
      `;

      const rect = container.getBoundingClientRect();
      sparkle.style.left = `${Math.random() * rect.width}px`;
      sparkle.style.top = `${Math.random() * rect.height}px`;

      container.appendChild(sparkle);

      setTimeout(() => sparkle.remove(), 1000);
    }
  }

  /* ===========================================================================
     3. PREMIUM CARD INTERACTIONS
     Enhanced hover effects and click animations
     =========================================================================== */

  class PremiumCards {
    constructor() {
      this.init();
    }

    init() {
      // Add 3D tilt effect on mouse move
      document.addEventListener('mousemove', (e) => {
        const cards = document.querySelectorAll('.conversation-card-premium:hover');
        cards.forEach(card => {
          this.tiltCard(card, e);
        });
      });

      // Add click ripple effect
      document.addEventListener('click', (e) => {
        const card = e.target.closest('.conversation-card-premium');
        if (card) {
          this.addRipple(card, e);
        }
      });
    }

    tiltCard(card, event) {
      const rect = card.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;

      const centerX = rect.width / 2;
      const centerY = rect.height / 2;

      const rotateX = (y - centerY) / 20;
      const rotateY = (centerX - x) / 20;

      card.style.transform = `
        translateY(-8px)
        scale(1.02)
        perspective(1000px)
        rotateX(${rotateX}deg)
        rotateY(${rotateY}deg)
      `;
    }

    addRipple(card, event) {
      const ripple = document.createElement('span');
      const rect = card.getBoundingClientRect();

      const size = Math.max(rect.width, rect.height);
      const x = event.clientX - rect.left - size / 2;
      const y = event.clientY - rect.top - size / 2;

      ripple.style.cssText = `
        position: absolute;
        width: ${size}px;
        height: ${size}px;
        left: ${x}px;
        top: ${y}px;
        background: radial-gradient(circle, rgba(102, 126, 234, 0.3) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
        animation: ripple 0.6s ease-out;
      `;

      card.style.position = 'relative';
      card.appendChild(ripple);

      setTimeout(() => ripple.remove(), 600);
    }
  }

  /* ===========================================================================
     4. STATS DASHBOARD ANIMATION
     Count-up animations and chart visualizations
     =========================================================================== */

  class StatsDashboard {
    constructor() {
      this.init();
    }

    init() {
      // Fetch stats and create dashboard
      this.fetchStats().then(stats => {
        this.createDashboard(stats);
      });
    }

    async fetchStats() {
      try {
        const response = await fetch('/api/stats');
        return await response.json();
      } catch (error) {
        console.error('Failed to fetch stats:', error);
        return null;
      }
    }

    createDashboard(stats) {
      if (!stats) return;

      const dashboardContainer = document.getElementById('stats-dashboard-premium');
      if (!dashboardContainer) return;

      const html = `
        <div class="stats-grid-premium">
          <div class="stat-card-premium">
            <div class="stat-value-premium" data-value="${stats.total_conversations}">0</div>
            <div class="stat-label-premium">Total Conversations</div>
          </div>
          <div class="stat-card-premium">
            <div class="stat-value-premium" data-value="${Math.round(stats.avg_score || 0)}">0</div>
            <div class="stat-label-premium">Average Score</div>
          </div>
          <div class="stat-card-premium">
            <div class="stat-value-premium" data-value="${stats.sources?.length || 0}">0</div>
            <div class="stat-label-premium">AI Sources</div>
          </div>
          <div class="stat-card-premium">
            <div class="stat-value-premium" data-value="${stats.top_tags?.length || 0}">0</div>
            <div class="stat-label-premium">Unique Tags</div>
          </div>
        </div>
      `;

      dashboardContainer.innerHTML = html;

      // Animate the numbers
      setTimeout(() => {
        this.animateStats();
      }, 200);
    }

    animateStats() {
      const statValues = document.querySelectorAll('.stat-value-premium[data-value]');

      statValues.forEach((element, index) => {
        setTimeout(() => {
          const target = parseInt(element.getAttribute('data-value'));
          this.animateNumber(element, 0, target, 1500);
        }, index * 100);
      });
    }

    animateNumber(element, start, end, duration) {
      const startTime = performance.now();

      const updateNumber = (currentTime) => {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        const easeOut = 1 - Math.pow(1 - progress, 3);
        const current = Math.floor(start + (end - start) * easeOut);

        element.textContent = current.toLocaleString();

        if (progress < 1) {
          requestAnimationFrame(updateNumber);
        }
      };

      requestAnimationFrame(updateNumber);
    }
  }

  /* ===========================================================================
     5. TAG CLOUD INTERACTIONS
     Interactive, filterable, beautiful
     =========================================================================== */

  class TagCloud {
    constructor() {
      this.selectedTags = new Set();
      this.init();
    }

    init() {
      document.addEventListener('click', (e) => {
        const tag = e.target.closest('.tag-premium');
        if (tag) {
          this.toggleTag(tag);
        }
      });
    }

    toggleTag(tagElement) {
      const tagName = tagElement.textContent.trim();

      if (this.selectedTags.has(tagName)) {
        this.selectedTags.delete(tagName);
        tagElement.style.background = '';
        tagElement.style.transform = '';
      } else {
        this.selectedTags.add(tagName);
        tagElement.style.background = 'linear-gradient(135deg, var(--color-primary-500), var(--color-secondary-500))';
        tagElement.style.color = 'white';
        tagElement.style.transform = 'scale(1.1)';
      }

      // Filter conversations by selected tags
      this.filterByTags();
    }

    filterByTags() {
      if (this.selectedTags.size === 0) {
        // Show all
        document.querySelectorAll('.conversation-card-premium').forEach(card => {
          card.style.display = '';
        });
        return;
      }

      // Filter cards
      document.querySelectorAll('.conversation-card-premium').forEach(card => {
        const cardTags = Array.from(card.querySelectorAll('.tag-premium'))
          .map(tag => tag.textContent.trim());

        const hasMatch = Array.from(this.selectedTags).some(tag =>
          cardTags.includes(tag)
        );

        card.style.display = hasMatch ? '' : 'none';
      });
    }
  }

  /* ===========================================================================
     6. SCROLL REVEAL ANIMATIONS
     Elements appear as you scroll
     =========================================================================== */

  class ScrollReveal {
    constructor() {
      this.init();
    }

    init() {
      if (!('IntersectionObserver' in window)) return;

      const observer = new IntersectionObserver(
        (entries) => {
          entries.forEach(entry => {
            if (entry.isIntersecting) {
              entry.target.style.opacity = '1';
              entry.target.style.transform = 'translateY(0)';
              observer.unobserve(entry.target);
            }
          });
        },
        { threshold: 0.1, rootMargin: '50px' }
      );

      // Observe elements
      document.querySelectorAll('.stat-card-premium, .conversation-card-premium').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(30px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
      });
    }
  }

  /* ===========================================================================
     INITIALIZATION
     Initialize all spectacular features
     =========================================================================== */

  function init() {
    // Initialize all features
    window.scoreVisualizer = new ScoreVisualizer();
    window.heroSearch = new HeroSearch();
    window.premiumCards = new PremiumCards();
    window.statsDashboard = new StatsDashboard();
    window.tagCloud = new TagCloud();
    window.scrollReveal = new ScrollReveal();

    // Add required CSS animations
    const style = document.createElement('style');
    style.textContent = `
      @keyframes ripple {
        to {
          transform: scale(4);
          opacity: 0;
        }
      }

      @keyframes sparkle-fade {
        0% {
          transform: translate(0, 0) scale(0);
          opacity: 1;
        }
        100% {
          transform: translate(
            ${Math.random() * 40 - 20}px,
            ${Math.random() * 40 - 20}px
          ) scale(1);
          opacity: 0;
        }
      }
    `;
    document.head.appendChild(style);

    console.log('✨ CogRepo Spectacular Interactions initialized');
  }

  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
