/**
 * CogRepo Service Worker
 *
 * Provides:
 * - Offline support for the app shell
 * - Caching of API responses
 * - Background sync for searches
 */

const CACHE_NAME = 'cogrepo-v1';
const STATIC_CACHE = 'cogrepo-static-v1';
const API_CACHE = 'cogrepo-api-v1';

// Static assets to cache immediately
const STATIC_ASSETS = [
  '/',
  '/index-v2.html',
  '/static/css/design-system.css',
  '/static/css/components.css',
  '/static/js/api.js',
  '/static/js/ui.js',
  '/static/js/app.js'
];

// API routes to cache
const API_PATTERNS = [
  /\/api\/conversations/,
  /\/api\/stats/,
  /\/api\/tags/,
  /\/api\/sources/
];

// =============================================================================
// Install Event - Cache Static Assets
// =============================================================================

self.addEventListener('install', (event) => {
  console.log('[SW] Installing...');

  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then((cache) => {
        console.log('[SW] Caching static assets');
        return cache.addAll(STATIC_ASSETS.map(url => {
          return new Request(url, { cache: 'reload' });
        })).catch(err => {
          console.warn('[SW] Some assets failed to cache:', err);
        });
      })
      .then(() => {
        console.log('[SW] Installed successfully');
        return self.skipWaiting();
      })
  );
});

// =============================================================================
// Activate Event - Clean Old Caches
// =============================================================================

self.addEventListener('activate', (event) => {
  console.log('[SW] Activating...');

  event.waitUntil(
    caches.keys()
      .then((cacheNames) => {
        return Promise.all(
          cacheNames
            .filter((name) => {
              // Delete old versions of our caches
              return name.startsWith('cogrepo-') &&
                     name !== STATIC_CACHE &&
                     name !== API_CACHE;
            })
            .map((name) => {
              console.log('[SW] Deleting old cache:', name);
              return caches.delete(name);
            })
        );
      })
      .then(() => {
        console.log('[SW] Activated successfully');
        return self.clients.claim();
      })
  );
});

// =============================================================================
// Fetch Event - Handle Requests
// =============================================================================

self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Skip non-GET requests
  if (request.method !== 'GET') {
    return;
  }

  // Skip chrome-extension and other non-http(s) requests
  if (!url.protocol.startsWith('http')) {
    return;
  }

  // Handle API requests
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(handleAPIRequest(request));
    return;
  }

  // Handle static assets with cache-first strategy
  event.respondWith(handleStaticRequest(request));
});

// =============================================================================
// Request Handlers
// =============================================================================

/**
 * Handle static asset requests (Cache First)
 */
async function handleStaticRequest(request) {
  try {
    // Try cache first
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      // Refresh cache in background
      refreshCache(request);
      return cachedResponse;
    }

    // Fetch from network
    const networkResponse = await fetch(request);

    // Cache successful responses
    if (networkResponse.ok) {
      const cache = await caches.open(STATIC_CACHE);
      cache.put(request, networkResponse.clone());
    }

    return networkResponse;
  } catch (error) {
    console.error('[SW] Static fetch failed:', error);

    // Return offline page if available
    const offlineResponse = await caches.match('/index-v2.html');
    if (offlineResponse) {
      return offlineResponse;
    }

    return new Response('Offline', {
      status: 503,
      statusText: 'Service Unavailable'
    });
  }
}

/**
 * Handle API requests (Network First with Cache Fallback)
 */
async function handleAPIRequest(request) {
  const url = new URL(request.url);

  // Determine if this request should be cached
  const shouldCache = API_PATTERNS.some(pattern => pattern.test(url.pathname));

  try {
    // Try network first
    const networkResponse = await fetch(request, {
      // Set a reasonable timeout
      signal: AbortSignal.timeout(10000)
    });

    // Cache successful responses
    if (networkResponse.ok && shouldCache) {
      const cache = await caches.open(API_CACHE);
      cache.put(request, networkResponse.clone());
    }

    return networkResponse;
  } catch (error) {
    console.warn('[SW] API fetch failed, trying cache:', error.message);

    // Try cache fallback
    if (shouldCache) {
      const cachedResponse = await caches.match(request);
      if (cachedResponse) {
        // Add header to indicate cached response
        const headers = new Headers(cachedResponse.headers);
        headers.set('X-From-Cache', 'true');

        return new Response(cachedResponse.body, {
          status: cachedResponse.status,
          statusText: cachedResponse.statusText,
          headers
        });
      }
    }

    // Return error response
    return new Response(JSON.stringify({
      error: 'Offline',
      message: 'Unable to connect to server. Please check your connection.'
    }), {
      status: 503,
      statusText: 'Service Unavailable',
      headers: { 'Content-Type': 'application/json' }
    });
  }
}

/**
 * Refresh cache in background
 */
async function refreshCache(request) {
  try {
    const networkResponse = await fetch(request);
    if (networkResponse.ok) {
      const cache = await caches.open(STATIC_CACHE);
      cache.put(request, networkResponse);
    }
  } catch (error) {
    // Silently fail - this is a background refresh
  }
}

// =============================================================================
// Message Handlers
// =============================================================================

self.addEventListener('message', (event) => {
  const { type, payload } = event.data || {};

  switch (type) {
    case 'SKIP_WAITING':
      self.skipWaiting();
      break;

    case 'CLEAR_CACHE':
      clearCache(payload?.cacheName);
      break;

    case 'CACHE_CONVERSATION':
      cacheConversation(payload);
      break;

    default:
      console.log('[SW] Unknown message type:', type);
  }
});

/**
 * Clear specific or all caches
 */
async function clearCache(cacheName) {
  if (cacheName) {
    await caches.delete(cacheName);
  } else {
    const cacheNames = await caches.keys();
    await Promise.all(
      cacheNames
        .filter(name => name.startsWith('cogrepo-'))
        .map(name => caches.delete(name))
    );
  }

  // Notify clients
  const clients = await self.clients.matchAll();
  clients.forEach(client => {
    client.postMessage({ type: 'CACHE_CLEARED' });
  });
}

/**
 * Cache a specific conversation for offline viewing
 */
async function cacheConversation(conversation) {
  if (!conversation?.convo_id) return;

  const cache = await caches.open(API_CACHE);
  const url = `/api/conversation/${conversation.convo_id}`;

  const response = new Response(JSON.stringify(conversation), {
    headers: { 'Content-Type': 'application/json' }
  });

  await cache.put(url, response);
}

// =============================================================================
// Background Sync (if supported)
// =============================================================================

self.addEventListener('sync', (event) => {
  if (event.tag === 'sync-searches') {
    event.waitUntil(syncSearches());
  }
});

async function syncSearches() {
  // Future: Sync saved searches when back online
  console.log('[SW] Syncing searches...');
}

// =============================================================================
// Push Notifications (if needed in future)
// =============================================================================

self.addEventListener('push', (event) => {
  if (!event.data) return;

  const data = event.data.json();

  event.waitUntil(
    self.registration.showNotification(data.title || 'CogRepo', {
      body: data.body,
      icon: '/static/icons/icon-192.png',
      badge: '/static/icons/badge-72.png',
      data: data.data
    })
  );
});

self.addEventListener('notificationclick', (event) => {
  event.notification.close();

  event.waitUntil(
    self.clients.matchAll({ type: 'window' })
      .then((clients) => {
        // Focus existing window or open new one
        if (clients.length > 0) {
          return clients[0].focus();
        }
        return self.clients.openWindow('/');
      })
  );
});
