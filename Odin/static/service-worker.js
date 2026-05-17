// Service Worker - Offline Destek
const CACHE_NAME = 'odin-weather-v1';
const urlsToCache = [
  '/',
  '/static/styles.css',
  '/static/app.js',
  '/offline.html'
];

// Cache'i kur
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      return cache.addAll(urlsToCache);
    })
  );
});

// Network-first strategy
self.addEventListener('fetch', event => {
  event.respondWith(
    fetch(event.request)
      .then(response => {
        const clone = response.clone();
        caches.open(CACHE_NAME).then(cache => {
          cache.put(event.request, clone);
        });
        return response;
      })
      .catch(() => {
        return caches.match(event.request)
          .then(response => response || caches.match('/offline.html'));
      })
  );
});

// Push Notification
self.addEventListener('push', event => {
  const data = event.data.json();
  const options = {
    body: data.body,
    icon: '/static/icon-192.png',
    badge: '/static/badge-72.png',
    tag: 'weather-alert',
    requireInteraction: true,
    actions: [
      { action: 'open', title: 'Aç' },
      { action: 'close', title: 'Kapat' }
    ]
  };
  event.waitUntil(
    self.registration.showNotification(data.title, options)
  );
});

self.addEventListener('notificationclick', event => {
  event.notification.close();
  if (event.action === 'open' || !event.action) {
    clients.openWindow('/');
  }
});
