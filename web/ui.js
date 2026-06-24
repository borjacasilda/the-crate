/* ════════════════════════════════════════════════════════════════════════
   The Crate — unified UI feedback layer.
   ONE system for transient feedback and confirmations, so every page stops
   reinventing banners/modals (native confirm(), bespoke #overlay modals, inline
   .err/.note/.status, ad-hoc importStatus innerHTML). Include AFTER util.js and
   BEFORE the page script:  <script src="/ui.js"></script>

   Exposes globals:
     toast(message, {type, ms})      ephemeral feedback (ok|bad), auto-dismisses.
     confirmDialog({title, body,     a Promise<bool> confirmation — replaces
       confirmLabel, danger})        confirm() and the hand-rolled modals.

   CRITICAL-BUG FIX (banner reappears after browser Back): feedback lives only in
   the DOM/JS, never in the URL or storage, so it cannot be "restored". And on a
   bfcache restore (pageshow with event.persisted) we tear down any open modal and
   re-derive from the source of truth — pages expose window.refreshFromServer() to
   re-fetch their state; absent that we hard-reload. This makes back/forward never
   show a stale confirmation on ANY page that includes this file.
   ════════════════════════════════════════════════════════════════════════ */
(function () {
  // ── toast host (bottom-right stack) ──
  const host = document.createElement('div');
  host.id = 'ui-toasts';
  document.body.appendChild(host);

  window.toast = function (message, opts) {
    opts = opts || {};
    const t = document.createElement('div');
    t.className = 'ui-toast ' + (opts.type || 'ok');
    t.textContent = message;
    host.appendChild(t);
    requestAnimationFrame(() => t.classList.add('show'));   // trigger the transition
    const ms = opts.ms || 2800;
    setTimeout(() => { t.classList.remove('show'); setTimeout(() => t.remove(), 240); }, ms);
    return t;
  };

  // ── confirmation modal → Promise<bool> ──
  // One implementation, Berghain-styled (theme.css .ui-modal). Resolves true on
  // confirm, false on cancel / click-outside / Escape. Removed from the DOM either
  // way, so it can never linger to be restored by bfcache.
  window.confirmDialog = function (o) {
    o = o || {};
    return new Promise(resolve => {
      const ov = document.createElement('div');
      ov.className = 'ui-overlay open';
      ov.setAttribute('role', 'dialog');
      ov.setAttribute('aria-modal', 'true');
      ov.innerHTML =
        '<div class="ui-modal ' + (o.danger ? 'danger' : '') + '">' +
          '<h2></h2>' +
          (o.body ? '<p></p>' : '') +
          '<div class="ui-row">' +
            '<button class="btn ' + (o.danger ? 'danger' : 'primary') + '" data-yes></button>' +
            '<button class="btn" data-no>Cancel</button>' +
          '</div>' +
        '</div>';
      // textContent (never innerHTML) for caller-supplied strings — no injection.
      ov.querySelector('h2').textContent = o.title || 'Are you sure?';
      if (o.body) ov.querySelector('p').textContent = o.body;
      ov.querySelector('[data-yes]').textContent = o.confirmLabel || 'Confirm';

      let done = false;
      const finish = v => {
        if (done) return; done = true;
        document.removeEventListener('keydown', onKey);
        ov.remove();
        resolve(v);
      };
      const onKey = e => { if (e.key === 'Escape') finish(false); else if (e.key === 'Enter') finish(true); };

      ov.querySelector('[data-yes]').onclick = () => finish(true);
      ov.querySelector('[data-no]').onclick = () => finish(false);
      ov.addEventListener('click', e => { if (e.target === ov) finish(false); });   // click outside = cancel
      document.addEventListener('keydown', onKey);
      document.body.appendChild(ov);
      ov.querySelector('[data-yes]').focus();
    });
  };

  // ── bfcache / back-forward guard (the critical-bug fix, applied to EVERY page) ──
  // A page restored from the back/forward cache keeps its exact DOM — including a
  // banner/modal you already dealt with. We never want that: tear down any open
  // confirmation and re-derive state from the server.
  window.addEventListener('pageshow', e => {
    if (!e.persisted) return;                       // normal load already re-derives
    document.querySelectorAll('.ui-overlay, #overlay').forEach(m => m.remove());
    if (typeof window.refreshFromServer === 'function') window.refreshFromServer();
    else window.location.reload();
  });
})();
