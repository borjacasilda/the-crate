/* ════════════════════════════════════════════════════════════════════════
   The Crate — shared "KEY" (Camelot) column.
   Adds DJ-style harmonic-key handling to every track list, with no per-page
   wiring beyond a <th class="key"> header and a <td class="key"> cell per row.
   Include AFTER player.js (so rows already carry their key cell):
       <script src="/keycol.js"></script>

   Click the KEY header → a dropdown with exactly three actions:
       • Hide keys      — blank the column (toggles to "Show keys"), like BPM.
       • Sort high → low — highest Camelot key first (12B → 1A).
       • Sort low → high — the reverse (1A → 12B).
   Hide is pure CSS (body.hide-key). Sort reorders the tbody rows and re-applies
   itself after any page re-render (a guarded MutationObserver), so it survives
   the frequent list refreshes. Both preferences persist in localStorage and are
   shared across every page and mode — set it once, it holds everywhere.
   ════════════════════════════════════════════════════════════════════════ */
(function () {
  const HIDE = 'thecrate_hide_key';
  const SORT = 'thecrate_sort_key';                 // 'none' | 'asc' | 'desc'
  const getHide = () => localStorage.getItem(HIDE) === '1';
  const getSort = () => localStorage.getItem(SORT) || 'none';

  /* Camelot "9A" / "12B" → a total-order rank (1A<1B<2A<…<12B); the minor (A)
     ring sorts before the major (B) ring of the same number. Unknown/missing
     keys get Infinity so they always sink to the bottom, either direction. */
  function rank(c) {
    const m = /^(\d{1,2})([AB])$/.exec(String(c || '').trim().toUpperCase());
    return m ? parseInt(m[1], 10) * 2 + (m[2] === 'B' ? 1 : 0) : Infinity;
  }
  const cellKey = tr => {
    const td = tr.querySelector('td.key');
    return td ? td.textContent : '';
  };

  /* ── header label reflects the current state (arrow = active sort) ── */
  function paint() {
    const hide = getHide(), s = getSort();
    const tag = hide ? 'KEY ⊘'
              : 'KEY' + (s === 'asc' ? ' ↑' : s === 'desc' ? ' ↓' : ' ▾');
    document.querySelectorAll('th.key').forEach(th => {
      th.textContent = tag;
      th.title = 'harmonic key (Camelot) — click for options';
    });
    document.body.classList.toggle('hide-key', hide);
  }

  /* ── sort: reorder every tbody that carries a key column ── */
  let obs;
  const connect = () => document.querySelectorAll('table').forEach(
    t => obs.observe(t, { childList: true, subtree: true }));

  function sortAll() {
    if (getSort() === 'none') return;
    const asc = getSort() === 'asc';
    obs.disconnect();                                // ignore our own row moves
    document.querySelectorAll('table').forEach(t => {
      const body = t.tBodies[0];
      if (!body || !body.querySelector('td.key')) return;
      [...body.rows]
        .sort((x, y) => {
          const rx = rank(cellKey(x)), ry = rank(cellKey(y));
          if (rx === ry) return 0;
          if (rx === Infinity) return 1;             // unknown keys always last…
          if (ry === Infinity) return -1;            // …in BOTH directions
          return asc ? rx - ry : ry - rx;
        })
        .forEach(r => body.appendChild(r));
    });
    connect();
  }

  const setHide = v => { localStorage.setItem(HIDE, v ? '1' : '0'); paint(); };
  const setSort = v => { localStorage.setItem(SORT, v); paint(); sortAll(); };

  /* Pages rebuild their tbodies on most actions; re-apply the sort afterwards.
     The observer is disconnected during our own reorder (see sortAll), so this
     never recurses. Watching the tables only (not the whole body) keeps live
     mode's VU/clock updates from triggering spurious re-sorts. rAF coalesces a
     burst of row inserts into one sort. */
  let raf = 0;
  obs = new MutationObserver(() => {
    if (getSort() === 'none' || raf) return;
    raf = requestAnimationFrame(() => { raf = 0; sortAll(); });
  });
  connect();

  /* ── header dropdown (floating, so the table never clips it) ── */
  const menu = document.createElement('div');
  menu.className = 'keymenu hidden';
  document.body.appendChild(menu);

  function openMenu(th) {
    const hide = getHide(), s = getSort();
    menu.innerHTML = '';
    const add = (label, on, fn) => {
      const b = document.createElement('button');
      b.textContent = label;
      if (on) b.classList.add('on');
      b.onclick = e => { e.stopPropagation(); fn(); closeMenu(); };
      menu.appendChild(b);
    };
    // Exactly three actions. "Hide" is a toggle (relabels to "Show" once hidden)
    // so the column can always be brought back. "Sort high → low" puts the
    // highest Camelot key (12B) on top; "Sort low → high" is the reverse.
    add(hide ? 'Show keys' : 'Hide keys', false, () => setHide(!hide));
    add('Sort  high → low', s === 'desc', () => setSort('desc'));
    add('Sort  low → high', s === 'asc',  () => setSort('asc'));
    // place under the header, clamped so it never spills off the viewport edge
    menu.classList.remove('hidden');
    const r = th.getBoundingClientRect();
    const maxLeft = window.scrollX + document.documentElement.clientWidth
                    - menu.offsetWidth - 8;
    menu.style.left = Math.max(8, Math.min(window.scrollX + r.left, maxLeft)) + 'px';
    menu.style.top  = (window.scrollY + r.bottom + 4) + 'px';
  }
  const closeMenu = () => menu.classList.add('hidden');

  document.addEventListener('click', e => {
    const th = e.target.closest && e.target.closest('th.key');
    if (th) { e.stopPropagation(); openMenu(th); return; }
    if (!menu.contains(e.target)) closeMenu();
  });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeMenu(); });

  /* initial state (headers are static; rows arrive async and the observer sorts) */
  paint();
  sortAll();
})();
