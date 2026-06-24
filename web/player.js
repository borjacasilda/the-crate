/* ════════════════════════════════════════════════════════════════════════
   The Crate — shared preview player (deejay.de-style, Berghain minimal).
   One <audio>, one fixed bottom bar, the SAME on every page that lists
   tracks. Layout:  ⏮  ▶/❚❚  ⏭ │ ▁▃▆█▅▂ waveform (played=red) │ time │ name │ ✕
   The waveform IS the progress/seek bar — click anywhere on it to seek. No
   ±10 buttons. Prev/Next walk the track rows in DOM order. Styles: theme.css.
   Include after the body content:  <script src="/player.js"></script>

   API:
     VinylPlayer.bind(tr, trackId, label)  Make a row previewable: inserts a
         leading ▶ cell (empty when trackId is null — history rows whose track
         was deleted), tags the row, and wires a row click to play/pause.
     VinylPlayer.refresh()  Restore the playing-row highlight after a re-render.
     VinylPlayer.stop()     Stop and hide the bar.
   ════════════════════════════════════════════════════════════════════════ */
(function () {
  const bar = document.createElement('div');
  bar.id = 'player';
  bar.className = 'hidden';
  bar.innerHTML =
    '<button class="pbtn" id="plPrev" title="anterior">⏮</button>' +
    '<button class="pbtn" id="plToggle" title="play/pausa">▶</button>' +
    '<button class="pbtn" id="plNext" title="siguiente">⏭</button>' +
    '<canvas id="plWave" title="click para buscar"></canvas>' +
    '<span class="ptime" id="plTime">0:00 / 0:00</span>' +
    '<span class="pname" id="plName"></span>' +
    '<button class="pbtn" id="plClose" title="cerrar">✕</button>';
  document.body.appendChild(bar);

  const el = id => document.getElementById(id);
  const canvas = el('plWave');
  const ctx = canvas.getContext('2d');
  const audio = new Audio();
  let playingId = null;
  let peaks = [];                       // current track's waveform (0–1)
  const COL_DIM = '#3a3a3a', COL_HOT = '#ff2222';
  const fmt = s => Math.floor(s / 60) + ':' + String(Math.floor(s % 60)).padStart(2, '0');

  /* ── waveform rendering ── */
  function sizeCanvas() {
    const r = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(r.width * dpr));
    canvas.height = Math.max(1, Math.floor(r.height * dpr));
  }
  function drawWave() {
    const w = canvas.width, h = canvas.height, n = peaks.length;
    ctx.clearRect(0, 0, w, h);
    if (!n) return;
    const progress = audio.duration ? audio.currentTime / audio.duration : 0;
    const bw = w / n, mid = h / 2, gap = bw > 3 ? 1 : 0;
    for (let i = 0; i < n; i++) {
      const bh = Math.max(1, peaks[i] * h * 0.92);
      ctx.fillStyle = (i / n) <= progress ? COL_HOT : COL_DIM;
      ctx.fillRect(i * bw, mid - bh / 2, Math.max(1, bw - gap), bh);
    }
  }
  async function loadWave(trackId) {
    peaks = [];
    drawWave();
    try {
      const r = await fetch('/tracks/' + trackId + '/waveform');
      if (!r.ok) return;
      peaks = (await r.json()).peaks || [];
      sizeCanvas();
      drawWave();
    } catch (e) { /* leave it flat */ }
  }

  /* ── playback ── */
  function rows() {                    // unique track rows in DOM order
    const seen = new Set(), out = [];
    document.querySelectorAll('tr[data-id]').forEach(tr => {
      if (!seen.has(tr.dataset.id)) { seen.add(tr.dataset.id); out.push(tr); }
    });
    return out;
  }
  function play(trackId, label) {
    playingId = trackId;
    audio.src = '/tracks/' + trackId + '/audio';
    audio.play().catch(() => {});
    el('plName').textContent = label || '';
    bar.classList.remove('hidden');
    document.body.classList.add('playing');
    loadWave(trackId);
    refresh();
  }
  function toggle(trackId, label) {
    if (playingId === trackId) {        // same row: play/pause flip
      audio.paused ? audio.play() : audio.pause();
      return;
    }
    play(trackId, label);
  }
  function step(delta) {
    const list = rows();
    const idx = list.findIndex(tr => tr.dataset.id === playingId);
    if (idx < 0) return;
    const target = list[idx + delta];
    if (target) play(target.dataset.id, target.dataset.label || '');
  }
  function stop() {
    audio.pause();
    audio.removeAttribute('src');
    playingId = null;
    bar.classList.add('hidden');
    document.body.classList.remove('playing');
    refresh();
  }
  function refresh() {
    // The playing row shows ❚❚ (and only while actually playing); every other
    // row's cell resets to ▶ — a one-glance "you are here" in the list.
    const live = playingId && !audio.paused;
    document.querySelectorAll('tr[data-id]').forEach(tr => {
      const here = tr.dataset.id === playingId;
      tr.classList.toggle('playing', here);
      const cell = tr.querySelector('td.play');
      if (cell) cell.textContent = (here && live) ? '❚❚' : '▶';
    });
  }

  function bind(tr, trackId, label, opts) {
    opts = opts || {};
    const td = document.createElement('td');
    td.className = 'play';
    tr.insertBefore(td, tr.firstChild);
    // Discogs cover thumbnail, sitting between ▶ and the artist (opt-in per page,
    // so only tables whose header has a cover column get the extra cell).
    if (opts.cover && trackId) {
      const c = document.createElement('td');
      c.className = 'cover';
      const img = document.createElement('img');
      img.loading = 'lazy'; img.alt = '';
      img.src = `/tracks/${trackId}/cover`;
      img.onerror = () => { c.classList.add('nocover'); img.remove(); };
      c.appendChild(img);
      tr.insertBefore(c, td.nextSibling);
    }
    if (!trackId) return;               // history row without a track
    td.textContent = '▶';
    td.title = 'escuchar';
    tr.dataset.id = tr.dataset.id || trackId;
    tr.dataset.label = label || '';
    // Listen on the PLAY CELL only, not the whole row. This lets other cells (artist,
    // track, ep, label) have their own click handlers without the play handler interfering.
    td.addEventListener('click', () => toggle(trackId, label));
  }

  /* ── audio events ── */
  audio.addEventListener('timeupdate', () => {
    if (audio.duration) {
      el('plTime').textContent = fmt(audio.currentTime) + ' / ' + fmt(audio.duration);
      drawWave();
    }
  });
  audio.addEventListener('play',  () => { el('plToggle').textContent = '❚❚'; refresh(); });
  audio.addEventListener('pause', () => { el('plToggle').textContent = '▶'; refresh(); });
  audio.addEventListener('ended', () => step(1));   // roll into the next row

  /* ── controls ── */
  el('plToggle').onclick = () => { if (audio.src) audio.paused ? audio.play() : audio.pause(); };
  el('plPrev').onclick = () => step(-1);
  el('plNext').onclick = () => step(1);
  el('plClose').onclick = stop;
  canvas.addEventListener('click', e => {
    if (!audio.duration) return;
    const r = canvas.getBoundingClientRect();
    audio.currentTime = audio.duration * (e.clientX - r.left) / r.width;
    drawWave();
  });
  window.addEventListener('resize', () => { sizeCanvas(); drawWave(); });

  window.VinylPlayer = { bind, toggle, refresh, stop };
})();
