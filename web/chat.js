/* ════════════════════════════════════════════════════════════════════════
   The Crate — shared right-rail AI assistant panel.
   Self-injecting (like player.js): a collapsed "ASK" tab on the right edge that
   opens a chat panel. Streams answers from POST /chat (SSE), passes page context
   (the crate/session you're viewing), and offers a model picker with per-RAM
   suitability warnings. Degrades gracefully when Ollama is down / no model.
   Include once per page:  <script src="/chat.js"></script>
   ════════════════════════════════════════════════════════════════════════ */
(function () {
  const J = (u, o) => fetch(u, o).then(r => r.ok ? r.json() : r.text().then(t => Promise.reject(t)));
  const esc = s => String(s ?? '').replace(/[&<>"]/g,
    c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  // ── markdown → HTML for the data-engine output (the assistant answers in
  // markdown tables / lists / headers — see assistant/scope.py). XSS-safe: the
  // WHOLE source is HTML-escaped FIRST, so the block/inline passes below only ever
  // re-add our own tags around already-inert text. Supports just what the UI emits:
  // GFM pipe tables, # headers, -/* and 1. lists, **bold**, `code`, links. ──
  const mdInline = s => s
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
             '<a href="$2" target="_blank" rel="noopener">$1</a>')
    .replace(/(^|[\s(])(https?:\/\/[^\s<]+)/g,
             '$1<a href="$2" target="_blank" rel="noopener">$2</a>');
  const mdRow = l => l.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map(c => c.trim());
  const mdIsSep = l => /-/.test(l) && /^[\s|:-]+$/.test(l);   // |---|---| separator row

  function mdRender(src) {
    const lines = esc(src).replace(/\r/g, '').split('\n');
    const out = [];
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      // GFM table: a row with pipes immediately followed by a |---| separator.
      if (line.includes('|') && i + 1 < lines.length && mdIsSep(lines[i + 1])) {
        const head = mdRow(line); i += 2;
        const body = [];
        while (i < lines.length && lines[i].includes('|') && lines[i].trim()) { body.push(mdRow(lines[i])); i++; }
        out.push('<div class="md-table"><table><thead><tr>' +
          head.map(c => `<th>${mdInline(c)}</th>`).join('') + '</tr></thead><tbody>' +
          body.map(r => '<tr>' + r.map(c => `<td>${mdInline(c)}</td>`).join('') + '</tr>').join('') +
          '</tbody></table></div>');
        continue;
      }
      const h = line.match(/^\s*#{1,4}\s+(.*)$/);
      if (h) { out.push(`<div class="md-h">${mdInline(h[1])}</div>`); i++; continue; }
      if (/^\s*[-*]\s+/.test(line)) {
        const items = [];
        while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) { items.push(lines[i].replace(/^\s*[-*]\s+/, '')); i++; }
        out.push('<ul>' + items.map(t => `<li>${mdInline(t)}</li>`).join('') + '</ul>');
        continue;
      }
      if (/^\s*\d+[.)]\s+/.test(line)) {
        const items = [];
        while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i])) { items.push(lines[i].replace(/^\s*\d+[.)]\s+/, '')); i++; }
        out.push('<ol>' + items.map(t => `<li>${mdInline(t)}</li>`).join('') + '</ol>');
        continue;
      }
      if (!line.trim()) { i++; continue; }
      // Paragraph: gather consecutive plain lines (a stray pipe line lands here too).
      const para = [];
      while (i < lines.length && lines[i].trim() && !/^\s*#{1,4}\s+/.test(lines[i]) &&
             !/^\s*[-*]\s+/.test(lines[i]) && !/^\s*\d+[.)]\s+/.test(lines[i]) &&
             !(lines[i].includes('|') && i + 1 < lines.length && mdIsSep(lines[i + 1]))) {
        para.push(lines[i]); i++;
      }
      out.push(`<p>${mdInline(para.join('<br>'))}</p>`);
    }
    return out.join('');
  }

  // ── page context so "recommend something like this" just works ──
  function pageContext() {
    const p = new URLSearchParams(location.search), ctx = {};
    if (location.pathname === '/crate' && p.get('id')) ctx.crate_id = p.get('id');
    if (location.pathname === '/session' && p.get('id')) ctx.session_id = p.get('id');
    return ctx;
  }

  // ── DOM ──
  const tab = document.createElement('button');
  tab.id = 'askTab'; tab.textContent = 'ASK'; tab.title = 'Ask the assistant';
  const panel = document.createElement('div');
  panel.id = 'chatPanel'; panel.className = 'closed';
  panel.innerHTML =
    '<div class="chat-head">' +
      '<span class="chat-title"><span class="mark">▌</span>ASSISTANT</span>' +
      '<button class="chat-new" id="chatNew" title="new conversation">NEW</button>' +
      '<a class="chat-kb" href="/knowledge" title="Knowledge base">KB</a>' +
      '<button class="chat-model" id="chatModel" title="model">…</button>' +
      '<button class="chat-x" id="chatClose" title="close">✕</button>' +
    '</div>' +
    '<div class="chat-confirm" id="chatConfirm"></div>' +
    '<div class="chat-banner" id="chatBanner"></div>' +
    '<div class="chat-models" id="chatModels"></div>' +
    '<div class="chat-msgs" id="chatMsgs"></div>' +
    '<form class="chat-input" id="chatForm">' +
      '<input id="chatText" type="text" autocomplete="off" ' +
        'placeholder="recommend something like this · artists like Mulero…">' +
      '<button class="pbtn" type="submit">↵</button>' +
    '</form>';
  document.body.appendChild(tab);
  document.body.appendChild(panel);
  const el = id => document.getElementById(id);

  let STATUS = null, busy = false, controller = null;

  function open() { panel.classList.remove('closed'); document.body.classList.add('chat-open');
                    loadStatus(); el('chatText').focus(); }
  function close() { panel.classList.add('closed'); document.body.classList.remove('chat-open'); }
  tab.onclick = open;
  el('chatClose').onclick = close;

  // ── new conversation ──
  // The assistant is stateless (each /chat is independent — the server keeps no
  // history), so a "new conversation" is purely a client reset: abort any in-flight
  // answer, wipe the transcript, and confirm with a brief banner.
  function flashConfirm(text) {
    const c = el('chatConfirm');
    c.textContent = text;
    c.classList.add('show');
    clearTimeout(c._t);
    c._t = setTimeout(() => c.classList.remove('show'), 2600);
  }
  function newConversation() {
    if (controller) { try { controller.abort(); } catch (e) { /* already done */ } }
    busy = false;
    el('chatMsgs').innerHTML = '';
    flashConfirm('New conversation — previous chat cleared');
    el('chatText').focus();
  }
  el('chatNew').onclick = newConversation;

  // ── status + model picker ──
  async function loadStatus() {
    try {
      STATUS = await J('/assistant/status');
      const active = STATUS.models.find(m => m.tag === STATUS.active_model);
      el('chatModel').textContent = (active ? active.params : STATUS.active_model);
      const installed = STATUS.models.some(m => m.role === 'chat' && m.installed);
      const banner = el('chatBanner');
      if (!STATUS.ollama_up) {
        banner.className = 'chat-banner bad';
        banner.innerHTML = 'Local LLM offline. Start it: <code>brew services start ollama</code>';
      } else if (!installed) {
        banner.className = 'chat-banner';
        banner.innerHTML = 'No model installed yet — pick one below (' + STATUS.ram_gb + ' GB RAM).';
        renderModels(true);
      } else {
        banner.className = 'chat-banner hidden';
      }
    } catch (e) {
      el('chatBanner').className = 'chat-banner bad';
      el('chatBanner').textContent = 'API not reachable.';
    }
  }

  el('chatModel').onclick = () => {
    const box = el('chatModels');
    box.classList.contains('hidden') === false ? box.classList.add('hidden') : renderModels(false);
  };

  function renderModels(forceOpen) {
    const box = el('chatModels');
    box.classList.remove('hidden');
    box.innerHTML = '<div class="cm-head">MODEL · ' + (STATUS ? STATUS.ram_gb : '?') + ' GB RAM</div>' +
      STATUS.models.filter(m => m.role === 'chat').map(m => {
        const cls = m.verdict; const active = m.tag === STATUS.active_model;
        const right = m.installed
          ? `<button class="cm-use ${active ? 'on' : ''}" data-use="${m.tag}">${active ? 'in use' : 'use'}</button>`
          : `<button class="cm-get" data-get="${m.tag}">download</button>`;
        return `<div class="cm-row">` +
          `<div><span class="cm-name">${esc(m.label)}</span> ` +
          `<span class="cm-badge ${cls}">${cls}</span>` +
          (m.warning ? `<div class="cm-warn">${esc(m.warning)}</div>` : '') +
          `<div class="cm-note">${esc(m.note)}</div></div>${right}</div>`;
      }).join('') +
      '<div id="cmProgress" class="cm-progress hidden"></div>';
    box.querySelectorAll('[data-use]').forEach(b => b.onclick = () => useModel(b.dataset.use));
    box.querySelectorAll('[data-get]').forEach(b => b.onclick = () => pullModel(b.dataset.get));
  }

  async function useModel(tag) {
    try { await J('/assistant/model', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                                        body: JSON.stringify({ model: tag }) });
      await loadStatus(); renderModels(false); el('chatModels').classList.add('hidden');
    } catch (e) { el('cmProgress').classList.remove('hidden'); el('cmProgress').textContent = String(e); }
  }

  async function pullModel(tag) {
    const prog = el('cmProgress'); prog.classList.remove('hidden');
    prog.textContent = `pulling ${tag}…`;
    const res = await fetch('/assistant/pull', { method: 'POST',
      headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model: tag }) });
    const reader = res.body.getReader(), dec = new TextDecoder(); let buf = '';
    while (true) {
      const { done, value } = await reader.read(); if (done) break;
      buf += dec.decode(value, { stream: true }); let i;
      while ((i = buf.indexOf('\n\n')) >= 0) {
        const line = buf.slice(0, i); buf = buf.slice(i + 2);
        if (line.startsWith('data: ')) { const e = JSON.parse(line.slice(6));
          prog.textContent = `${tag}: ${e.status}`;
          if (e.status === 'done') { await loadStatus(); renderModels(false); }
        }
      }
    }
  }

  // ── chat ──
  function bubble(role, html) {
    const d = document.createElement('div'); d.className = 'msg ' + role; d.innerHTML = html;
    el('chatMsgs').appendChild(d); el('chatMsgs').scrollTop = el('chatMsgs').scrollHeight;
    return d;
  }

  el('chatForm').onsubmit = async (ev) => {
    ev.preventDefault();
    const text = el('chatText').value.trim();
    if (!text || busy) return;
    el('chatText').value = ''; busy = true;
    bubble('user', esc(text));
    const a = bubble('bot', '<span class="dots">…</span>');
    let acc = '';
    controller = new AbortController();
    try {
      const res = await fetch('/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, context: pageContext() }), signal: controller.signal });
      const reader = res.body.getReader(), dec = new TextDecoder(); let buf = '';
      while (true) {
        const { done, value } = await reader.read(); if (done) break;
        buf += dec.decode(value, { stream: true }); let i;
        while ((i = buf.indexOf('\n\n')) >= 0) {
          const line = buf.slice(0, i); buf = buf.slice(i + 2);
          if (!line.startsWith('data: ')) continue;
          const e = JSON.parse(line.slice(6));
          if (e.delta) { acc += e.delta; a.innerHTML = mdRender(acc); }
          else if (e.error) { a.innerHTML = `<span class="bad">${esc(e.error)}</span>`; loadStatus(); }
          else if (e.done && !acc) a.innerHTML = '<span class="dim">(no answer)</span>';
          el('chatMsgs').scrollTop = el('chatMsgs').scrollHeight;
        }
      }
    } catch (e) {
      if (e && e.name === 'AbortError') { a.remove(); return; }  // reset mid-answer: drop it quietly
      a.innerHTML = `<span class="bad">${esc(String(e))}</span>`;
    } finally { busy = false; controller = null; }
    el('chatText').focus();
  };
})();
