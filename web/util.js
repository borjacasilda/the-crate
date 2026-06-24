/* ════════════════════════════════════════════════════════════════════════
   The Crate — shared front-end helpers.
   The handful of functions every page redefined inline, hoisted to one place
   (same pattern as theme.css / player.js). Include FIRST, before player.js and
   the page script:  <script src="/util.js"></script>
   Exposes globals: j, post, esc, parseLabel.
   ════════════════════════════════════════════════════════════════════════ */

/* fetch → JSON, throwing the response body on a non-2xx. opts is optional, so
   j(url) and j(url, {headers}) both work. */
async function j(url, opts){
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

/* POST JSON (or an empty body) and return the parsed reply. */
function post(url, body){
  return j(url, { method: 'POST', headers: { 'Content-Type': 'application/json' },
                  body: body ? JSON.stringify(body) : null });
}

/* HTML-escape a value for safe innerHTML interpolation. */
const esc = s => String(s ?? '').replace(/[&<>"]/g,
  c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

/* "Artist - Track [EP].ext" → {artist, track, ep}. The EP goes in brackets by
   convention; without " - " the whole name lands in the Track column. */
function parseLabel(filename){
  let name = String(filename || '').replace(/\.(wav|mp3|flac|aiff?)$/i, '').trim();
  let ep = '';
  const m = name.match(/\[(.+?)\]/);
  if (m){ ep = m[1]; name = name.replace(m[0], '').replace(/\s+/g, ' ').trim(); }
  const i = name.indexOf(' - ');
  if (i < 0) return { artist: '—', track: name, ep };
  return { artist: name.slice(0, i), track: name.slice(i + 3), ep };
}

/* Original file extension incl. the dot (".mp3"), or "" if none. */
function fileExt(filename){
  const m = String(filename || '').match(/\.(wav|mp3|flac|aiff?)$/i);
  return m ? m[0] : '';
}

/* Rebuild the "Artist - Track [EP].ext" label after editing ONE field. */
function rebuildFilename(filename, field, value){
  const p = parseLabel(filename);
  p[field] = String(value).trim();
  const artist = (p.artist && p.artist !== '—') ? p.artist : '';
  let base = artist ? `${artist} - ${p.track}` : p.track;
  if (p.ep) base += ` [${p.ep}]`;
  return base + fileExt(filename);
}

/* Make a table cell edit-in-place on click (like renaming a file).
   field ∈ 'artist'|'track'|'ep'. onSave(track_id, newFilename) does the per-track
   PATCH + list reload (for 'track'/'ep'). For 'artist', if onRenameArtist is given
   the edit renames the artist ENTITY across the whole library (confirm + global
   PATCH) instead of just this one label. A re-render mid-edit is avoided by the
   page (see isEditing()). */
function bindEdit(td, track, field, onSave, onRenameArtist){
  td.classList.add('editable');
  td.title = 'click to edit';
  td.addEventListener('dblclick', e => e.stopPropagation());
  td.addEventListener('click', e => {
    e.stopPropagation();                       // never trigger row-play
    if (td.querySelector('input')) return;     // already editing
    // artist/track/ep come from the filename; label is a separate DB field.
    const cur = (field === 'label') ? (track.label || '—') : parseLabel(track.filename)[field];
    const shown = cur === '—' ? '' : cur;
    const input = document.createElement('input');
    input.type = 'text'; input.value = shown; input.className = 'cell-edit';
    td.textContent = ''; td.appendChild(input);
    input.focus(); input.select();
    let done = false;
    const finish = async (save) => {
      if (done) return; done = true;
      const val = input.value.trim();
      if (save && val && val !== shown){
        try {
          if (field === 'artist' && onRenameArtist){
            // Renaming the artist is GLOBAL: it changes the entity on every track,
            // not just this row. onRenameArtist returns false if the user cancels
            // the confirm, in which case we revert the cell.
            const ok = await onRenameArtist(shown, val);
            if (!ok) td.textContent = cur;
          } else if (field === 'label'){
            // Label is a Discogs field (not in the filename) — save the raw value.
            await onSave(track.track_id, val);
          } else {
            await onSave(track.track_id, rebuildFilename(track.filename, field, val));
          }
        } catch(e){ td.textContent = cur; }    // revert on failure
      } else {
        td.textContent = cur;                  // unchanged / cancelled
      }
    };
    input.addEventListener('keydown', ev => {
      if (ev.key === 'Enter'){ ev.preventDefault(); finish(true); }
      else if (ev.key === 'Escape'){ ev.preventDefault(); finish(false); }
    });
    input.addEventListener('blur', () => finish(true));
  });
}

/* Global artist rename — renames the artist ENTITY across ALL its tracks (and
   similar_artists), not just the clicked label. Confirms first; returns true when
   applied (caller reloads its lists), false when the user cancels. */
async function renameArtist(oldName, newName){
  if (!oldName || oldName === '—') return false;     // no real artist in this row
  // confirmDialog is provided by ui.js (loaded on every page right after util.js).
  const ok = await confirmDialog({
    title: 'Rename artist?',
    body: `Rename "${oldName}" to "${newName}" across the whole library — every linked track is updated.`,
    confirmLabel: 'Rename' });
  if (!ok) return false;
  await post('/artists/rename', { old: oldName, new: newName });
  return true;
}

/* True while any inline editor is focused — pages check this to skip a list
   re-render that would clobber the input the user is typing in. */
function isEditing(){
  return document.activeElement &&
         document.activeElement.classList.contains('cell-edit');
}

/* SAFE rename for an OBJECT name (a session, a crate…). Unlike bindEdit — which
   commits on blur and is fine for low-stakes track metadata — this never saves by
   accident: an explicit ✎ enters edit, and it commits ONLY on Enter or the ✓
   button; blur and Escape CANCEL. `el` is the element showing the name; onSave(value)
   does the persist (and should reload). Idempotent per element (safe to call on every
   re-render): it replaces its own previous pencil. */
function editNameSafe(el, current, onSave){
  if (el._namePencil) el._namePencil.remove();     // drop a stale pencil from a prior render
  const pencil = document.createElement('button');
  pencil.type = 'button'; pencil.className = 'name-edit'; pencil.textContent = '✎';
  pencil.title = 'rename';
  pencil.onclick = () => {
    if (el.querySelector('input')) return;          // already editing
    const shown = el.textContent;
    const input = document.createElement('input');
    input.type = 'text'; input.className = 'name-edit-input'; input.value = current;
    const ok = document.createElement('button');
    ok.type = 'button'; ok.className = 'name-ok'; ok.textContent = '✓'; ok.title = 'save (Enter)';
    el.textContent = ''; el.append(input, ok);
    pencil.style.display = 'none';
    input.focus(); input.select();
    let settled = false;
    const close = () => { pencil.style.display = ''; };
    const commit = async (save) => {
      if (settled) return;
      const val = input.value.trim();
      if (!save || !val || val === current){ settled = true; el.textContent = shown; close(); return; }
      settled = true;
      try { await onSave(val); }                    // caller reloads on success
      catch(e){ el.textContent = shown; close(); toast('Rename failed: ' + e.message, { type: 'bad' }); }
    };
    ok.onclick = () => commit(true);
    input.addEventListener('keydown', ev => {
      if (ev.key === 'Enter'){ ev.preventDefault(); commit(true); }
      else if (ev.key === 'Escape'){ ev.preventDefault(); commit(false); }
    });
    // blur CANCELS (never an accidental save). The timeout lets a ✓ click land first.
    input.addEventListener('blur', () => setTimeout(() => commit(false), 120));
  };
  el.after(pencil);
  el._namePencil = pencil;
}
