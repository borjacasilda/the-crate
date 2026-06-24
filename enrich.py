"""
enrich.py — Discogs enrichment orchestration (Phase 3b).

The bridge between the pure Discogs client (discogs.py) and The Crate's database:
it parses a track's "Artist - Title" from the filename, asks discogs.py for the
best release match, and — on a confident match — writes the metadata, downloads
the cover image, links the label, and refreshes that label's sonic centroid (so
"labels that sound like X" works immediately, the same EffNet ANN as artists).

The auto + confirm-doubtful flow lives here:
  • matched   → applied automatically.
  • doubtful  → stored with a candidate shortlist for the user to confirm.
  • unmatched → recorded so we do not re-query it every run.
`confirm_match` applies a user's chosen release from the doubtful queue.
"""
import logging
import re
from pathlib import Path

import analyze
import config
import database
import discogs

logger = logging.getLogger("thecrate.enrich")


def _parse(filename: str) -> tuple:
    """'Artist - Track [EP].ext' → (artist, title). Mirrors the UI's parseLabel:
    the [bracket] is the EP/label tag and is dropped for matching."""
    name = re.sub(r"\.(wav|mp3|flac|aiff?)$", "", filename or "", flags=re.I)
    m = re.search(r"\[(.+?)\]", name)
    if m:
        name = name.replace(m.group(0), "").strip()
    i = name.find(" - ")
    if i < 0:
        return ("", name.strip())
    return (name[:i].strip(), name[i + 3:].strip())


def _label_discogs_id(release: dict):
    labels = release.get("labels") or []
    return labels[0].get("id") if labels else None


def _download_cover(track_id: str, release: dict) -> tuple:
    """Download the release's primary image to covers/<track_id>.jpg.
    Returns (cover_url, cover_path|None)."""
    images = release.get("images") or []
    cover_url = images[0] if images else None
    if not cover_url:
        return (None, None)
    dest = config.COVERS_DIR / f"{track_id}.jpg"
    saved = discogs.download_cover(cover_url, dest)
    return (cover_url, str(saved) if saved else None)


def _apply_release(track_id: str, release: dict, status: str,
                   confidence: float, candidates: list = None) -> dict:
    """Persist a confident/confirmed release: metadata + cover + label + centroid."""
    cover_url, cover_path = _download_cover(track_id, release)
    database.upsert_track_discogs(
        track_id,
        release_id=release.get("release_id"), master_id=release.get("master_id"),
        label=release.get("label"), catno=release.get("catno"),
        year=release.get("year"), country=release.get("country"),
        genres=release.get("genres"), styles=release.get("styles"),
        cover_url=cover_url, cover_path=cover_path,
        status=status, confidence=confidence, candidates=candidates or [])
    # label as a first-class entity + refresh its centroid
    if release.get("label"):
        for lid in database.relink_track_label(track_id, release["label"],
                                               discogs_id=_label_discogs_id(release)):
            try:
                analyze.persist_label_embedding(lid)
            except Exception as e:
                logger.warning("label centroid refresh failed for %s: %s", lid, e)
    return {"track_id": track_id, "status": status, "confidence": confidence,
            "label": release.get("label"), "year": release.get("year"),
            "styles": release.get("styles"), "cover": bool(cover_path)}


def enrich_track(track_id: str) -> dict:
    """Match one track to Discogs and apply the outcome. Never invents a match."""
    t = database.get_track(track_id)
    if not t:
        return {"track_id": track_id, "status": "error", "error": "no such track"}
    artist, title = _parse(t["filename"])
    res = discogs.best_match(artist, title)
    st = res.get("status")
    if st == "matched":
        return _apply_release(track_id, res["release"], "matched",
                              res.get("confidence"), res.get("candidates"))
    # doubtful / unmatched / unconfigured — record so we do not re-query blindly
    status = "doubtful" if st == "doubtful" else "unmatched"
    database.upsert_track_discogs(track_id, status=status,
                                  confidence=res.get("confidence"),
                                  candidates=res.get("candidates") or [])
    return {"track_id": track_id, "status": status,
            "confidence": res.get("confidence"),
            "candidates": res.get("candidates") or [],
            "query": {"artist": artist, "title": title}}


def confirm_match(track_id: str, release_id) -> dict:
    """Apply a release the user picked from the doubtful queue (status=confirmed)."""
    release = discogs.get_release(release_id)
    return _apply_release(track_id, release, "confirmed", 1.0)


def refresh_cover(track_id: str) -> dict:
    """Best-effort cover-only re-search, triggered when a track's metadata is edited
    and it still has NO cover. Unlike enrich_track, it ONLY downloads/sets the cover
    (artist/title may now be corrected) and leaves label/year/styles untouched, so a
    user's manual edits are never clobbered. Confident matches only — we would rather
    show no cover than the wrong one. Returns {"cover": bool, ...}; never invents."""
    if not discogs.is_configured():
        return {"track_id": track_id, "cover": False, "skipped": "discogs-unconfigured"}
    t = database.get_track(track_id)
    if not t:
        return {"track_id": track_id, "cover": False, "error": "no such track"}
    existing = database.get_track_discogs(track_id) or {}
    cp = existing.get("cover_path")
    if cp and Path(cp).exists():
        return {"track_id": track_id, "cover": True, "skipped": "already-has-cover"}
    artist, title = _parse(t["filename"])
    res = discogs.best_match(artist, title)
    if res.get("status") != "matched":           # only a confident match earns a cover
        return {"track_id": track_id, "cover": False, "status": res.get("status")}
    cover_url, cover_path = _download_cover(track_id, res["release"])
    if cover_path:
        database.set_track_cover(track_id, cover_url, cover_path)
    return {"track_id": track_id, "cover": bool(cover_path)}


def iter_enrich(limit: int = None):
    """Generator: enrich every pending track, yielding a progress dict each.

    Pending = analysed tracks with no Discogs row (or still unmatched). Yields a
    final {'done': True, 'counts': {...}} so a streamer can show a summary.
    """
    pending = database.tracks_pending_discogs()
    if limit:
        pending = pending[:limit]
    counts = {"matched": 0, "doubtful": 0, "unmatched": 0, "error": 0}
    for i, t in enumerate(pending, 1):
        try:
            r = enrich_track(str(t["track_id"]))
        except Exception as e:
            logger.warning("enrich failed for %s: %s", t["filename"], e)
            r = {"track_id": str(t["track_id"]), "status": "error", "error": str(e)}
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        yield {"i": i, "n": len(pending), "filename": t["filename"], **r}
    yield {"done": True, "counts": counts, "total": len(pending)}


def enrich_all(limit: int = None) -> dict:
    """Run the whole pending queue and return the summary (non-streaming)."""
    last = {}
    for ev in iter_enrich(limit):
        last = ev
    return last
