"""
discogs.py — isolated client for the Discogs API (Phase 3b enrichment).

Everything that talks to Discogs lives here and NOWHERE else, so the rest of The
Crate never imports `httpx` for this or knows the wire format. The goal is to
squeeze the maximum useful metadata out of a release — label, catalogue number,
year, country, genres/styles, master, and especially the COVER IMAGE — and to
match a crate track to the right release with a confidence score so the caller
can auto-accept the strong matches and queue the doubtful ones for the user.

Auth is a Discogs personal access token (`DISCOGS_ACCESS_TOKEN` in .env). The API
requires a descriptive User-Agent and rate-limits to ~60 req/min authenticated;
both are handled here. When the token is absent every call degrades to "not
configured" instead of raising, so the app runs fine without enrichment.
"""
import logging
import os
import re
import threading
import time
from pathlib import Path

import httpx

logger = logging.getLogger("thecrate.discogs")

BASE = "https://api.discogs.com"
USER_AGENT = "TheCrate/0.1 +https://github.com/thecrate"


def _token() -> str:
    """Personal access token (preferred auth). Read lazily so import order never
    decides if it is set (.env is loaded by database.py at import)."""
    return os.environ.get("DISCOGS_ACCESS_TOKEN", "").strip()


def _consumer() -> tuple:
    """Consumer key/secret pair — the alternative auth if no personal token."""
    return (os.environ.get("DISCOGS_CONSUMER_KEY", "").strip(),
            os.environ.get("DISCOGS_CONSUMER_SECRET", "").strip())


def _auth_header() -> "str | None":
    """The Discogs Authorization header for whichever credentials are present:
    a personal access token (preferred) or a consumer key+secret pair."""
    tok = _token()
    if tok:
        return f"Discogs token={tok}"
    key, secret = _consumer()
    if key and secret:
        return f"Discogs key={key}, secret={secret}"
    return None


# Authenticated Discogs allows 60 requests/minute. Stay under it with a minimum
# spacing between calls (≈55/min) plus 429 back-off — one throttle for the whole
# process, guarded by a lock so concurrent callers still respect the budget.
_MIN_INTERVAL = 1.1
_lock = threading.Lock()
_last_call = 0.0


def is_configured() -> bool:
    """True when Discogs credentials are available (enrichment can run)."""
    return _auth_header() is not None


def _headers() -> dict:
    h = {"User-Agent": USER_AGENT}
    auth = _auth_header()
    if auth:
        h["Authorization"] = auth
    return h


def _throttle() -> None:
    global _last_call
    with _lock:
        wait = _MIN_INTERVAL - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()


def _get(path: str, params: dict = None, timeout: float = 15.0) -> dict:
    """GET a Discogs JSON endpoint with throttling and one 429 back-off."""
    if not is_configured():
        raise RuntimeError("discogs-not-configured")
    url = path if path.startswith("http") else f"{BASE}{path}"
    for attempt in range(2):
        _throttle()
        r = httpx.get(url, params=params, headers=_headers(), timeout=timeout)
        if r.status_code == 429 and attempt == 0:          # rate-limited → wait
            time.sleep(2.0)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()
    return r.json()


# ── normalisation ─────────────────────────────────────────────────────────────
def _norm(s: str) -> str:
    """Loose-match normaliser: lowercase, strip accents-ish noise & bracketed bits."""
    s = (s or "").lower()
    s = re.sub(r"\(.*?\)|\[.*?\]", " ", s)         # drop (Original Mix) / [LABEL001]
    s = re.sub(r"feat\.?|featuring|ft\.?|\bvs\.?\b|\bremix\b|\bedit\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


_MIX_SUFFIX = re.compile(
    r"\s*[\(\[](original mix|original|[^()\[\]]+\s+remix|[^()\[\]]+\s+edit"
    r"|[^()\[\]]+\s+version|[^()\[\]]+\s+dub|[^()\[\]]+\s+mix"
    r"|live edit|live)[)\]]\s*$", re.I
)


def _clean_title(title: str) -> str:
    """Strip common DJ-file suffixes like '(Original Mix)', '(X Remix)', '(Live Edit)'
    so Discogs searches match the release/EP name rather than the expanded track title."""
    return _MIX_SUFFIX.sub("", title or "").strip()


def _first_artist(artist: str) -> str:
    """'Obscure Shape, SHDW' → 'Obscure Shape' — helps when Discogs stores the
    primary artist only. Falls back to full string for single-artist names."""
    for sep in (",", "&", " x ", " X "):
        if sep in artist:
            return artist.split(sep)[0].strip()
    return artist


def _similarity(a: str, b: str) -> float:
    """Token-set Jaccard on normalised strings — cheap, dependency-free 0..1."""
    sa, sb = set(_norm(a).split()), set(_norm(b).split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _release_images(release: dict) -> list:
    """All image URLs on a release, primary first (type == 'primary')."""
    imgs = release.get("images") or []
    imgs = sorted(imgs, key=lambda im: 0 if im.get("type") == "primary" else 1)
    return [im.get("uri") for im in imgs if im.get("uri")]


# ── public API ────────────────────────────────────────────────────────────────
def _parse_results(data: dict) -> list:
    """Convert raw Discogs search results to candidate dicts."""
    out = []
    for r in data.get("results", []):
        out.append({
            "release_id": r.get("id"),
            "title": r.get("title"),                 # "Artist - Release"
            "year": _to_int(r.get("year")),
            "label": (r.get("label") or [None])[0],
            "catno": r.get("catno"),
            "genres": r.get("genre") or [],
            "styles": r.get("style") or [],
            "country": r.get("country"),
            "format": r.get("format") or [],
            "thumb": r.get("thumb"),
            "cover_image": r.get("cover_image"),
            "master_id": r.get("master_id"),
        })
    return out


def search_release(artist: str, title: str, per_page: int = 10) -> list:
    """Search Discogs for releases of `title` by `artist`. Returns lightweight
    candidate dicts (no extra calls). Empty list when nothing/unconfigured.

    Two-pass strategy: first try `track=` (tracklist match, covers individual
    tracks on EPs) then `release_title=` (matches the release name itself, covers
    EPs/albums named after the track). Results are merged and deduplicated.
    """
    if not is_configured():
        return []
    if not artist and not title:
        return []

    base = {"type": "release", "per_page": per_page}
    if artist:
        base["artist"] = artist

    seen_ids: set = set()
    merged: list = []

    def _fetch(extra: dict) -> list:
        try:
            data = _get("/database/search", {**base, **extra})
            return _parse_results(data)
        except Exception as e:
            logger.warning("discogs search failed for %s - %s: %s", artist, title, e)
            return []

    def _add(candidates):
        for c in candidates:
            if c["release_id"] not in seen_ids:
                seen_ids.add(c["release_id"])
                merged.append(c)

    clean = _clean_title(title)          # "Impulse (Original Mix)" → "Impulse"
    first = _first_artist(artist)        # "Obscure Shape, SHDW" → "Obscure Shape"

    # pass 1: track-title (exact, good for individual tracks on EPs)
    if title:
        _add(_fetch({"track": title}))

    # pass 2: release-title (EP/album named after the file title)
    if title:
        _add(_fetch({"release_title": title}))

    # pass 3: cleaned title — strips "(Original Mix)" / "(X Remix)" etc. that
    # rarely appear on Discogs release titles. Also tries first artist only for
    # "Artist1, Artist2" filenames where Discogs stores one primary artist.
    if clean != title or first != artist:
        retry_base = {**base}
        if first != artist:
            retry_base["artist"] = first
        if clean != title:
            _add(_fetch({**retry_base, "track": clean}))
            _add(_fetch({**retry_base, "release_title": clean}))
        elif first != artist:
            _add(_fetch({**retry_base, "track": title}))
            _add(_fetch({**retry_base, "release_title": title}))

    return merged


def get_release(release_id) -> dict:
    """Full release detail — labels, tracklist, images, genres/styles, etc.
    Returns a flattened dict tuned for storage + cover download."""
    r = _get(f"/releases/{release_id}")
    labels = [{"name": l.get("name"), "catno": l.get("catno"), "id": l.get("id")}
              for l in (r.get("labels") or [])]
    return {
        "release_id": r.get("id"),
        "master_id": r.get("master_id"),
        "title": r.get("title"),
        "year": _to_int(r.get("year")),
        "country": r.get("country"),
        "genres": r.get("genres") or [],
        "styles": r.get("styles") or [],
        "labels": labels,
        "label": labels[0]["name"] if labels else None,
        "catno": labels[0]["catno"] if labels else None,
        "artists": [a.get("name") for a in (r.get("artists") or [])],
        "tracklist": [t.get("title") for t in (r.get("tracklist") or []) if t.get("title")],
        "images": _release_images(r),
        "uri": r.get("uri"),
    }


def artist_for_sale(artist: str, limit: int = 5, scan: int = 12) -> list:
    """An artist's vinyl releases currently FOR SALE on the Discogs marketplace
    (second-hand, sold by users), ranked by how the community VALUES them
    (community rating average) and then by recency (year, newest first).

    Two-tier ranking matching the product spec: primarily the most-valued records;
    records without a rating fall back to recency. Only releases with num_for_sale>0
    are kept. Costs one search + up to `scan` release-detail calls (rating +
    num_for_sale live only on the detail), so it is bounded. Empty list when Discogs
    is unconfigured or nothing is for sale — never invents listings."""
    if not is_configured() or not artist:
        return []
    try:
        data = _get("/database/search",
                    {"type": "release", "artist": _first_artist(artist),
                     "format": "Vinyl", "sort": "year", "sort_order": "desc",
                     "per_page": scan})
        cands = _parse_results(data)
    except Exception as e:
        logger.warning("discogs artist_for_sale search failed for %s: %s", artist, e)
        return []
    out = []
    for c in cands:
        rid = c.get("release_id")
        if not rid:
            continue
        try:
            raw = _get(f"/releases/{rid}")           # num_for_sale + rating live here
        except Exception:
            continue
        if (raw.get("num_for_sale") or 0) <= 0:      # only what is actually buyable now
            continue
        rating = ((raw.get("community") or {}).get("rating") or {}).get("average")
        out.append({
            "release_id": rid,
            "title": raw.get("title"),
            "year": _to_int(raw.get("year")),
            "rating": float(rating) if rating else None,
            "num_for_sale": raw.get("num_for_sale"),
            "lowest_price": raw.get("lowest_price"),
            "url": f"https://www.discogs.com/sell/release/{rid}",
        })
    # primary: community rating desc (most valued); secondary: year desc (recent).
    # Rated releases sort above unrated; unrated fall back to recency alone.
    out.sort(key=lambda x: (x["rating"] is not None, x["rating"] or 0.0, x["year"] or 0),
             reverse=True)
    return out[:limit]


def score_match(artist: str, title: str, candidate: dict,
                release: dict = None) -> float:
    """Confidence 0..1 that `candidate` is the release for artist+title.

    Uses the search-result title (≈ "Artist - Release") for a base signal, and
    — when a fetched `release` is supplied — boosts strongly if our track title
    appears in its tracklist (the most reliable confirmation).
    """
    cand_title = candidate.get("title") or ""
    artist_part, _, rel_part = cand_title.partition(" - ")
    a = _similarity(artist, artist_part)
    t_title = _similarity(title, rel_part or cand_title)
    base = 0.6 * a + 0.4 * t_title
    if release and release.get("tracklist"):
        best_tl = max((_similarity(title, t) for t in release["tracklist"]),
                      default=0.0)
        # tracklist confirmation can only raise confidence, never lower it:
        # if the title matches a track → strong boost; if not (EP/album name)
        # the base release-title score already earned its confidence.
        tl_score = 0.4 * a + 0.6 * best_tl
        base = max(base, tl_score)
    return round(min(1.0, base), 3)


# Confidence bands for the auto + confirm-doubtful flow.
AUTO_THRESHOLD = 0.72        # ≥ → accept automatically
DOUBT_THRESHOLD = 0.40       # [DOUBT, AUTO) → queue for the user; < → no match


def best_match(artist: str, title: str, confirm_depth: int = 2) -> dict:
    """Resolve a crate track to a Discogs release.

    Searches, scores the top candidates (fetching the release for the top
    `confirm_depth` to confirm via tracklist), and classifies the outcome:
      status 'matched'   — confident; `release` is the full detail.
      status 'doubtful'  — plausible; `candidates` is a shortlist to confirm.
      status 'unmatched' — nothing convincing.
    Never invents: a weak result becomes 'unmatched', not a fake match.
    """
    if not is_configured():
        return {"status": "unconfigured", "candidates": []}
    cands = search_release(artist, title)
    if not cands:
        return {"status": "unmatched", "candidates": []}

    # cheap pre-score, then confirm the top few with a release fetch
    for c in cands:
        c["score"] = score_match(artist, title, c)
    cands.sort(key=lambda c: c["score"], reverse=True)

    best, best_rel = None, None
    for c in cands[:confirm_depth]:
        try:
            rel = get_release(c["release_id"])
        except Exception as e:
            logger.warning("discogs release %s fetch failed: %s", c["release_id"], e)
            continue
        c["score"] = score_match(artist, title, c, release=rel)
        if best is None or c["score"] > best["score"]:
            best, best_rel = c, rel
    cands.sort(key=lambda c: c["score"], reverse=True)

    if best and best["score"] >= AUTO_THRESHOLD:
        return {"status": "matched", "confidence": best["score"],
                "release": best_rel, "candidates": cands[:5]}
    if cands and cands[0]["score"] >= DOUBT_THRESHOLD:
        return {"status": "doubtful", "confidence": cands[0]["score"],
                "candidates": cands[:5]}
    return {"status": "unmatched", "confidence": cands[0]["score"] if cands else 0.0,
            "candidates": cands[:5]}


def download_cover(image_url: str, dest: Path) -> "Path | None":
    """Download a cover image to `dest` (parents created). Returns the path, or
    None on failure. Honours the Discogs User-Agent/token requirements."""
    if not image_url:
        return None
    try:
        _throttle()
        r = httpx.get(image_url, headers=_headers(), timeout=30.0,
                      follow_redirects=True)
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        return dest
    except Exception as e:
        logger.warning("discogs cover download failed (%s): %s", image_url, e)
        return None


def _to_int(v) -> "int | None":
    try:
        return int(str(v)[:4]) if v else None
    except (ValueError, TypeError):
        return None
