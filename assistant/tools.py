"""
assistant/tools.py — the assistant's tools (Phase 1).

Typed functions the LLM may call. Each is a thin wrapper over database.py /
analyze.py — NO new query logic lives here. Recommendations are anchored in
AUDIO embeddings (the strongest signal); metadata only filters/annotates.

These are plain functions (DB-only, no LLM) so they unit-test on their own; the
agent registers them in agent.py.
"""

import analyze
import database

_MV = None


def _effnet_version() -> str:
    global _MV
    if _MV is None:
        _MV = analyze._model_version("effnet")
    return _MV


def _row(r: dict, distance: float = None) -> dict:
    """Format a track DB row into a clean recommendation dict for the LLM."""
    f = r.get("features") or {}
    out = {
        "track_id": str(r["track_id"]),
        "artists": database.track_artist_names(str(r["track_id"])),
        "filename": r.get("filename"),
        "bpm": round(f["bpm"], 1) if f.get("bpm") else None,
        "camelot": f.get("camelot"),
    }
    if distance is not None:
        out["similarity"] = round(1.0 - distance, 3)
        out["distance"] = round(distance, 3)
    return out


def audio_similarity(query: str = "", track_id: str = "", n: int = 5) -> dict:
    """Tracks that SOUND like a track or artist (EffNet audio embeddings — the STRONGEST
    recommendation signal; prefer for "recommend something like X"). Pass `track_id` for
    "tracks like this", or `query` (artist name or part of a title). Returns {resolved,
    results:[{track_id, artists, filename, bpm, camelot, similarity}]} or {error}."""
    mv = _effnet_version()
    n = max(1, min(int(n or 5), 20))
    vec, resolved, exclude = None, None, None
    if track_id:
        vec = database.get_track_embedding(track_id, mv)
        exclude = track_id
        row = database.get_track(track_id) if vec is not None else None
        resolved = row.get("filename") if row else track_id
    elif query:
        art = database.get_artist(query)
        if art:
            vec = database.get_artist_embedding(str(art["artist_id"]), mv)
            resolved = f"artist: {art['name']}"
        if vec is None:                                   # fall back to a track match
            row = database.find_track_by_query(query)
            if row:
                vec = database.get_track_embedding(str(row["track_id"]), mv)
                exclude = str(row["track_id"])
                resolved = f"track: {row['filename']}"
    if vec is None:
        return {"error": f"could not find anything matching '{query or track_id}' "
                         f"in the collection.", "results": []}
    hits = database.find_similar_effnet(vec, n=n + (1 if exclude else 0),
                                        exclude_track_id=exclude)
    results = [_row(h, float(h.get("cosine_distance", 0.0))) for h in hits][:n]
    return {"resolved": resolved, "results": results}


def similar_artists(artist: str, n: int = 5) -> dict:
    """ARTISTS whose overall sound resembles a given artist (audio centroid ANN). For
    "artists like Dasha Rush / Oscar Mulero". Returns {resolved, results:[{name, n_tracks,
    similarity}]} or {error}."""
    mv = _effnet_version()
    art = database.get_artist(artist)
    if not art:
        return {"error": f"unknown artist '{artist}' in the collection.", "results": []}
    vec = database.get_artist_embedding(str(art["artist_id"]), mv)
    if vec is None:
        return {"error": f"'{art['name']}' has no audio centroid yet.", "results": []}
    hits = database.find_similar_artists(vec, n=max(1, min(int(n or 5), 15)) + 1,
                                         exclude_artist_id=str(art["artist_id"]))
    results = [{"name": h["name"], "n_tracks": h["n_tracks"],
               "similarity": round(1.0 - float(h["cosine_distance"]), 3)} for h in hits]
    return {"resolved": art["name"], "results": results}


def similar_labels(label: str, n: int = 5) -> dict:
    """LABELS whose roster sound resembles a given label (audio centroid ANN, same signal
    as artists). For "labels like Ostgut Ton / Token"; only labels enriched from Discogs
    are known. Returns {resolved, results:[{name, n_tracks, similarity}]} or {error}."""
    mv = _effnet_version()
    lab = database.get_label(label)
    if not lab:
        return {"error": f"unknown label '{label}' (not enriched yet).", "results": []}
    vec = database.get_label_embedding(str(lab["label_id"]), mv)
    if vec is None:
        return {"error": f"'{lab['name']}' has no audio centroid yet.", "results": []}
    hits = database.find_similar_labels(vec, n=max(1, min(int(n or 5), 15)) + 1,
                                        exclude_label_id=str(lab["label_id"]))
    results = [{"name": h["name"], "n_tracks": h["n_tracks"],
               "similarity": round(1.0 - float(h["cosine_distance"]), 3)} for h in hits]
    return {"resolved": lab["name"], "results": results}


def metadata_search(artist: str = "", bpm_min: float = None, bpm_max: float = None,
                    camelot: str = "", on_spot: bool = None, n: int = 20) -> dict:
    """Catalogue search by artist / BPM range / Camelot key / on-spot — categorical
    queries ("tracks by Kwartz", "130-136 BPM", "what's on spot"). Returns
    {results:[{track_id, artists, filename, bpm, camelot}], count}."""
    rows = database.search_tracks(
        artist=artist or None, bpm_min=bpm_min, bpm_max=bpm_max,
        camelot=camelot or None, on_spot=on_spot, limit=int(n or 20))
    return {"results": [_row(r) for r in rows], "count": len(rows)}


# RAG retrieval over the text knowledge base. Distance >0.6 is essentially
# unrelated for nomic-embed-text, so it is dropped — the LLM is never handed
# off-topic context to hallucinate from.
_KB_MAX_DISTANCE = 0.6


def kb_rag_search(query: str, n: int = 5, category: str = "") -> dict:
    """Search the user's ingested knowledge base for FACTUAL/encyclopaedic answers about
    music (artists, labels, genres, history, scenes, gear, theory, books) — "who is…",
    "history of…", "what does theory say about…". Pass `category` (e.g. "music-theory",
    "dj", "label", "book") ONLY when the question clearly targets one kind; else leave it
    empty. Ground replies in the returned passages, add nothing not there. Returns
    {results:[{text, title, category, score}], count} best-first (empty when nothing fits)."""
    from assistant import embed_text
    n = max(1, min(int(n or 5), 10))
    try:
        qvec = embed_text.embed_query(query)
    except Exception as e:
        return {"error": f"knowledge base unavailable: {e}", "results": []}
    hits = database.search_kb_chunks(qvec, n=n, max_distance=_KB_MAX_DISTANCE,
                                     category=(category or None))
    results = [{"text": h["text"], "title": h["title"], "category": h.get("category"),
               "score": round(1.0 - float(h["cosine_distance"]), 3)} for h in hits]
    return {"results": results, "count": len(results)}


def resident_advisor_search(query: str = "", location: str = "", date: str = "",
                            kind: str = "events", n: int = 6) -> dict:
    """Look up the LIVE electronic-music scene on Resident Advisor (es.ra.co) —
    the web complement to kb_rag_search (which is only the user's own notes).

    Use it for things that live on RA rather than in the collection: upcoming
    EVENTS/parties, and ARTIST or LABEL info or NEWS or VENUES. `kind` selects what to look up:
      • A SPECIFIC ARTIST's upcoming gigs ANYWHERE ("where is X playing?", "next
        events of X") → put the artist name in `query` and leave `location` EMPTY.
        No date needed — it returns that artist's upcoming events in every city.
        (You may also set kind="artist_events".)
      • kind="events" in a CITY on/around a date → pass BOTH `location` (e.g.
        "Madrid") AND `date` ("2026-06-20", "this weekend", "tonight"). If a city
        is intended but EITHER is missing, ASK the user, then call again.
      • kind="artist" / kind="label" → profile/info only (no events); name in `query`.

    Returns {"source": "ra-graphql"|"ra-web", "results": [...]}. Events carry
    title/date/venue/where/artists/url (render DATE · EVENT · VENUE · LINE-UP ·
    CITY, using `where` for CITY); artists/labels carry name/type/url. Returns {"need_info": [...]}
    when a city/date is required, or {"error": ...} when RA cannot be reached — in
    which case say you do not have it, never invent events.
    """
    from assistant import web_sources
    from assistant import profile
    kind = (kind or "events").strip().lower()
    n = max(1, min(int(n or 6), 15))
    if kind in ("event", "events", "gig", "party", "artist_events", "artist-events"):
        artist = query.strip()
        # An artist's gigs ANYWHERE: name given and no explicit city (or asked for
        # by kind) → use the artist→events path, which needs no city/date.
        if kind in ("artist_events", "artist-events") or (artist and not location.strip()):
            if not artist:
                return {"need_info": ["query"],
                        "message": "Ask the user which artist's events to look up."}
            return web_sources.find_artist_events(artist, limit=n)
        # Otherwise a city browse: needs a city (remembered location is fine) + date.
        loc = location.strip() or (profile.get_location() or "")
        dt = date.strip()
        if not loc or not dt:
            need = ([] if loc else ["location"]) + ([] if dt else ["date"])
            return {"need_info": need,
                    "message": f"To recommend events I need the {' and '.join(need)}. "
                               f"Ask the user before searching."}
        return web_sources.find_events(loc, dt, artist, limit=n)
    # artist / label / news lookup
    term = query.strip() or location.strip()
    if not term:
        return {"need_info": ["query"],
                "message": "Ask the user which artist or label to look up."}
    return web_sources.find_info(term, kind=kind, limit=n)


def set_user_location(location: str) -> dict:
    """Remember WHERE THE USER PHYSICALLY IS (a city/area), so location-dependent
    answers (events, "what's on near me") know it without asking again.

    Call this whenever the user states where they are — "I'm in Madrid", "this
    weekend I'm in Ibiza", "I live in Berlin". It persists across sessions until
    updated and is shown to you in the context line each turn. Pass an empty
    string to clear it. Returns {"saved_location": …}.
    """
    from assistant import profile
    loc = profile.set_location(location)
    return {"saved_location": loc} if loc else {"saved_location": None,
                                                "note": "location cleared"}


def vinyl_stock_search(query: str = "", n: int = 8, store: str = "") -> dict:
    """Search record shops for VINYL IN STOCK right now (live availability check).

    Use when the user wants to BUY or find a record to purchase: "is X in stock",
    "where can I buy Y", "find the Surgeon LP", "available vinyl by Z". Pass an
    artist, a release title, or "artist title" as `query`. `store` optionally limits
    to one shop (e.g. "hardwax"); empty searches all configured shops. This is the
    live shop check — not the user's own collection (use audio_similarity/metadata
    for that).

    Returns {"query", "stores", "results": [{store, artist, title, label, format,
    price, currency, in_stock, url}, …]} ordered in-stock first, PLUS a separate
    "marketplace" list (up to 5 of the artist's records for sale SECOND-HAND on
    Discogs, most community-valued first: {title, year, rating, num_for_sale,
    lowest_price, url}). Present the first-hand shop "results" first, then the
    Discogs "marketplace" as a complement. in_stock is true/false/null (null = the
    shop did not say). Report only what is returned — never invent stock or listings.
    """
    from assistant import vinyl_stores
    n = max(1, min(int(n or 8), 20))
    stores = [store.strip().lower()] if store.strip() else None
    return vinyl_stores.search_stock(query.strip(), stores=stores, limit=n)


# Beyond this cosine distance a cached web snippet is essentially unrelated for
# nomic-embed-text, so it is dropped (same guard as kb_rag_search).
_WEB_CACHE_MAX_DISTANCE = 0.6


def reference_search(query: str, n: int = 6) -> dict:
    """Search the user's REGISTERED reference websites (added on the Knowledge page)
    for fresh info, remembering what it finds.

    Use for music questions the collection, the knowledge base and Resident Advisor
    do not cover but a site the user trusts might — a gear forum, a blog, a label
    site, a wiki. Each registered site carries a topic describing what it is for.
    PRIMARY: a live web search scoped to those sites (results are embedded into a
    local cache so they are reusable). FALLBACK when the live web is unreachable: the
    semantic cache of past results / page snapshots.

    Returns {"source": "web"|"cache", "results": [{title, url, snippet, topic?}, …]}
    or an empty list with a note — never invent pages, facts or links.
    """
    from urllib.parse import urlparse
    import config
    from assistant import web_sources, embed_text
    query = (query or "").strip()
    n = max(1, min(int(n or 6), 12))
    if not query:
        return {"results": [], "note": "empty query."}
    sources = database.list_web_sources()
    if not sources:
        return {"results": [], "note": "no reference websites registered yet — the "
                                       "user can add some on the Knowledge page."}
    by_host = {}                                   # host → source row (topic + id)
    for s in sources:
        host = urlparse(s["url"]).netloc.lower()
        if host:
            by_host[host] = s

    def _match(url: str) -> "dict | None":
        h = urlparse(url).netloc.lower()
        return next((s for host, s in by_host.items() if host in h), None)

    # PRIMARY — live scoped web search over the registered domains.
    try:
        hits = web_sources.search_domains(query, list(by_host), limit=n)
    except Exception:
        hits = []
    if hits:
        results = [{**h, "topic": (_match(h["url"]) or {}).get("topic")} for h in hits]
        try:                                       # persist embeddings (best-effort)
            texts = [f"{r['title']} {r.get('snippet', '')}".strip() for r in results]
            vecs = embed_text.embed_documents(texts)
            database.insert_web_cache([
                {"source_id": (_match(r["url"]) or {}).get("source_id"),
                 "query": query, "title": r["title"], "url": r["url"],
                 "text": t, "embedding": v}
                for r, t, v in zip(results, texts, vecs)])
            database.evict_web_cache(config.WEB_CACHE_MAX_CHUNKS)
        except Exception:
            pass
        return {"source": "web", "results": results}

    # FALLBACK — semantic cache of past results / page snapshots.
    try:
        qv = embed_text.embed_query(query)
        hits = database.search_web_cache(qv, n=n, max_distance=_WEB_CACHE_MAX_DISTANCE)
        results = [{"title": h["title"], "url": h["url"], "snippet": h["text"][:300],
                    "topic": None,
                    "score": round(1.0 - float(h["cosine_distance"]), 3)} for h in hits]
        if results:
            return {"source": "cache", "results": results,
                    "note": "live web unavailable — showing cached past results."}
    except Exception:
        pass
    return {"results": [], "note": "no live results and nothing cached for this query."}


# ── unified live-web tool ───────────────────────────────────────────────────────
# One LLM-facing tool over the three live-web lookups above. The small model picks a
# `kind` instead of choosing among separate tools — fewer schemas in the prompt and far
# less routing confusion for a 4B. The branches reuse the SAME functions (now internal
# helpers), so the isolated service adapters (web_sources.py for RA + registered sites,
# vinyl_stores.py for shops) stay untouched. The result is tagged with the resolved
# `kind` so the model knows which render shape it got.
_VINYL_KINDS = {"vinyl", "stock", "buy", "shop", "record", "records", "store"}
_REFERENCE_KINDS = {"reference", "web", "site", "sites", "page", "pages", "ref"}
_PROFILE_KINDS = {"profile", "artist", "label", "news"}


def _auto_kind(query: str, location: str, date: str) -> str:
    """Deterministic intent guess for kind='auto' (no LLM call): a keyword decides which
    SINGLE source to hit, so auto never fans out to all three and stacks their latencies."""
    q = (query or "").lower()
    if any(w in q for w in ("buy", "stock", "vinyl", "record", "price", "for sale", "in stock")):
        return "vinyl"
    if location or date or any(w in q for w in (
            "event", "gig", "playing", "play ", "festival", "club night", "tonight",
            "this weekend", "line-up", "lineup", "party", "where is", "what's on", "whats on")):
        return "events"
    return "reference"


def music_web_search(query: str = "", kind: str = "auto", location: str = "",
                     date: str = "", n: int = 6) -> dict:
    """Search the LIVE web for music, all behind one tool. Choose `kind`:
      - "events": upcoming gigs/parties on Resident Advisor. For ONE artist's gigs
        ANYWHERE, put the artist in `query` and leave `location` EMPTY. For a CITY's
        listings pass BOTH `location` and `date` (use the known location from context;
        ask only if a city is intended but unknown).
      - "profile": RA artist / label / news info (no events); name in `query`.
      - "vinyl": record shops checked live for stock to BUY, plus the Discogs second-hand
        marketplace. Pass artist and/or title in `query`. Present shop results first, then
        the marketplace.
      - "reference": the user's OWN registered reference websites (added on the Knowledge
        page), searched live with a semantic-cache fallback.
      - "auto" (default): routes to the single best source for the query.

    Returns the matched source's data with an added `kind` field (so you know the render
    shape), or {"need_info": [...]} / {"error": ...}. Never invent results — if a source
    errors or returns nothing, say the live source has nothing.
    """
    k = (kind or "auto").strip().lower()
    if k == "auto":
        k = _auto_kind(query, location, date)
    if k in _VINYL_KINDS:
        return {"kind": "vinyl", **vinyl_stock_search(query=query, n=n)}
    if k in _REFERENCE_KINDS:
        return {"kind": "reference", **reference_search(query, n=n)}
    if k in _PROFILE_KINDS:
        ra_kind = k if k in ("label", "news") else "artist"
        return {"kind": "profile",
                **resident_advisor_search(query=query, location=location, date=date,
                                          kind=ra_kind, n=n)}
    # default (and the auto-events branch) → live events
    return {"kind": "events",
            **resident_advisor_search(query=query, location=location, date=date,
                                      kind="events", n=n)}
