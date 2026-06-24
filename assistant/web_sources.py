"""
web_sources.py — the agent's live "scene" lookup (events, artists, labels).

Complements the static vector RAG (assistant/kb.py) with FRESH data from the web.
Resident Advisor has no official API, so the primary source is RA's own GraphQL
endpoint (config.RA_GRAPHQL); it is undocumented and can change, so every lookup
degrades gracefully:

    1. PRIMARY  — RA GraphQL (areas + eventListings for events; search for
                  artists/labels). Structured, no key needed.
    2. FALLBACK — a plain web search scoped to config.WEB_SOURCES (es.ra.co by
                  default, user-expandable) when the GraphQL query fails.
    3. GUARDRAIL— if BOTH fail, return {"error": …} so the assistant says it does
                  not have the info instead of inventing.

Every primary/fallback failure is written to logs/web_sources.log (and the
logger) so a future auto-update routine can notice that RA changed and adapt.

Isolated like discogs.py: nothing else in The Crate imports httpx for this or
knows RA's wire format.
"""
import datetime
import logging
import re

import httpx

import config
from assistant import webutil

logger = logging.getLogger("The Crate")

# RA needs a browser-like UA + Referer (Cloudflare-fronted; the GraphQL endpoint expects
# a site-like request). The UA itself is shared with the other web adapters (webutil.UA).
_UA = webutil.UA
_HEADERS = {"User-Agent": _UA, "Content-Type": "application/json",
            "Referer": "https://ra.co/", "Origin": "https://ra.co"}
_RA_SITE = "https://es.ra.co"          # for turning relative contentUrl into links

_FAIL_LOG = config.PROJECT_ROOT / "logs" / "web_sources.log"


def _log_failure(stage: str, detail) -> None:
    """Record a primary/fallback failure for future auto-update awareness."""
    webutil.log_failure(_FAIL_LOG, logger, "web_sources", stage, detail)


# ── RA GraphQL (primary) ──────────────────────────────────────────────────────
def _gql(query: str, variables: dict, timeout: float = config.WEB_HTTP_TIMEOUT) -> dict:
    """POST a GraphQL query to RA. Raises on transport / GraphQL errors."""
    r = httpx.post(config.RA_GRAPHQL, headers=_HEADERS, timeout=timeout,
                   json={"query": query, "variables": variables})
    r.raise_for_status()
    data = r.json()
    if data.get("errors"):
        raise RuntimeError(f"graphql errors: {data['errors'][:1]}")
    return data.get("data") or {}


_AREAS_Q = ("query($s:String){ areas(searchTerm:$s, limit:1){ "
            "id name urlName country { name urlCode } eventsCount } }")

_EVENTS_Q = (
    "query($f:FilterInputDtoInput,$ps:Int){ eventListings(filters:$f, pageSize:$ps, "
    "sort:{listingDate:{order:ASCENDING}}){ totalResults data { event { "
    "title date contentUrl venue { name area { name country { name } } } "
    "artists { name } } } } }")

_SEARCH_Q = ("query($s:String,$i:[IndexType!],$l:Int){ search(searchTerm:$s, "
             "indices:$i, limit:$l){ value searchType contentUrl areaName "
             "countryName date clubName } }")

# A specific artist's UPCOMING events, ANY location (no area/date filter). RA's
# eventListings has no artist filter (introspected: it filters by area/date/genre
# only), so artist gigs come off the artist node itself: artist(slug).events with
# type:LATEST = the upcoming listings the RA artist page shows.
_ARTIST_EVENTS_Q = (
    "query($s:String!,$l:Int){ artist(slug:$s){ name upcomingEventsCount "
    "events(limit:$l, type:LATEST){ id title date contentUrl "
    "venue { name area { name country { name } } } artists { name } } } }")


def _resolve_area(location: str) -> "dict | None":
    """City/region name → RA area ({id, name, country}) via the areas query."""
    data = _gql(_AREAS_Q, {"s": location})
    areas = data.get("areas") or []
    return areas[0] if areas else None


def _link(content_url: str) -> str:
    if not content_url:
        return ""
    return content_url if content_url.startswith("http") else _RA_SITE + content_url


def _depipe(s):
    """Markdown tables are pipe-delimited, so a literal '|' inside a value (RA venue
    names like 'Berghain | Panorama Bar | Säule') would split the cell and shift every
    column. Swap it for '/'. Non-strings pass through."""
    return re.sub(r"\s*\|\s*", " / ", s).strip() if isinstance(s, str) else s


def _where(city: "str | None", country: "str | None") -> str:
    """Render an event's place for the CITY column: 'Berlin (Germany)'. Festivals
    sit in the synthetic 'All' area, so there we show just the country."""
    if not city or city.lower() == "all":
        return country or ""
    return f"{city} ({country})" if country else city


def _map_event(ev: dict) -> dict:
    """One RA event node → the flat shape the assistant renders (DATE · EVENT ·
    VENUE · LINE-UP · CITY). Shared by the city-browse and artist-events paths. Only
    the rendered fields are returned (the raw 'All' area is folded into `where`) so
    the small model has nothing misleading to drop into a column, and every text
    value is pipe-stripped so it can't break the markdown table."""
    venue = ev.get("venue") or {}
    area = venue.get("area") or {}
    return {
        "title": _depipe(ev.get("title")),
        "date": (ev.get("date") or "")[:10],
        "venue": _depipe(venue.get("name")),
        "where": _where(area.get("name"), (area.get("country") or {}).get("name")),
        "artists": [_depipe(a.get("name")) for a in (ev.get("artists") or [])],
        "url": _link(ev.get("contentUrl")),
    }


def _lead_artist(names: list, lead: str) -> None:
    """In-place: float the queried artist to the front of a line-up so the row
    reads 'Red Rooms, Adiel, …' rather than burying them mid-bill."""
    if not names or not lead:
        return
    for i, n in enumerate(names):
        if (n or "").lower() == lead.lower():
            names.insert(0, names.pop(i))
            return


def _resolve_artist_slug(name: str) -> "str | None":
    """Artist name → RA url slug (the '/dj/<slug>' segment) via the search index.
    Takes the first ARTIST hit (RA returns the best match first)."""
    data = _gql(_SEARCH_Q, {"s": name, "i": ["ARTIST"], "l": 5})
    for r in (data.get("search") or []):
        if (r.get("searchType") or "").upper() == "ARTIST" and r.get("contentUrl"):
            return r["contentUrl"].rstrip("/").split("/")[-1]
    return None


# ── date handling ─────────────────────────────────────────────────────────────
def _date_window(date_str: str) -> "tuple[str, str, str]":
    """Turn a user date phrase into an (gte, lte, human-label) ISO window.

    Accepts an ISO date (YYYY-MM-DD), or simple phrases in EN/ES: today/tonight/
    hoy, tomorrow/mañana, this weekend/finde, this week/semana. Anything else
    falls back to a 7-day window from today.
    """
    s = (date_str or "").strip().lower()
    today = datetime.date.today()

    def iso(d0: datetime.date, d1: datetime.date, label: str):
        return (f"{d0.isoformat()}T00:00:00.000Z",
                f"{d1.isoformat()}T23:59:59.000Z", label)

    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        d = datetime.date(int(m[1]), int(m[2]), int(m[3]))
        return iso(d, d, d.isoformat())
    if any(w in s for w in ("tonight", "today", "hoy", "esta noche", "now")):
        return iso(today, today, "today")
    if any(w in s for w in ("tomorrow", "mañana", "manana")):
        d = today + datetime.timedelta(days=1)
        return iso(d, d, "tomorrow")
    if any(w in s for w in ("weekend", "finde", "fin de semana", "sábado", "sabado", "saturday")):
        sat = today + datetime.timedelta(days=(5 - today.weekday()) % 7)
        return iso(sat, sat + datetime.timedelta(days=1), "this weekend")
    # default: a week from today
    return iso(today, today + datetime.timedelta(days=6), "this week")


# ── web fallback (DuckDuckGo HTML, scoped to WEB_SOURCES) ──────────────────────
def _web_fallback(query: str, limit: int) -> list:
    """RA's web fallback: a DuckDuckGo search scoped to the configured WEB_SOURCES domains
    (es.ra.co by default). Thin wrapper over the shared scoped search (webutil.ddg_search)."""
    return webutil.ddg_search(query, config.WEB_SOURCES or ["es.ra.co"], limit,
                              config.WEB_HTTP_TIMEOUT)


# ── public API (used by the assistant's music_web_search tool) ────────────────
def find_events(location: str, date: str, query: str = "", limit: int = 6) -> dict:
    """Events on Resident Advisor for a city + date. GraphQL first, web fallback,
    then a guardrail error. `query` optionally filters by a term (artist/genre)."""
    gte, lte, label = _date_window(date)
    # 1) PRIMARY — RA GraphQL
    try:
        area = _resolve_area(location)
        if not area:
            # Not a transport failure — RA simply has no such area. Try the web
            # fallback (handles misspellings / smaller towns) before giving up.
            raise LookupError(f"no RA area for '{location}'")
        flt = {"areas": {"eq": int(area["id"])},
               "listingDate": {"gte": gte, "lte": lte}}
        data = _gql(_EVENTS_Q, {"f": flt, "ps": max(1, min(limit, 20))})
        listing = (data.get("eventListings") or {})
        rows = listing.get("data") or []
        events = []
        for row in rows:
            ev = row.get("event") or {}
            if query and query.lower() not in (ev.get("title") or "").lower() \
               and not any(query.lower() in (a.get("name") or "").lower()
                           for a in (ev.get("artists") or [])):
                continue
            events.append(_map_event(ev))
        return {"source": "ra-graphql", "area": area["name"], "window": label,
                "total": listing.get("totalResults", len(events)),
                "results": events[:limit]}
    except Exception as e:
        _log_failure("events:graphql", e)

    # 2) FALLBACK — web search scoped to RA
    try:
        hits = _web_fallback(f"{location} {query} events {label}".strip(), limit)
        if hits:
            return {"source": "ra-web", "window": label, "results": hits,
                    "note": "RA GraphQL unavailable — showing web results scoped to "
                            "Resident Advisor; verify dates on the linked pages."}
        _log_failure("events:web", "no results")
    except Exception as e:
        _log_failure("events:web", e)

    # 3) GUARDRAIL
    return {"error": f"could not reach Resident Advisor for events in '{location}'. "
                     f"Tell the user the live source is unavailable right now.",
            "results": []}


def find_artist_events(artist: str, limit: int = 6) -> dict:
    """Upcoming events for a specific ARTIST, in ANY location (no city/date needed).

    Answers "where is <artist> playing?" — the gap city-scoped find_events cannot
    cover. RA exposes this only on the artist node (artist(slug).events, type:LATEST),
    so we resolve the name to a slug, then read its upcoming listings. GraphQL first,
    web fallback, then a guardrail error. Each result carries the same DATE/EVENT/
    VENUE/LINE-UP/CITY shape as find_events, with the queried artist led in the line-up.
    """
    artist = (artist or "").strip()
    # 1) PRIMARY — RA GraphQL: artist → upcoming events
    try:
        slug = _resolve_artist_slug(artist)
        if not slug:
            raise LookupError(f"no RA artist matching '{artist}'")
        data = _gql(_ARTIST_EVENTS_Q, {"s": slug, "l": max(1, min(limit, 20))})
        art = data.get("artist") or {}
        events = [_map_event(ev) for ev in (art.get("events") or [])]
        for ev in events:
            _lead_artist(ev["artists"], art.get("name") or artist)
        return {"source": "ra-graphql", "artist": art.get("name") or artist,
                "total": art.get("upcomingEventsCount", len(events)),
                "results": events[:limit]}
    except Exception as e:
        _log_failure("artist_events:graphql", e)

    # 2) FALLBACK — web search scoped to RA
    try:
        hits = _web_fallback(f"{artist} events", limit)
        if hits:
            return {"source": "ra-web", "results": hits,
                    "note": "RA GraphQL unavailable — web results scoped to "
                            "Resident Advisor; verify dates on the linked pages."}
        _log_failure("artist_events:web", "no results")
    except Exception as e:
        _log_failure("artist_events:web", e)

    # 3) GUARDRAIL
    return {"error": f"could not reach Resident Advisor for {artist or 'that artist'}'s "
                     f"events right now. Tell the user the live source is unavailable.",
            "results": []}


_KIND_INDICES = {
    "artist": ["ARTIST"], "label": ["LABEL"], "event": ["EVENT"],
    "news": ["NEWS", "REVIEW", "FEATURE"], "release": ["LABEL", "REVIEW"],
}


def find_info(query: str, kind: str = "artist", limit: int = 6) -> dict:
    """Artist / label / news lookup on RA. GraphQL `search` first, web fallback,
    then a guardrail error."""
    indices = _KIND_INDICES.get(kind, ["ARTIST", "LABEL", "EVENT"])
    # 1) PRIMARY — RA GraphQL search
    try:
        data = _gql(_SEARCH_Q, {"s": query, "i": indices, "l": max(1, min(limit, 15))})
        rows = data.get("search") or []
        results = [{
            "name": r.get("value"),
            "type": r.get("searchType"),
            "area": r.get("areaName"),
            "country": r.get("countryName"),
            "date": (r.get("date") or "")[:10] if r.get("date") else None,
            "url": _link(r.get("contentUrl")),
        } for r in rows]
        if results:
            return {"source": "ra-graphql", "results": results[:limit]}
        _log_failure(f"info:{kind}:graphql", "no results")
    except Exception as e:
        _log_failure(f"info:{kind}:graphql", e)

    # 2) FALLBACK — web search scoped to RA
    try:
        hits = _web_fallback(query, limit)
        if hits:
            return {"source": "ra-web", "results": hits,
                    "note": "RA GraphQL unavailable — web results scoped to RA."}
        _log_failure(f"info:{kind}:web", "no results")
    except Exception as e:
        _log_failure(f"info:{kind}:web", e)

    # 3) GUARDRAIL
    return {"error": f"could not find '{query}' on Resident Advisor right now. "
                     f"Say the live source has nothing instead of inventing.",
            "results": []}


# ── generic registered-source search (assistant web scouting) ──────────────────
def search_domains(query: str, domains: list, limit: int = 6) -> list:
    """DuckDuckGo search scoped to the user's OWN registered reference domains — the same
    scoped search as the RA fallback, just with a caller-supplied domain list. Returns
    on-domain {title, url, snippet} dicts; empty when there are no domains. Raises on
    transport errors so the caller can fall back to the cache."""
    domains = [d.lower() for d in (domains or []) if d]
    if not domains:
        return []
    return webutil.ddg_search(query, domains, limit, config.WEB_HTTP_TIMEOUT)


def fetch_page_text(url: str, limit_chars: int = 12000) -> str:
    """Fetch a page and return its visible text (script/style + tags stripped),
    capped. A best-effort snapshot for the web cache; raises on transport errors so
    the caller can skip seeding rather than crash."""
    r = httpx.get(url, headers={"User-Agent": _UA}, timeout=config.WEB_HTTP_TIMEOUT,
                  follow_redirects=True)
    r.raise_for_status()
    body = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", r.text, flags=re.S | re.I)
    return webutil.clean_html(body, tag_repl="", unescape_entities=False)[:limit_chars]
