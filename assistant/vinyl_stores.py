"""
assistant/vinyl_stores.py — "is this record in stock right now?" across shops.

Isolated like discogs.py / web_sources.py: nothing else in The Crate imports httpx
for record shops or knows their HTML. Each shop is its OWN adapter because both the
markup and the stock signal differ per shop; a shop that blocks us, changes layout
or times out degrades to nothing (logged to logs/vinyl_stores.log) and never breaks
the tool.

Hardwax (hardwax.com) is the reference shop and the one fully parsed: artist, title,
format, price and the live in/out-of-stock state. deejay.de / decks.de / hhv.de are
left as adapter STUBS — deejay renders results in a JS-built layout where the title
and price are not parseably tied to the stock cell, decks.de hard-blocks scrapers
(HTTP 403), and hhv.de exposes no scrapable search endpoint. They are wired into the
registry so a bespoke parser can drop in later; only shops listed in
config.VINYL_STORES are queried (default: hardwax).
"""
import concurrent.futures as _cf
import logging
import re
from urllib.parse import quote_plus, unquote

import httpx

import config
from assistant import webutil

logger = logging.getLogger("thecrate.vinyl_stores")

# Several shops 403 a default client, so use the shared browser-like UA. Per-shop
# timeouts keep one slow shop from stalling the whole search (adapters run concurrently).
_UA = webutil.UA
_TIMEOUT = config.VINYL_HTTP_TIMEOUT
_FAIL_LOG = config.PROJECT_ROOT / "logs" / "vinyl_stores.log"


def _log_failure(store: str, detail) -> None:
    """Record a shop failure so a future fix can see which adapter went stale."""
    webutil.log_failure(_FAIL_LOG, logger, "vinyl_stores", store, detail)


def _get(url: str) -> str:
    r = httpx.get(url, headers={"User-Agent": _UA}, timeout=_TIMEOUT,
                  follow_redirects=True)
    r.raise_for_status()
    return r.text


def _deslug(s: str) -> str:
    """Turn a URL slug into a display name: 'model-500' → 'Model 500',
    'applications-ii' → 'Applications II' (roman numerals upper-cased)."""
    out = []
    for w in unquote(s).replace("-", " ").replace("_", " ").split():
        if re.fullmatch(r"[ivxlcdm]+", w):
            out.append(w.upper())
        elif w.isupper():
            out.append(w)
        else:
            out.append(w.capitalize())
    return " ".join(out)


# ── Hardwax (the reference shop, fully parsed) ──────────────────────────────────
# Release links are the one stable anchor (the CSS classes are minified and churn):
#   /<release_id>/<artist-slug>/<title-slug>/
# Everything for a release sits between its link and the next release's link, so we
# group the page that way and read the format, the vinyl price and the stock text
# out of each block.
_HW_ANCHOR = re.compile(r'href="/(\d+)/([^/"?]+)/([^/"?]+)/"')
_HW_FORMAT = re.compile(r'\b(2x12"|3x12"|12"|10"|7"|Do LP|Mini LP|LP|EP|Box|CD)\b')
_HW_PRICE = re.compile(r"€\s*([0-9]+(?:[.,][0-9]{2})?)")


def _hardwax(query: str, limit: int) -> list:
    """Parse hardwax.com/?find=<query> (the real search; ?q= is ignored and just
    returns the homepage). The vinyl price is the LARGEST € in a release block
    (track/MP3 prices are small, ~1–7€). Returns record dicts (see search_stock)."""
    html = _get(f"https://hardwax.com/?find={quote_plus(query)}")
    # Stock is keyed by RELEASE ID, not by text: each item carries BOTH "in stock"
    # and "out of stock" labels in the DOM (CSS toggles which shows), so the words
    # lie. The reliable tells are id-keyed: a "#notify/<id>" link ("notify me when
    # back in stock") means SOLD OUT; an "item-<id>" purchase block without a notify
    # link means IN STOCK; neither → unknown.
    sold_out = set(re.findall(r"#notify/(\d+)", html))
    buyable = set(re.findall(r'id="item-(\d+)"', html))
    first, order = {}, []                       # release_id → (pos, artist, title)
    for m in _HW_ANCHOR.finditer(html):
        rid = m.group(1)
        if rid not in first:
            first[rid] = (m.start(), m.group(2), m.group(3))
            order.append(rid)
    out = []
    for i, rid in enumerate(order):
        pos, a_slug, t_slug = first[rid]
        end = first[order[i + 1]][0] if i + 1 < len(order) else len(html)
        block = html[pos:end]
        in_stock = (False if rid in sold_out
                    else True if rid in buyable else None)
        prices = [float(p.replace(",", ".")) for p in _HW_PRICE.findall(block)]
        price = max(prices) if prices else None
        fmt = _HW_FORMAT.search(webutil.clean_html(block))
        out.append({
            "store": "hardwax",
            "artist": _deslug(a_slug),
            "title": _deslug(t_slug),
            "label": None,
            "format": fmt.group(1) if fmt else None,
            "price": price,
            "currency": "EUR" if price is not None else None,
            "in_stock": in_stock,
            "url": f"https://hardwax.com/{rid}/{a_slug}/{t_slug}/",
        })
        if len(out) >= limit:
            break
    return out


# ── stubs for shops that resist scraping (see the module docstring) ─────────────
def _unsupported(name: str):
    def adapter(query: str, limit: int) -> list:
        raise NotImplementedError(
            f"{name}: no scrapable in-stock endpoint yet — needs a bespoke adapter")
    return adapter


_ADAPTERS = {
    "hardwax": _hardwax,
    "deejay": _unsupported("deejay.de"),    # JS layout: title/price not tied to stock cell
    "decks": _unsupported("decks.de"),      # hard anti-bot block (HTTP 403)
    "hhv": _unsupported("hhv.de"),          # no scrapable search endpoint found
}

# Ordering: in-stock first, then unknown, then sold-out.
_STOCK_RANK = {True: 0, None: 1, False: 2}


def _marketplace(query: str) -> "list | None":
    """The Discogs second-hand marketplace for `query` — up to 5 of the artist's records
    for sale, most community-valued first. Best-effort and never raises, so it can be
    submitted to the pool and overlapped with the shop adapters (Discogs throttles with
    429 backoff, so its latency is exactly what we want to hide behind the shop calls)."""
    try:
        import discogs
        return discogs.artist_for_sale(query, limit=5) or None
    except Exception as e:
        _log_failure("discogs-marketplace", e)
        return None


def search_stock(query: str, stores: "list | None" = None, limit: int = 8,
                 marketplace: bool = True) -> dict:
    """Search the configured record shops for `query` and return what is buyable
    now. Shops run CONCURRENTLY; any that fails is logged and skipped (the tool
    never raises). Results are de-duplicated and ordered in-stock → unknown →
    sold-out. When `marketplace` is on, the first-hand shop stock is COMPLEMENTED
    with up to 5 of the artist's records for sale SECOND-HAND on the Discogs
    marketplace (most community-valued first) under a separate "marketplace" key.

    Returns {"query", "stores": [shops used], "results": [ {store, artist, title,
    label, format, price, currency, in_stock, url}, … ], "marketplace"?: [ {title,
    year, rating, num_for_sale, lowest_price, url}, … ], "note"?, "unavailable_shops"?}.
    """
    query = (query or "").strip()
    if not query:
        return {"query": query, "stores": [], "results": [],
                "note": "empty query — nothing to search."}
    names = [s for s in (stores or config.VINYL_STORES) if s in _ADAPTERS]
    if not names:
        return {"query": query, "stores": [], "results": [],
                "note": "no known shops selected."}
    per = max(2, min(limit, 20))
    results, used, failed, marketplace_rows = [], [], [], None
    with _cf.ThreadPoolExecutor(max_workers=len(names) + 1) as ex:
        futs = {ex.submit(_ADAPTERS[n], query, per): n for n in names}
        # The Discogs marketplace lookup is independent of the shops, so run it in the SAME
        # pool concurrently and read it after — its (throttled) latency overlaps the shop
        # calls instead of adding on top.
        mk_fut = ex.submit(_marketplace, query) if marketplace else None
        for fut in _cf.as_completed(futs):
            n = futs[fut]
            try:
                results.extend(fut.result() or [])
                used.append(n)
            except Exception as e:
                _log_failure(n, e)
                failed.append(n)
        if mk_fut:
            marketplace_rows = mk_fut.result()
    seen, deduped = set(), []                   # de-dupe by (store, url)
    for r in results:
        k = (r.get("store"), r.get("url"))
        if k in seen:
            continue
        seen.add(k)
        deduped.append(r)
    deduped.sort(key=lambda r: _STOCK_RANK.get(r.get("in_stock"), 1))
    out = {"query": query, "stores": used, "results": deduped[:limit]}
    # The Discogs marketplace (second-hand, fetched concurrently above) is kept SEPARATE so
    # the agent presents it as a complement, not as first-hand shop stock.
    if marketplace_rows:
        out["marketplace"] = marketplace_rows
    if not deduped and not out.get("marketplace"):
        out["note"] = ("nothing came back (shops may be unreachable). Tell the user "
                       "you found no stock; do not invent records or availability.")
    if failed:
        out["unavailable_shops"] = failed
    return out
