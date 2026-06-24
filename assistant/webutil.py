"""
assistant/webutil.py — tiny shared helpers for the live-web adapters.

web_sources.py (Resident Advisor + registered-site search) and vinyl_stores.py (record
shops) stay ISOLATED per service — neither imports the other's HTTP client or knows its
wire format. But four generic, service-agnostic things were copy-pasted between them: the
browser-like User-Agent several sites demand, an HTML→text cleaner, a JSON failure log
(for "the source changed / blocked us" diagnostics), and the DuckDuckGo scoped search both
use as a fallback. They carry no service knowledge, so they live here once.
"""
import datetime
import json
import re
from html import unescape
from urllib.parse import unquote, urlparse

import httpx

# A browser-like UA — RA sits behind Cloudflare and several record shops 403 a default
# client, so every adapter must look like a real browser.
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

_TAG_RE = re.compile(r"<[^>]+>")


def clean_html(html: str, tag_repl: str = " ", unescape_entities: bool = True) -> str:
    """Strip tags and collapse whitespace into a one-line snippet. `tag_repl` is what each
    tag becomes (" " keeps word boundaries, "" joins them); `unescape_entities` turns
    &amp; etc. back into text. Defaults match the record-shop parser; the RA/DDG parser
    passes tag_repl="" / unescape_entities=False to keep its original behaviour."""
    text = re.sub(r"\s+", " ", _TAG_RE.sub(tag_repl, html or "")).strip()
    return unescape(text) if unescape_entities else text


def log_failure(logfile, logger, source: str, stage: str, detail) -> None:
    """Append one JSON line recording an adapter failure (so a later fix can see which
    source went stale) and warn the logger. Best-effort — never raises."""
    logger.warning("%s: %s failed — %s", source, stage, str(detail)[:200])
    try:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        with logfile.open("a") as f:
            f.write(json.dumps({"ts": datetime.datetime.now().isoformat(timespec="seconds"),
                                "stage": stage, "detail": str(detail)[:300]}) + "\n")
    except Exception:
        pass


# ── DuckDuckGo HTML search, scoped to site: domains (the agent's web fallback) ───
_DDG = "https://html.duckduckgo.com/html/"
_RESULT_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.S)
_SNIPPET_RE = re.compile(r'class="result__snippet"[^>]*>(.*?)</a>', re.S)


def _real_url(href: str) -> str:
    """DDG wraps result links as //duckduckgo.com/l/?uddg=<encoded> — unwrap it."""
    m = re.search(r"uddg=([^&]+)", href or "")
    return unquote(m.group(1)) if m else href


def ddg_search(query: str, domains: list, limit: int, timeout: float) -> list:
    """Plain DuckDuckGo HTML search scoped to `domains` via site: filters. Returns on-domain
    {title, url, snippet} dicts (ads / off-domain hits dropped). Raises on transport errors
    so the caller can fall back. Shared by web_sources' RA fallback and the registered-site
    search. `domains` must be non-empty (callers pass a sensible default)."""
    domains = [d.lower() for d in (domains or []) if d]
    scope = " OR ".join(f"site:{d}" for d in domains)
    r = httpx.post(_DDG, headers={"User-Agent": UA}, timeout=timeout,
                   data={"q": f"{query} {scope}".strip()})
    r.raise_for_status()
    html = r.text
    out = []
    for m in _RESULT_RE.finditer(html):
        url = _real_url(m.group(1))
        host = urlparse(url).netloc.lower()
        if domains and not any(d in host for d in domains):    # drop ads / off-domain hits
            continue
        sm = _SNIPPET_RE.search(html, m.end(), m.end() + 1500)
        out.append({
            "title": clean_html(m.group(2), tag_repl="", unescape_entities=False),
            "url": url,
            "snippet": (clean_html(sm.group(1), tag_repl="", unescape_entities=False)
                        if sm else ""),
        })
        if len(out) >= limit:
            break
    return out
