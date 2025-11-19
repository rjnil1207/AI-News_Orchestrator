import os
import sys
import json
import sqlite3
import logging
import requests
import feedparser
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
from dateutil import parser as dateparser
from datetime import datetime
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import concurrent.futures

logging.basicConfig(level=logging.INFO)

DB_PATH = "news_cache.db"


# -----------------------------
# DB
# -----------------------------
def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            url TEXT PRIMARY KEY,
            title TEXT,
            published_iso TEXT,
            content TEXT,
            source TEXT,
            fetched_at TEXT
        )
    """)
    conn.commit()
    return conn


def cache_article(conn, article):
    cur = conn.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO articles (url, title, published_iso, content, source, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        article.get("url"),
        article.get("title"),
        article.get("published_iso"),
        article.get("content"),
        article.get("source"),
        datetime.utcnow().isoformat()
    ))
    conn.commit()


def get_cached_articles(conn, query=None, limit=10):
    cur = conn.cursor()
    if query:
        q = f"%{query}%"
        cur.execute("""
            SELECT url, title, published_iso, content, source FROM articles
            WHERE title LIKE ? OR content LIKE ?
            ORDER BY published_iso DESC LIMIT ?
        """, (q, q, limit))
    else:
        cur.execute("""
            SELECT url, title, published_iso, content, source FROM articles
            ORDER BY fetched_at DESC LIMIT ?
        """, (limit,))
    rows = cur.fetchall()
    return [
        {"url": r[0], "title": r[1], "published_iso": r[2], "content": r[3], "source": r[4]}
        for r in rows
    ]


# -----------------------------
# Helpers
# -----------------------------
def normalize_date(x):
    try:
        if not x:
            return None
        dt = dateparser.parse(x)
        return dt.isoformat() if dt else None
    except:
        return None


def clean_html(text):
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    for bad in soup(["script", "style", "noscript"]):
        bad.decompose()
    txt = soup.get_text(" ", strip=True)
    return " ".join(txt.split())


def is_english(text: str) -> bool:
    """Very light heuristic — avoids heavy libraries."""
    if not text:
        return False
    english_chars = sum(c.isascii() for c in text)
    return english_chars / max(len(text), 1) >= 0.85


def fetch_article_text(url):
    # Use a pooled session with retries where possible for better performance
    try:
        session = get_session()
        if not url:
            return ""
        resp = session.get(url, timeout=6)
        if resp.status_code != 200:
            return ""
        text = clean_html(resp.text)
        # cap size to keep memory usage bounded
        return text[:12000]
    except Exception:
        return ""


_SESSION = None

def get_session():
    """Return a configured requests.Session with retry/backoff and common headers."""
    global _SESSION
    if _SESSION is not None:
        return _SESSION

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; AI-News-Orchestrator/1.0)"})

    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    _SESSION = session
    return _SESSION


# -----------------------------
# Google News RSS (English only)
# -----------------------------
def fetch_from_google_rss(query, top_n=10):
    q = quote_plus(query)
    rss_url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"

    feed = feedparser.parse(rss_url)
    articles = []

    for e in feed.entries[:top_n]:
        title = getattr(e, "title", "")
        if not is_english(title):  # ENGLISH CHECK
            continue

        published_iso = normalize_date(
            getattr(e, "published", None) or getattr(e, "updated", None)
        )
        content = clean_html(getattr(e, "summary", "") or getattr(e, "description", ""))

        articles.append({
            "title": title,
            "published_iso": published_iso,
            "content": content,
            "url": getattr(e, "link", None),
            "source": "GoogleNewsRSS"
        })

    return articles


# -----------------------------
# GDELT (Global – filter non-English)
# -----------------------------
def fetch_from_gdelt(query, max_results=10):
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {"query": query, "mode": "artlist", "format": "json"}

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        out = []
        for a in data.get("articles", [])[:max_results]:
            title = a.get("title") or ""
            if not is_english(title):  # ENGLISH CHECK
                continue

            out.append({
                "title": title,
                "published_iso": normalize_date(a.get("seendate")),
                "content": "",
                "url": a.get("url"),
                "source": a.get("domain") or "GDELT"
            })

        return out

    except:
        return []


# -----------------------------
# NewsAPI (English forced)
# -----------------------------
def fetch_from_newsapi(query, max_results=10):
    api_key = os.environ.get("NEWSAPI_KEY")
    if not api_key:
        logging.warning("NEWSAPI_KEY not found. Skipping NewsAPI source.")
        return []

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "apiKey": api_key,
        "language": "en",   # FORCED ENGLISH ✔
        "sortBy": "publishedAt",
        "pageSize": max_results
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "ok":
            return []

        out = []
        for a in data.get("articles", [])[:max_results]:
            out.append({
                "title": a.get("title"),
                "published_iso": normalize_date(a.get("publishedAt")),
                "content": a.get("description") or a.get("content", ""),
                "url": a.get("url"),
                "source": a.get("source", {}).get("name") or "NewsAPI"
            })
        return out

    except:
        return []


# -----------------------------
# Trending News – ensures variation
# -----------------------------
def fetch_top_headlines(max_results=4):
    """
    Always return different top headlines.
    Light. Fresh. English only.
    """
    rss_url = (
        "https://feeds.bbci.co.uk/news/world/rss.xml"
    )

    feed = feedparser.parse(rss_url)

    items = []
    for e in feed.entries:
        title = getattr(e, "title", "")
        if not is_english(title):
            continue

        items.append({
            "title": title,
            "url": getattr(e, "link", ""),
            "source": "Google News",
            "published_iso": getattr(e, "published", "") or getattr(e, "updated", "")
        })

    if len(items) <= max_results:
        return items

    import random
    random.shuffle(items)
    return items[:max_results]


# -----------------------------
# Aggregator
# -----------------------------
def dedupe(articles):
    seen, out = set(), []
    for a in articles:
        key = a.get("url") or a.get("title")
        if key and key not in seen:
            seen.add(key)
            out.append(a)
    return out


def aggregate(query, top_n=10, sources=("gdelt", "gnews", "newsapi"), use_cache=True):
    conn = init_db()

    if use_cache:
        cached = get_cached_articles(conn, query, top_n)
        if cached and len(cached) >= top_n:
            return cached

    results = []

    if "gdelt" in sources:
        results.extend(fetch_from_gdelt(query, top_n))

    if "gnews" in sources or "rss" in sources:
        results.extend(fetch_from_google_rss(query, top_n))

    if "newsapi" in sources:
        results.extend(fetch_from_newsapi(query, top_n))

    # Scrape missing content in parallel using a small threadpool to improve throughput
    to_fetch = [a for a in results if not a.get("content") and a.get("url")]
    session = get_session()
    if to_fetch:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            future_to_article = {ex.submit(fetch_article_text, a.get("url")): a for a in to_fetch}
            for fut in tqdm(concurrent.futures.as_completed(future_to_article), total=len(future_to_article), desc="Scraping full text"):
                art = future_to_article[fut]
                try:
                    art["content"] = fut.result()
                except Exception:
                    art["content"] = ""
                cache_article(conn, art)
    else:
        # Nothing to fetch, but still ensure we cache existing entries
        for art in results:
            cache_article(conn, art)

    cleaned = dedupe(results)
    cleaned = sorted(cleaned, key=lambda a: normalize_date(a.get("published_iso")) or "", reverse=True)

    return cleaned[:top_n]
