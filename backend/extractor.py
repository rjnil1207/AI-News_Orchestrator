"""
extractor.py - Hybrid extractor (Article-by-Article for short pieces, Batch for long)
- Hybrid mode chosen: per-article for short texts (<=2000 chars), batch for long ones
- No embeddings
- Stronger LLM prompt requesting multiple chronological events
- Lightweight rule-based aggressive fallback if LLM returns too few events
- Uses GROQ (langchain_openai.ChatOpenAI). Falls back to rule-based if API key missing.
"""

import os
import time
import json
import logging
import re
from typing import List, Dict
from datetime import datetime, timezone
from dateutil import parser as dateparser
from tqdm import tqdm
import hashlib
import pickle
from collections import defaultdict

logging.basicConfig(level=logging.INFO)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
DEFAULT_MODEL = "llama-3.1-8b-instant"

MAX_RETRIES = 3
BASE_DELAY = 1

# Extraction cache directory
CACHE_DIR = os.path.join('.cache', 'extractor')


def _ensure_cache_dir():
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
    except Exception:
        pass


def _make_cache_key(article: Dict, model: str):
    # Use URL + published_iso + title + short-content-hash + model to make stable key
    url = (article.get('url') or '')
    title = (article.get('title') or '')
    pub = (article.get('published_iso') or '')
    content_snip = (article.get('content') or '')[:5000]
    h = hashlib.sha256((content_snip).encode('utf-8', errors='ignore')).hexdigest()
    s = f"{url}|{title}|{pub}|{h}|{model}"
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def _load_cache(key: str):
    _ensure_cache_dir()
    path = os.path.join(CACHE_DIR, f"{key}.pkl")
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        try:
            os.remove(path)
        except Exception:
            pass
        return None


def _save_cache(key: str, value):
    _ensure_cache_dir()
    path = os.path.join(CACHE_DIR, f"{key}.pkl")
    try:
        with open(path, 'wb') as f:
            pickle.dump(value, f)
    except Exception:
        pass


def chunk_text(text: str, max_chars: int = 20000) -> List[str]:
    if not text:
        return []
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end < len(text):
            split_at = text.rfind('.', start, end)
            if split_at <= start:
                split_at = end
            else:
                split_at += 1
            chunks.append(text[start:split_at].strip())
            start = split_at
        else:
            chunks.append(text[start:].strip())
            break
    return chunks


def call_llm(messages, model=DEFAULT_MODEL, max_tokens=1500, retry_count=0):
    """Call LLM (Groq via langchain_openai.ChatOpenAI) with basic retry handling."""
    try:
        from langchain_openai import ChatOpenAI
    except Exception as e:
        raise RuntimeError("langchain_openai not installed or cannot be imported.") from e

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise RuntimeError("GROQ_API_KEY not found in environment.")

    llm = ChatOpenAI(
        model=model,
        api_key=groq_key,
        base_url="https://api.groq.com/openai/v1",
        temperature=0.0,
        max_tokens=max_tokens,
    )

    try:
        resp = llm.invoke(messages)
        # resp could be an object with .content
        if hasattr(resp, "content"):
            return resp.content
        return str(resp)
    except Exception as e:
        err = str(e).lower()
        if ('429' in err or 'rate limit' in err or 'too many requests' in err) and retry_count < MAX_RETRIES:
            wait = BASE_DELAY * (2 ** retry_count)
            logging.warning(f"Rate limited, sleeping {wait}s then retrying...")
            time.sleep(wait)
            return call_llm(messages, model=model, max_tokens=max_tokens, retry_count=retry_count+1)
        if retry_count < MAX_RETRIES:
            time.sleep(BASE_DELAY * (2 ** retry_count))
            return call_llm(messages, model=model, max_tokens=max_tokens, retry_count=retry_count+1)
        raise


PROMPT_HEADER = (
    "You are an assistant that extracts chronological milestones from news article text.\n"
    "Return ONLY a JSON array of objects exactly like:\n"
    '[{"date":"YYYY-MM-DD or Unknown","event":"one-line summary","sources":["URL or source names"],"confidence":0.0}, ...]\n'
    "Requirements:\n"
    "- Extract up to 8 chronological events present in the article. Prefer concrete dates; if not present, infer using the article published date and context. Use 'Unknown' when date cannot be determined.\n"
    "- Each event must be a short (<=200 chars) single-line description.\n"
    "- Provide a numeric confidence between 0.0 and 1.0.\n"
    "- Do NOT hallucinate facts; extract only what is present or strongly implied.\n"
    "- If the article is brief, still try to extract at least 2 events (inference allowed), but mark confidence lower.\n"
    "Important: output must be valid JSON array only. No extra commentary."
)


def build_prompt_for_article(title: str, url: str, text: str, published_iso: str = None) -> List[Dict]:
    system = {"role": "system", "content": PROMPT_HEADER}
    user_content = (
        f"Article title: {title}\n"
        f"Article URL: {url}\n"
        f"Published: {published_iso}\n\n"
        f"Article text:\n{text}\n\n"
        f"Extract chronological events from the article and return JSON array as specified."
    )
    user = {"role": "user", "content": user_content}
    return [system, user]


def build_prompt_for_batch(batch_text: str, id_map: Dict[int, Dict]) -> List[Dict]:
    system = {"role": "system", "content": PROMPT_HEADER.replace("article", "these articles")}
    user_content = (
        "You will be given multiple articles labeled ARTICLE <n>.\n"
        "Return a JSON array of events. For the 'sources' field, use article numbers (e.g., [1,3]).\n\n"
        "CONTENT:\n\n"
        f"{batch_text}\n\n"
        "Important: output must be valid JSON array only."
    )
    user = {"role": "user", "content": user_content}
    return [system, user]


# Simple rule-based extractor for fallback (English)
EVENT_VERBS = [
    "launched", "announced", "landed", "lifted", "took off", "arrived", "signed",
    "declared", "died", "arrested", "claimed", "won", "lost", "filed", "released",
    "achieved", "failed", "completed", "confirmed", "revealed", "reported", "attacked",
    "suffered", "rescued", "collapsed", "approved", "denied"
]


def simple_rule_extract(article: Dict) -> List[Dict]:
    text = (article.get("content") or "").replace("\n", " ")
    sentences = re.split(r'(?<=[.!?])\s+', text)
    events = []
    for s in sentences:
        low = s.lower()
        if any(verb in low for verb in EVENT_VERBS):
            date = article.get("published_iso") or "Unknown"
            events.append({
                "date": date,
                "event": s.strip()[:200],
                "sources": [article.get("url") or article.get("source")],
                "confidence": 0.45
            })
    return events


def aggressive_fallback_from_articles(articles: List[Dict], min_events=3) -> List[Dict]:
    """Combine headlines and first sentences to create minimal events when LLM yields too few."""
    output = []
    for a in articles:
        title = a.get("title", "")
        date = a.get("published_iso") or "Unknown"
        first = ""
        content = a.get("content") or ""
        if content:
            parts = re.split(r'(?<=[.!?])\s+', content.strip())
            first = parts[0] if parts else ""
        line = title if len(title) > 10 else first[:180]
        if line:
            output.append({
                "date": date,
                "event": line.strip()[:200],
                "sources": [a.get("url") or a.get("source")],
                "confidence": 0.35
            })
        if len(output) >= min_events:
            break
    return output


def parse_json_array_from_text(text: str):
    """Attempt to find the first JSON array in the model response and parse it."""
    try:
        start = text.find('[')
        end = text.rfind(']')
        if start == -1 or end == -1 or end <= start:
            return None
        jtxt = text[start:end+1]
        parsed = json.loads(jtxt)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        return None
    return None


def normalize_date_str(d):
    if not d:
        return None
    try:
        return dateparser.parse(d).date().isoformat()
    except:
        return None


def normalize_event_obj(ev, default_source=None):
    date = ev.get("date") or ev.get("published") or None
    date = normalize_date_str(date) or (default_source.get("published_iso") if isinstance(default_source, dict) else None) or "Unknown"
    event_text = (ev.get("event") or ev.get("description") or "")[:600]
    sources = ev.get("sources") or ( [default_source.get("url")] if isinstance(default_source, dict) and default_source.get("url") else [])
    confidence = float(ev.get("confidence") or 0.5)
    return {"date": date, "event": event_text, "sources": sources, "confidence": confidence}


def merge_and_dedupe_events(events: List[Dict], min_len=20) -> List[Dict]:
    """Simple dedupe: keep unique by (date, normalized event start). Uses fuzzy but fast approach."""
    out = []
    seen = set()
    for e in events:
        text = (e.get("event") or "").strip()
        if not text or len(text) < min_len:
            continue
        key = (e.get("date") or "Unknown", text[:140].lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def extract_events(articles: List[Dict], model: str = DEFAULT_MODEL) -> List[Dict]:
    """
    Hybrid extractor:
    - Short articles (<=2000 chars) -> per-article LLM call
    - Longer / many articles -> batched calls (grouped by size)
    - If no GROQ key -> fallback to simple rule-based extraction across articles
    - Ensures a reasonable number of events by applying aggressive_fallback if LLM returns too few
    """
    if not articles:
        return []

    # Quick English-only filter: prefer articles where content or title looks like english (basic heuristic)
    def looks_english(s):
        if not s:
            return True
        # presence of many English stopwords as heuristic
        common = [" the ", " and ", " of ", " to ", " in "]
        low = s.lower()
        score = sum(1 for c in common if c in low)
        return score >= 1

    articles = [a for a in articles if looks_english(a.get("title", "") + " " + (a.get("content") or ""))]

    # If no GROQ key, do rule-based extraction for all and return
    if not os.getenv("GROQ_API_KEY"):
        logging.info("GROQ_API_KEY missing; using rule-based extraction for all articles.")
        events = []
        for a in articles:
            events.extend(simple_rule_extract(a))
        events = merge_and_dedupe_events(events)
        if len(events) < 3:
            events.extend(aggressive_fallback_from_articles(articles, min_events=3))
        return events

    short_articles = []
    long_articles = []
    for a in articles:
        text = (a.get("title", "") + "\n" + (a.get("content") or "")).strip()
        if len(text) <= 2000:
            short_articles.append((a, text))
        else:
            long_articles.append((a, text[:15000]))  # cap long text

    all_events = []

    # --- Per-article LLM for short pieces (higher recall) ---
    for a, text in tqdm(short_articles, desc="Per-article extraction"):
        # try per-article cache first
        try:
            key = _make_cache_key(a, model)
            cached = _load_cache(key)
            if cached:
                all_events.extend(cached)
                continue
        except Exception:
            key = None

        messages = build_prompt_for_article(a.get("title", ""), a.get("url", a.get("source")), text, a.get("published_iso"))
        try:
            resp = call_llm(messages, model=model, max_tokens=1000)
            parsed = parse_json_array_from_text(resp or "")
            article_parsed_events = []
            if parsed:
                for ev in parsed:
                    eobj = normalize_event_obj(ev, default_source=a)
                    # ensure sources is list of strings
                    eobj["sources"] = eobj.get("sources") or [a.get("url") or a.get("source")]
                    all_events.append(eobj)
                    article_parsed_events.append(eobj)
            else:
                # fallback to rule-based for this article
                article_parsed_events = simple_rule_extract(a)
                all_events.extend(article_parsed_events)

            # save per-article cache
            try:
                if key and article_parsed_events:
                    _save_cache(key, article_parsed_events)
            except Exception:
                pass

        except Exception as e:
            logging.warning(f"Per-article LLM failed for {a.get('url')}: {e}")
            fallback = simple_rule_extract(a)
            all_events.extend(fallback)
            try:
                if key and fallback:
                    _save_cache(key, fallback)
            except Exception:
                pass

    # --- Batch mode for long articles (grouped to keep calls small) ---
    # group long_articles into batches of approx MAX_CHARS
    MAX_BATCH_CHARS = 18000
    batch = []
    batch_map = {}
    batch_chars = 0
    batch_id = 1
    # filter long_articles by per-article cache: collect those missing cached results
    remaining_long_articles = []
    for a, text in long_articles:
        try:
            k = _make_cache_key(a, model)
            c = _load_cache(k)
            if c:
                all_events.extend(c)
            else:
                remaining_long_articles.append((a, text))
        except Exception:
            remaining_long_articles.append((a, text))

    for idx, (a, text) in enumerate(remaining_long_articles, start=1):
        label = f"ARTICLE {batch_id}_{idx}"
        batch.append((label, a, text))
        batch_map[label] = a
        batch_chars += len(text)
        if batch_chars >= MAX_BATCH_CHARS:
            # flush batch
            batch_text = "\n\n".join([f"### {lbl}\n{text}" for (lbl, a, text) in batch])
            messages = build_prompt_for_batch(batch_text, {lbl: art for (lbl, art, _) in batch})
            try:
                resp = call_llm(messages, model=model, max_tokens=1500)
                parsed = parse_json_array_from_text(resp or "")
                if parsed:
                    # We'll collect per-article events to cache after mapping
                    article_url_to_events = defaultdict(list)
                    for ev in parsed:
                        sources = ev.get("sources") or []
                        mapped_sources = []
                        for s in sources:
                            try:
                                if isinstance(s, int):
                                    if 1 <= s <= len(batch):
                                        mapped_sources.append(batch[s-1][1].get("url") or batch[s-1][1].get("source"))
                                elif isinstance(s, str):
                                    found = False
                                    for lbl, art, _ in batch:
                                        if s.strip().lower().endswith(str(lbl).lower()) or s.strip().lower() in lbl.lower():
                                            mapped_sources.append(art.get("url") or art.get("source"))
                                            found = True
                                    if not found:
                                        mapped_sources.append(s)
                            except:
                                continue

                        eobj = normalize_event_obj(ev, default_source=batch[0][1] if batch else None)
                        eobj["sources"] = mapped_sources or eobj.get("sources") or []
                        all_events.append(eobj)

                        # record for caching per-article when mapped_sources include a batch article url
                        for url in mapped_sources:
                            # match url to batch article objects
                            for lbl, art, _ in batch:
                                art_url = art.get('url') or art.get('source')
                                if art_url and str(art_url) == str(url):
                                    article_url_to_events[art_url].append(eobj)

                    # save per-article caches for any events found
                    for lbl, art, _ in batch:
                        art_url = art.get('url') or art.get('source')
                        if not art_url:
                            continue
                        try:
                            k = _make_cache_key(art, model)
                            evs = article_url_to_events.get(art_url) or []
                            if evs:
                                _save_cache(k, evs)
                        except Exception:
                            pass
                else:
                    # no parse -> try rule-based on each article in batch
                    for lbl, art, txt in batch:
                        all_events.extend(simple_rule_extract(art))
            except Exception as e:
                logging.warning(f"Batch LLM failed: {e}")
                for lbl, art, txt in batch:
                    all_events.extend(simple_rule_extract(art))
            # reset
            batch = []
            batch_map = {}
            batch_chars = 0
            batch_id += 1

    # flush remaining batch if any
    if batch:
        batch_text = "\n\n".join([f"### {lbl}\n{text}" for (lbl, a, text) in batch])
        messages = build_prompt_for_batch(batch_text, {lbl: art for (lbl, art, _) in batch})
        try:
            resp = call_llm(messages, model=model, max_tokens=1500)
            parsed = parse_json_array_from_text(resp or "")
            if parsed:
                # Similar mapping + caching logic as above
                article_url_to_events = defaultdict(list)
                for ev in parsed:
                    sources = ev.get("sources") or []
                    mapped_sources = []
                    for s in sources:
                        if isinstance(s, int) and 1 <= s <= len(batch):
                            mapped_sources.append(batch[s-1][1].get("url") or batch[s-1][1].get("source"))
                        else:
                            mapped_sources.append(s)
                    eobj = normalize_event_obj(ev, default_source=batch[0][1] if batch else None)
                    eobj["sources"] = mapped_sources or eobj.get("sources") or []
                    all_events.append(eobj)
                    for url in mapped_sources:
                        for lbl, art, _ in batch:
                            art_url = art.get('url') or art.get('source')
                            if art_url and str(art_url) == str(url):
                                article_url_to_events[art_url].append(eobj)

                for lbl, art, _ in batch:
                    art_url = art.get('url') or art.get('source')
                    if not art_url:
                        continue
                    try:
                        k = _make_cache_key(art, model)
                        evs = article_url_to_events.get(art_url) or []
                        if evs:
                            _save_cache(k, evs)
                    except Exception:
                        pass
            else:
                for lbl, art, txt in batch:
                    all_events.extend(simple_rule_extract(art))
        except Exception as e:
            logging.warning(f"Final batch LLM failed: {e}")
            for lbl, art, txt in batch:
                all_events.extend(simple_rule_extract(art))

    # ---- Postprocess: merge, dedupe, sort ----
    merged = merge_and_dedupe_events(all_events, min_len=12)

    # If still too few events, apply aggressive fallback using headlines and first sentences
    if len(merged) < 3:
        merged.extend(aggressive_fallback_from_articles(articles, min_events=3))
        merged = merge_and_dedupe_events(merged, min_len=12)

    # Normalize 'sources' to be unique list strings
    for e in merged:
        s = e.get("sources") or []
        e["sources"] = list({str(x) for x in s if x})

    # ensure confidence in [0,1]
    for e in merged:
        try:
            c = float(e.get("confidence", 0.5))
            e["confidence"] = min(max(c, 0.0), 1.0)
        except:
            e["confidence"] = 0.5

    # sort by date where possible (Unknown goes last)
    def sort_key(x):
        try:
            d = x.get("date")
            if not d or d == "Unknown":
                return datetime.max.replace(tzinfo=timezone.utc)
            # try to parse and normalize to UTC-aware datetime
            try:
                dt = datetime.fromisoformat(d)
            except Exception:
                try:
                    dt = dateparser.parse(d)
                except Exception:
                    return datetime.max.replace(tzinfo=timezone.utc)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt
        except:
            return datetime.max.replace(tzinfo=timezone.utc)

    merged = sorted(merged, key=sort_key)
    return merged


# CLI convenience
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", help="path to articles json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    if not args.json:
        print("Provide --json path")
        raise SystemExit(1)
    with open(args.json, 'r', encoding='utf-8') as f:
        arts = json.load(f)
    evts = extract_events(arts, model=args.model)
    print(f"Extracted {len(evts)} events")
    with open("extracted_events.json", "w", encoding='utf-8') as f:
        json.dump(evts, f, indent=2, ensure_ascii=False)
