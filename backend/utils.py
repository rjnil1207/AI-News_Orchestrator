"""
utils.py - Single helper file with all improvements
Place this in the same directory as app.py
"""

import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
import pandas as pd
from urllib.parse import urlparse
import logging
from functools import wraps
import time
import hashlib
import pickle
import os
import json

# ============================================
# AUTHENTICITY CHECKING
# ============================================

TRUSTED_SOURCES = {
    "tier1": ["reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "npr.org"],
    "tier2": ["cnn.com", "nytimes.com", "theguardian.com", "washingtonpost.com", 
              "wsj.com", "ft.com", "bloomberg.com"],
    "tier3": ["cnbc.com", "axios.com", "politico.com", "thehill.com", "forbes.com"]
}

def get_source_tier(url):
    """Categorize source by reliability tier"""
    if not url:
        return 0
    try:
        s = str(url).strip()
        s_lower = s.lower()
        # If this looks like a URL, parse domain
        if s_lower.startswith('http://') or s_lower.startswith('https://') or '/' in s_lower:
            domain = urlparse(s).netloc.lower().replace("www.", "")
        else:
            # treat as plain source name, use it directly
            domain = s_lower

        for tier, domains in TRUSTED_SOURCES.items():
            for d in domains:
                if d in domain or domain in d:
                    return 3 if tier == "tier1" else 2 if tier == "tier2" else 1
    except Exception:
        return 0
    return 0

def calculate_source_diversity_score(timeline):
    """Higher score if events come from diverse sources"""
    if not timeline:
        return 0
    all_sources = set()
    tiers_present = set()
    for event in timeline:
        for s in event.get("sources", []):
            try:
                if isinstance(s, dict):
                    candidate = s.get('url') or s.get('source') or str(s)
                else:
                    candidate = str(s)
                candidate = candidate.strip()
                if candidate:
                    all_sources.add(candidate)
                    try:
                        tiers_present.add(get_source_tier(candidate))
                    except Exception:
                        pass
            except Exception:
                continue
    unique_sources = len(all_sources)
    if unique_sources == 0:
        return 0

    # Base diversity scaled to an 8-source expectation (>=8 -> 100)
    base = min(1.0, unique_sources / 8.0)

    # Slight boost if multiple credibility tiers are present (indicates cross-spectrum reporting)
    tier_variety = len([t for t in tiers_present if t and t > 0])
    tier_boost = 0.0
    if tier_variety >= 2:
        tier_boost = 0.08
    if tier_variety >= 3:
        tier_boost = 0.15

    diversity_score = int(min(1.0, base + tier_boost) * 100)
    return diversity_score

def calculate_cross_verification_score(timeline):
    """Score based on multi-source events"""
    if not timeline:
        return 0

    # Helper to normalize a source entry (string or dict) to a canonical string
    def _norm_source(s):
        try:
            if isinstance(s, dict):
                return s.get('url') or s.get('source') or str(s)
            return str(s)
        except Exception:
            return str(s)

    # Cluster events by normalized text similarity to detect the same event reported by multiple sources
    clusters = []  # each cluster is list of event indices
    texts = [str(e.get('event','')).strip() for e in timeline]
    from difflib import SequenceMatcher

    # more tolerant clustering: compare against all members in a cluster
    SEQ_THRESHOLD = 0.60
    JACCARD_THRESHOLD = 0.28
    import re
    def _jaccard(a, b):
        sa = set(re.findall(r"\w+", a.lower()))
        sb = set(re.findall(r"\w+", b.lower()))
        if not sa or not sb:
            return 0.0
        return float(len(sa & sb)) / float(len(sa | sb))

    for idx, t in enumerate(texts):
        placed = False
        norm_t = t.lower()
        for c in clusters:
            if not c or not norm_t:
                continue
            # compare against each current member of the cluster for robustness
            for member_idx in c:
                rep = texts[member_idx].lower() if member_idx < len(texts) else ''
                if not rep:
                    continue
                sim_seq = SequenceMatcher(None, rep, norm_t).ratio()
                sim_jac = _jaccard(rep, norm_t)
                if sim_seq >= SEQ_THRESHOLD or sim_jac >= JACCARD_THRESHOLD:
                    c.append(idx)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            clusters.append([idx])

    # For each cluster, count unique sources and compute a verification strength
    cluster_strengths = []  # total sources (capped) per cluster
    verified_strength = 0
    for c in clusters:
        srcs = set()
        for i in c:
            for s in timeline[i].get('sources', []):
                ns = _norm_source(s)
                if ns:
                    srcs.add(ns)
        unique_count = len(srcs)
        if unique_count == 0:
            continue
        # cap influence of extremely large source lists to avoid dominance
        strength = min(unique_count, 6)
        cluster_strengths.append(strength)
        if unique_count >= 2:
            verified_strength += strength

    total_strength = sum(cluster_strengths)
    if total_strength == 0:
        return 0

    # Score is proportion of verification-weighted strength across clusters
    score = int((verified_strength / total_strength) * 100)
    return score

def calculate_temporal_consistency_score(timeline):
    """Check if dates are logical"""
    if not timeline:
        return 0
    dates = []
    for e in timeline:
        date_str = e.get("date")
        if date_str and date_str != "Unknown":
            try:
                # normalize to timezone-aware UTC datetimes for safe comparisons
                try:
                    dt = datetime.fromisoformat(date_str)
                except Exception:
                    from dateutil import parser as dateparser
                    dt = dateparser.parse(date_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)
                dates.append(dt)
            except:
                pass
    if len(dates) < 2:
        return 60

    # Compute monotonicity: fraction of ordered pairs (i<j) where date_i <= date_j
    n = len(dates)
    total_pairs = n * (n - 1) / 2
    if total_pairs <= 0:
        return 60
    ordered_pairs = 0
    for i in range(n):
        for j in range(i+1, n):
            if dates[i] <= dates[j]:
                ordered_pairs += 1

    monotonicity = ordered_pairs / total_pairs

    # Penalize future-dated items (likely errors) but more moderately
    now_utc = datetime.now(timezone.utc)
    future = sum(1 for d in dates if d > now_utc)
    future_prop = future / n

    score = int(max(0, min(1.0, monotonicity - future_prop * 0.35)) * 100)
    return score

def calculate_source_quality_score(timeline):
    """Score based on quality of sources"""
    if not timeline:
        return 0
    # Gather unique sources across timeline and compute average reputation weight
    unique_sources = set()
    for event in timeline:
        for s in event.get('sources', []):
            try:
                if isinstance(s, dict):
                    candidate = s.get('url') or s.get('source') or str(s)
                else:
                    candidate = str(s)
                candidate = candidate.strip()
                if candidate:
                    unique_sources.add(candidate)
            except Exception:
                continue

    if not unique_sources:
        return 30

    # Map tiers to reputation weights (0-1)
    def _tier_weight(t):
        if t == 3:
            return 0.95
        if t == 2:
            return 0.70
        if t == 1:
            return 0.45
        return 0.25

    weights = []
    for u in unique_sources:
        try:
            t = get_source_tier(u)
            weights.append(_tier_weight(t))
        except Exception:
            weights.append(0.25)

    avg_weight = sum(weights) / len(weights)
    return int(avg_weight * 100)

def detect_bias_indicators(timeline):
    """Detect potential bias in event descriptions"""
    bias_keywords = [
        "allegedly", "claimed", "reportedly", "sources say", "rumored",
        "shocking", "devastating", "incredible", "unbelievable"
    ]
    flags = []
    for event in timeline:
        text = event.get("event", "").lower()
        for keyword in bias_keywords:
            if keyword in text:
                flags.append({
                    "event": event.get("event"),
                    "flag": keyword,
                    "type": "speculative" if keyword in ["allegedly", "claimed"] else "sensational"
                })
    return flags

def compute_advanced_authenticity(timeline):
    """Multi-factor authenticity scoring"""
    if not timeline:
        return {
            "overall_score": 0,
            "breakdown": {},
            "explanation": "No timeline data available",
            "bias_flags": [],
            "grade": "F"
        }
    
    # Calculate more robust component scores
    # Confidence: weighted average by sqrt(number_of_sources + 1) so multi-sourced events count more
    total_w = 0.0
    weighted_conf = 0.0
    for e in timeline:
        conf = float(e.get('confidence', 0) or 0)
        num_srcs = 0
        try:
            num_srcs = len(e.get('sources') or [])
        except Exception:
            num_srcs = 0
        weight = 1.0 + (num_srcs ** 0.5)
        weighted_conf += conf * weight
        total_w += weight
    confidence_avg = (weighted_conf / total_w) if total_w > 0 else 0.0
    confidence_score = int(max(0, min(1.0, confidence_avg)) * 100)

    diversity_score = calculate_source_diversity_score(timeline)
    verification_score = calculate_cross_verification_score(timeline)
    temporal_score = calculate_temporal_consistency_score(timeline)
    quality_score = calculate_source_quality_score(timeline)

    # Ensure a sensible floor for source quality when timeline exists
    try:
        if quality_score is None or quality_score <= 0:
            quality_score = 30
    except Exception:
        quality_score = 30

    # Normalize and clamp all component scores to integers 0-100
    def _clamp_score(v):
        try:
            iv = int(v)
        except Exception:
            try:
                iv = int(float(v))
            except Exception:
                iv = 0
        return max(0, min(100, iv))

    confidence_score = _clamp_score(confidence_score)
    diversity_score = _clamp_score(diversity_score)
    verification_score = _clamp_score(verification_score)
    temporal_score = _clamp_score(temporal_score)
    quality_score = _clamp_score(quality_score)

    # Weighted combination
    overall = int(
        confidence_score * 0.25 +
        diversity_score * 0.20 +
        verification_score * 0.25 +
        temporal_score * 0.15 +
        quality_score * 0.15
    )
    
    bias_flags = detect_bias_indicators(timeline)
    if len(bias_flags) > len(timeline) * 0.3:
        overall = int(overall * 0.85)
    
    explanation = f"""Authenticity Score Breakdown:
• Extraction Confidence: {confidence_score}/100 - Average confidence from AI extraction
• Source Diversity: {diversity_score}/100 - Variety of news sources consulted
• Cross-Verification: {verification_score}/100 - Events confirmed by multiple sources
• Temporal Consistency: {temporal_score}/100 - Logical chronological ordering
• Source Quality: {quality_score}/100 - Reliability of cited sources

Overall assessment: {'HIGH' if overall >= 75 else 'MEDIUM' if overall >= 50 else 'LOW'} confidence"""
    
    if bias_flags:
        explanation += f"\n⚠️ {len(bias_flags)} potential bias indicators detected"
    
    grade = "A" if overall >= 85 else "B" if overall >= 70 else "C" if overall >= 55 else "D" if overall >= 40 else "F"
    
    return {
        "overall_score": overall,
        "breakdown": {
            "confidence": confidence_score,
            "source_diversity": diversity_score,
            "cross_verification": verification_score,
            "temporal_consistency": temporal_score,
            "source_quality": quality_score
        },
        "explanation": explanation.strip(),
        "bias_flags": bias_flags,
        "grade": grade
    }


# ============================================
# SUBJECTIVITY / CLICKBAIT DETECTION
# ============================================

SUBJECTIVE_WORDS = [
    "i think", "i believe", "we believe", "in my opinion", "should", "must",
    "probably", "possibly", "likely", "seems", "suggests", "feels"
]

CLICKBAIT_PATTERNS = [
    "you won't believe", "won't believe", "what happens next", "this is why",
    "shocking", "will blow your mind", "# reasons", "reasons why", "top [0-9]+",
    "best .* ever", "worst .* ever", "what they did next"
]

def detect_subjective_tone(text):
    """Return a subjectivity score (0-100) and any matched phrases."""
    if not text:
        return {"score": 0, "flags": []}
    t = text.lower()
    matches = [w for w in SUBJECTIVE_WORDS if w in t]
    score = min(100, int(len(matches) / max(1, len(SUBJECTIVE_WORDS)) * 100))
    return {"score": score, "flags": matches}

def detect_clickbait_headline(title):
    """Heuristic clickbait detection for headlines. Returns score and matched patterns."""
    if not title:
        return {"score": 0, "flags": []}
    t = title.lower()
    flags = []
    # simple checks
    if "?" in title and len(title.split()) < 12:
        flags.append("question-headline")
    for p in CLICKBAIT_PATTERNS:
        try:
            import re
            if re.search(p, t):
                flags.append(p)
        except Exception:
            continue
    score = min(100, int(len(flags) / max(1, len(CLICKBAIT_PATTERNS)) * 100))
    return {"score": score, "flags": flags}


# ============================================
# HISTORY / DAILY DASHBOARD STORAGE
# ============================================

HISTORY_DIR = os.path.join(".cache", "history")
os.makedirs(HISTORY_DIR, exist_ok=True)

def _safe_filename(s: str):
    import re
    fname = re.sub(r"[^0-9a-zA-Z_-]", "_", s)
    return fname[:200]

def save_timeline_history(result: dict, query: str = None):
    """Save a timeline result JSON to the history folder with timestamp and query metadata."""
    try:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        qpart = _safe_filename(query or "query")
        filename = f"{ts}__{qpart}.json"
        path = os.path.join(HISTORY_DIR, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({"saved_at": ts, "query": query, "result": result}, f, ensure_ascii=False, indent=2)
        return path
    except Exception:
        return None

def load_timeline_history(limit: int = 50):
    """Load recent saved timelines (most recent first)."""
    try:
        files = sorted([f for f in os.listdir(HISTORY_DIR) if f.endswith('.json')], reverse=True)
        loaded = []
        for f in files[:limit]:
            try:
                with open(os.path.join(HISTORY_DIR, f), 'r', encoding='utf-8') as fh:
                    loaded.append(json.load(fh))
            except Exception:
                continue
        return loaded
    except Exception:
        return []


# ============================================
# VISUALIZATIONS
# ============================================

def create_interactive_timeline(timeline_data):
    """Create a rich, interactive timeline with Plotly"""
    if not timeline_data:
        return None
    
    events = []
    for idx, item in enumerate(timeline_data):
        date_str = item.get('date', 'Unknown')
        if date_str and date_str != 'Unknown':
            try:
                try:
                    date = datetime.fromisoformat(date_str)
                except Exception:
                    from dateutil import parser as dateparser
                    date = dateparser.parse(date_str)
                if date.tzinfo is None:
                    date = date.replace(tzinfo=timezone.utc)
                else:
                    date = date.astimezone(timezone.utc)
            except:
                date = datetime.now(timezone.utc) - timedelta(days=len(timeline_data)-idx)
        else:
            date = datetime.now(timezone.utc) - timedelta(days=len(timeline_data)-idx)
        
        events.append({
            'date': date,
            'event': item.get('event', ''),
            'confidence': item.get('confidence', 0.5) * 100,
            'sources': len(item.get('sources', []))
        })
    
    df = pd.DataFrame(events)
    
    fig = go.Figure()
    
    colors = df['confidence'].apply(
        lambda x: '#10b981' if x >= 80 else '#f59e0b' if x >= 60 else '#ef4444'
    )
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['confidence'],
        mode='markers+lines',
        marker=dict(
            size=df['sources'] * 8 + 10,
            color=colors,
            line=dict(width=2, color='white'),
            opacity=0.8
        ),
        line=dict(color='rgba(102, 126, 234, 0.3)', width=2),
        text=df['event'],
        hovertemplate='<b>%{text}</b><br>Date: %{x|%Y-%m-%d}<br>Confidence: %{y:.0f}%<br><extra></extra>'
    ))
    
    fig.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="High Confidence", opacity=0.3)
    fig.add_hline(y=60, line_dash="dash", line_color="orange", annotation_text="Medium Confidence", opacity=0.3)
    
    fig.update_layout(
        title={'text': '📊 Event Timeline with Confidence Levels', 'x': 0.5, 'xanchor': 'center'},
        xaxis=dict(title='Date', gridcolor='rgba(200, 200, 200, 0.2)', showgrid=True),
        yaxis=dict(title='Confidence Score (%)', gridcolor='rgba(200, 200, 200, 0.2)', range=[0, 105]),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        height=500
    )
    
    return fig

def create_confidence_distribution(timeline_data):
    """Enhanced confidence distribution chart"""
    if not timeline_data:
        return None
    
    confidences = [item.get('confidence', 0.5) * 100 for item in timeline_data if item.get('confidence')]
    
    fig = go.Figure(data=[go.Histogram(
        x=confidences,
        nbinsx=10,
        marker_color='rgba(102, 126, 234, 0.7)',
        marker_line_color='rgba(102, 126, 234, 1)',
        marker_line_width=1.5
    )])
    
    fig.update_layout(
        title='Event Confidence Distribution',
        xaxis_title='Confidence Score (%)',
        yaxis_title='Number of Events',
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=400
    )
    
    return fig

# ============================================
# ERROR HANDLING
# ============================================

def retry_with_exponential_backoff(max_retries=3, base_delay=1.0):
    """Decorator for exponential backoff retry logic"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_retries - 1:
                        break
                    if "429" in str(e) or "rate limit" in str(e).lower():
                        logging.warning(f"Rate limit hit, waiting {delay}s")
                    time.sleep(delay)
                    delay = min(delay * 2, 32.0)
            
            raise last_exception
        return wrapper
    return decorator

def validate_timeline_data(timeline_data):
    """Validate and sanitize timeline data"""
    required_fields = ['timeline', 'summary', 'highlights', 'authenticity_score']
    
    for field in required_fields:
        if field not in timeline_data:
            timeline_data[field] = [] if field != 'summary' else "No summary available"
    
    valid_events = []
    for event in timeline_data.get('timeline', []):
        if not isinstance(event, dict) or not event.get('event'):
            continue
        event.setdefault('date', 'Unknown')
        event.setdefault('confidence', 0.5)
        event.setdefault('sources', [])
        try:
            conf = float(event['confidence'])
            event['confidence'] = max(0.0, min(1.0, conf))
        except:
            event['confidence'] = 0.5
        valid_events.append(event)
    
    timeline_data['timeline'] = valid_events
    return timeline_data

def check_data_quality(articles, events):
    """Assess data quality and provide warnings"""
    warnings = []
    quality_score = 100
    
    if len(articles) < 3:
        warnings.append("⚠️ Very few articles found. Results may be incomplete.")
        quality_score -= 20
    
    if len(events) < 5:
        warnings.append("⚠️ Limited events extracted. Timeline may lack detail.")
        quality_score -= 20
    
    dated_events = [e for e in events if e.get('date') and e['date'] != 'Unknown']
    if len(dated_events) < len(events) * 0.5:
        warnings.append("⚠️ Many events lack specific dates.")
        quality_score -= 15
    
    all_sources = set()
    for event in events:
        for s in event.get('sources', []):
            try:
                if isinstance(s, dict):
                    candidate = s.get('url') or s.get('source') or str(s)
                else:
                    candidate = str(s)
                all_sources.add(candidate)
            except Exception:
                continue
    
    if len(all_sources) < 3:
        warnings.append("⚠️ Limited source diversity.")
        quality_score -= 15
    
    return {'quality_score': max(0, quality_score), 'warnings': warnings}

# ============================================
# PERFORMANCE
# ============================================

class ResultCache:
    """Simple caching system"""
    def __init__(self, cache_dir=".cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _generate_key(self, data):
        import json
        serialized = json.dumps(data, sort_keys=True).encode()
        return hashlib.sha256(serialized).hexdigest()
    
    def get(self, key):
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        return None
    
    def set(self, key, value):
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except:
            pass
    
    def cached_extraction(self, articles, model):
        cache_key = self._generate_key({'urls': [a.get('url', '') for a in articles], 'model': model})
        cached = self.get(cache_key)
        if cached:
            logging.info("Using cached extraction results")
            return cached
        from backend.extractor import extract_events
        events = extract_events(articles, model=model)
        self.set(cache_key, events)
        return events