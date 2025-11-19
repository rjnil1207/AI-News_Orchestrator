import re
import difflib
import os
from datetime import datetime, timezone
from backend.utils import compute_advanced_authenticity
import requests

# ===== NER & DATE EXTRACTION SETUP =====
# spaCy model loading (lazy-loaded on first use)
_spacy_nlp = None

def get_spacy_model():
    """Get spaCy model for NER, lazy-loaded"""
    global _spacy_nlp
    if _spacy_nlp is not None:
        return _spacy_nlp
    try:
        import spacy
        try:
            _spacy_nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Download model if not present
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], 
                          capture_output=True, timeout=60)
            _spacy_nlp = spacy.load("en_core_web_sm")
        return _spacy_nlp
    except Exception:
        return None


def get_groq_llm():
    """Get Groq LLM instance if available"""
    try:
        from langchain_openai import ChatOpenAI
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            return None
        return ChatOpenAI(
            model="llama-3.1-8b-instant",
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.3
        )
    except Exception:
        return None

def normalize(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def deduplicate_events(events, threshold=0.70):
    """Remove duplicate or near-duplicate events"""
    final = []
    seen = []  # stores normalized texts corresponding to final indexes

    for e in events:
        text = e.get("event", "") or ""
        norm = normalize(text)

        merged = False
        for idx, s in enumerate(seen):
            sim = difflib.SequenceMatcher(None, norm, s).ratio()
            if sim >= threshold:
                # Merge this event into the existing one at final[idx]
                existing = final[idx]
                # Merge sources (union), preserving strings
                srcs_existing = existing.get('sources', []) or []
                srcs_new = e.get('sources', []) or []
                try:
                    merged_srcs = list({str(x) for x in (list(srcs_existing) + list(srcs_new))})
                except Exception:
                    merged_srcs = list(srcs_existing) + list(srcs_new)
                existing['sources'] = merged_srcs

                # Prefer the event text that is longer (likely more descriptive)
                try:
                    if len(e.get('event', '') or '') > len(existing.get('event', '') or ''):
                        existing['event'] = e.get('event', '')
                except Exception:
                    pass

                # Merge confidence by taking the max (more confident source wins)
                try:
                    existing_conf = float(existing.get('confidence', 0) or 0)
                    new_conf = float(e.get('confidence', 0) or 0)
                    existing['confidence'] = max(existing_conf, new_conf)
                except Exception:
                    pass

                # Merge date: prefer the one that is not 'Unknown' or the earlier one
                try:
                    ed = existing.get('date')
                    nd = e.get('date')
                    if (not ed or ed == 'Unknown') and nd:
                        existing['date'] = nd
                except Exception:
                    pass

                merged = True
                break

        if not merged:
            # append a shallow copy to avoid mutating input objects
            final.append({k: v for k, v in e.items()})
            seen.append(norm)

    return final

def safe_date(d):
    try:
        if not d:
            return None
        try:
            dt = datetime.fromisoformat(d)
        except Exception:
            from dateutil import parser as dateparser
            dt = dateparser.parse(d)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except:
        return None

def generate_summary_llm(timeline):
    """Generate comprehensive narrative summary showing story reconstruction and information evolution"""
    if not timeline or len(timeline) == 0:
        return "No summary available."
    
    llm = get_groq_llm()
    if not llm:
        # Fallback to heuristic
        return generate_summary_heuristic(timeline)
    
    try:
        # Build context with temporal information
        events_text = "\n".join([f"- {e.get('date', 'Unknown')}: {e.get('event', '')} (confidence: {e.get('confidence', 0)})" 
                                for e in timeline[:10]])
        
        prompt = f"""Reconstruct the story from this timeline of events. Show how the narrative unfolds and how information evolved:

{events_text}

Create a 4-5 sentence narrative summary that:
1. Explains the core event or story
2. Shows how the situation developed or changed
3. Highlights key turning points or new developments
4. Indicates conflicts or multiple perspectives if present
5. Conveys the overall trajectory and current state

Focus on narrative flow and authentic clarity - reconstruct the STORY, not just list events."""

        response = llm.invoke(prompt)
        summary = getattr(response, "content", str(response)).strip()

        if not summary:
            return generate_summary_heuristic(timeline)

        # Prefer sentence-splitting; enforce 4-5 sentences
        import re as _re
        sentences = _re.split(r'(?<=[.!?])\s+', summary)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) >= 4:
            return "\n".join(sentences[:5])

        # Fallback: try splitting by lines
        lines = [l.strip() for l in summary.splitlines() if l.strip()]
        if len(lines) >= 4:
            # take up to 5 lines
            return "\n".join(lines[:5])

        # If LLM returned too little, fallback to heuristic to ensure 4-5 sentences
        return generate_summary_heuristic(timeline)
    except Exception:
        return generate_summary_heuristic(timeline)

def generate_summary_heuristic(timeline):
    """Fallback heuristic summary"""
    if not timeline:
        return "No summary available."

    first = timeline[0]["event"] if timeline else ""
    last = timeline[-1]["event"] if timeline else ""
    
    first_date = timeline[0].get('date', 'Unknown') if timeline else 'Unknown'
    last_date = timeline[-1].get('date', 'Unknown') if timeline else 'Unknown'
    avg_confidence = sum(e.get('confidence', 0) for e in timeline) / len(timeline) if timeline else 0
    
    # Produce 4-5 short sentences to match the LLM requirement
    sentences = []
    sentences.append(f"Overview: {len(timeline)} key events spanning {first_date} to {last_date}.")
    sentences.append(f"Average extraction confidence is {int(avg_confidence * 100)}%.")
    sentences.append(f"Opening: {first[:120].rstrip('.')}.")
    sentences.append(f"Closing: {last[:120].rstrip('.')}.")
    # include a short highlight (highest confidence event)
    try:
        top = sorted(timeline, key=lambda x: x.get('confidence', 0), reverse=True)[0]
        sentences.append(f"Top highlight: {top.get('event','')[:140].rstrip('.')}.")
    except Exception:
        pass

    # Ensure at least 4 sentences (if not, pad using available fragments)
    if len(sentences) < 4:
        # pad by repeating the top highlight or summary fragments
        frag = (timeline[0].get('event','')[:100] if timeline else '')
        while len(sentences) < 4:
            sentences.append((frag.rstrip('.') + '.') if frag else "Summary incomplete.")

    return "\n".join(sentences[:5])

def extract_highlights(timeline):
    if not timeline:
        return []
    sorted_items = sorted(timeline, key=lambda x: x.get("confidence", 0), reverse=True)
    return [item["event"] for item in sorted_items[:3]]

def find_turning_points_llm(timeline):
    """Identify turning points that show how the story evolved and information developed"""
    if not timeline or len(timeline) < 2:
        return []
    
    llm = get_groq_llm()
    if not llm:
        return find_turning_points_heuristic(timeline)
    
    try:
        events_text = "\n".join([f"{i+1}. [{e.get('date', 'Unknown')}] {e.get('event', '')}" 
                                for i, e in enumerate(timeline)])
        
        prompt = f"""Analyze this timeline to identify 2-3 critical turning points where the story fundamentally changed:

{events_text}

For STORY RECONSTRUCTION, identify moments where:
- A major development changed the narrative
- New information revealed or contradicted earlier reports
- The situation escalated, resolved, or pivoted
- Key stakeholders entered or changed their position

For each turning point, provide:
- The specific event that was the turning point
- Why it mattered (what changed in the narrative, why it's significant)

Format as JSON array with objects containing "event" and "reason" fields."""
        
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response)).strip()
        
        # Try to extract JSON
        import json
        try:
            # Find JSON array in response
            start = content.find('[')
            end = content.rfind(']') + 1
            if start != -1 and end > start:
                json_str = content[start:end]
                points = json.loads(json_str)
                if isinstance(points, list):
                    return points[:3]
        except:
            pass
        
        return find_turning_points_heuristic(timeline)
    except Exception:
        return find_turning_points_heuristic(timeline)

def find_turning_points_heuristic(timeline):
    """Heuristic-based turning point detection"""
    tps = []
    for i in range(1, len(timeline)):
        prev = timeline[i-1].get("confidence", 0)
        curr = timeline[i].get("confidence", 0)
        if curr - prev >= 0.35:
            tps.append({
                "event": timeline[i].get("event"),
                "reason": "Significant jump in confidence signals a major development."
            })
    return tps

def detect_discrepancies_llm(timeline):
    """Detect discrepancies using LLM analysis"""
    if not timeline or len(timeline) < 2:
        return []
    
    llm = get_groq_llm()
    if not llm:
        return detect_discrepancies_heuristic(timeline)
    
    try:
        events_text = "\n".join([f"- [{e.get('date', 'Unknown')}] {e.get('event', '')} (src: {', '.join(e.get('sources', [])[:2])})" 
                                for e in timeline])
        
        prompt = f"""Analyze this timeline to VERIFY AUTHENTICITY and detect conflicting narratives that threaten clarity:

{events_text}

Identify critical discrepancies that matter for reconstructing the true story:
- Contradictory outcomes (success vs failure, different results reported)
- Conflicting details that change the story (dates, who/what, impact)
- Unexplained gaps that suggest missing information
- Different interpretations that suggest multiple narratives
- Claims that contradict earlier established facts

For each significant discrepancy, explain:
1. What are the conflicting claims?
2. Which sources report each version?
3. What impact does this have on understanding the true story?

Format as JSON array with objects containing "issue", "explanation", and "impact" fields."""
        
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response)).strip()
        
        # Try to extract JSON
        import json
        try:
            start = content.find('[')
            end = content.rfind(']') + 1
            if start != -1 and end > start:
                json_str = content[start:end]
                issues = json.loads(json_str)
                if isinstance(issues, list):
                    # Attempt to attach source details to each issue by matching examples or event text
                    enriched = []
                    for it in issues[:10]:
                        srcs = set()
                        # try examples
                        examples = it.get('examples', []) if isinstance(it.get('examples', []), list) else []
                        for ex in examples:
                            for e in timeline:
                                if ex and ex.strip() and ex.strip() in (e.get('event') or ''):
                                    srcs.update(e.get('sources', []))
                        # fallback: search for keywords from explanation
                        if not srcs:
                            expl = it.get('explanation', '')
                            for e in timeline:
                                if any(tok.lower() in (e.get('event') or '').lower() for tok in expl.split()[:6]):
                                    srcs.update(e.get('sources', []))
                        if srcs:
                            it['sources'] = list({str(s) for s in srcs})
                            enriched.append(it)
                    if enriched:
                        return enriched[:5]
        except Exception:
            pass

        return detect_discrepancies_heuristic(timeline)
    except Exception:
        return detect_discrepancies_heuristic(timeline)

def detect_discrepancies_heuristic(timeline):
    """Heuristic discrepancy detection"""
    issues = []
    for item in timeline:
        text = item.get("event", "")
        if any(w in text.lower() for w in ("contradict", "dispute", "conflicting", "disagrees", "unclear")):
            srcs = item.get('sources', []) or []
            if srcs:
                issues.append({
                    "issue": "Conflicting reports detected",
                    "explanation": f"Event: '{item['event'][:140]}...' suggests source disagreement.",
                    "sources": [str(s) for s in srcs]
                })
    return issues[:5]


# Outcome keyword groups for basic contradiction detection
SUCCESS_TERMS = ["success", "successful", "soft landing", "achieved", "succeeded", "lifted off", "landed", "rescued"]
FAILURE_TERMS = ["fail", "failed", "crash", "destroyed", "lost", "unsuccessful", "did not"]


def identify_inconsistencies(timeline, similarity_threshold=0.75, divergence_threshold=0.35, min_text_length=40):
    """Identify cross-source inconsistencies by grouping events by date (or Unknown)
    and checking for opposing outcome terms or highly dissimilar descriptions for the
    same date. To reduce noisy flags, we:
      - Always flag explicit contradictory outcomes (success vs failure terms)
      - For divergent descriptions, require similarity < `divergence_threshold` AND both
        descriptions longer than `min_text_length` characters.

    Returns a list of discrepancy dicts with involved sources.
    """
    from collections import defaultdict
    clusters = defaultdict(list)
    for item in timeline:
        key = item.get('date') or 'Unknown'
        clusters[key].append(item)

    discrepancies = []
    for date_key, items in clusters.items():
        if len(items) < 2:
            continue
        # collect outcome signals
        has_success = any(any(t in (items[i].get('event') or '').lower() for t in SUCCESS_TERMS) for i in range(len(items)))
        has_failure = any(any(t in (items[i].get('event') or '').lower() for t in FAILURE_TERMS) for i in range(len(items)))

        # if both success and failure terms present -> discrepancy
        if has_success and has_failure:
            sources = []
            snippets = []
            for it in items:
                snippets.append(it.get('event'))
                sources.extend(it.get('sources', []))
            discrepancies.append({
                'issue': 'Contradictory outcomes reported',
                'explanation': f'Multiple sources report opposing outcomes for {date_key}.',
                'examples': snippets,
                'sources': list({str(s) for s in sources})
            })
            continue

        # compare pairwise similarity to detect divergent descriptions
        for i in range(len(items)):
            for j in range(i+1, len(items)):
                a = (items[i].get('event') or '').strip()
                b = (items[j].get('event') or '').strip()
                if not a or not b:
                    continue
                sim = difflib.SequenceMatcher(None, a, b).ratio()
                # Only flag divergence if both descriptions are sufficiently long
                if sim < divergence_threshold and len(a) >= min_text_length and len(b) >= min_text_length:
                    sources = list({*(items[i].get('sources', [])), *(items[j].get('sources', []))})
                    discrepancies.append({
                        'issue': 'Inconsistent descriptions',
                        'explanation': f'Event descriptions for {date_key} diverge (similarity={sim:.2f}).',
                        'examples': [a, b],
                        'sources': [str(s) for s in sources]
                    })
    return discrepancies


def fact_consistency_check(timeline):
    """Detect simple factual inconsistencies across events.
    Heuristics:
    - numeric conflicts for same named entity (different numbers reported)
    - direct negation contradictions (one says 'X' another 'not X')
    - date contradictions (different dates asserted for same event text)
    Returns list of issues.
    """
    import re
    issues = []

    # helper to extract numbers and normalized tokens
    def extract_numbers(s):
        nums = re.findall(r"\b\d{1,3}(?:[\,\d]*)(?:\.\d+)?\b", s)
        return [n.replace(',', '') for n in nums]

    def has_negation(s):
        return any(w in s.lower() for w in ["not ", "n't", "never", "no "])

    # Compare pairwise
    for i in range(len(timeline)):
        for j in range(i+1, len(timeline)):
            a = timeline[i].get('event', '')
            b = timeline[j].get('event', '')
            if not a or not b:
                continue

            # numeric conflict detection
            nums_a = extract_numbers(a)
            nums_b = extract_numbers(b)
            if nums_a and nums_b and nums_a != nums_b:
                issues.append({
                    'type': 'numeric_conflict',
                    'examples': [a, b],
                    'details': {'a_numbers': nums_a, 'b_numbers': nums_b},
                    'sources': list({*(timeline[i].get('sources', [])), *(timeline[j].get('sources', []))})
                })

            # negation contradiction
            neg_a = has_negation(a)
            neg_b = has_negation(b)
            # if one negates and the other asserts similar short phrase, flag
            if neg_a != neg_b:
                # cheap similarity check for main nouns/verbs
                sim = difflib.SequenceMatcher(None, normalize(a), normalize(b)).ratio()
                if sim >= 0.45:
                    issues.append({
                        'type': 'negation_contradiction',
                        'examples': [a, b],
                        'similarity': sim,
                        'sources': list({*(timeline[i].get('sources', [])), *(timeline[j].get('sources', []))})
                    })

            # date contradiction: if both mention explicit dates but differ
            date_re = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
            da = date_re.findall(a)
            db = date_re.findall(b)
            if da and db and set(da) != set(db):
                issues.append({
                    'type': 'date_conflict',
                    'examples': [a, b],
                    'details': {'a_dates': da, 'b_dates': db},
                    'sources': list({*(timeline[i].get('sources', [])), *(timeline[j].get('sources', []))})
                })

    # dedupe issues by signature
    seen = set()
    dedup = []
    for it in issues:
        sig = (it.get('type'), tuple(it.get('examples', [])))
        if sig in seen:
            continue
        seen.add(sig)
        dedup.append(it)

    return dedup


def extract_entities(timeline):
    """Entity extraction using advanced heuristics with regex patterns.
    Returns dict with lists: dates, persons, locations, organizations, outcomes, misc.
    """
    import re
    from datetime import datetime
    from dateutil import parser as dateparser
    
    dates = set()
    persons = set()
    locations = set()
    orgs = set()
    misc = set()
    outcomes = set()

    # Enhanced date patterns (ISO, ordinals, month name variants, numeric variants)
    date_patterns = [
        # ISO date or datetime: 2025-11-16 or 2025-11-16T15:15:00+00:00 or 2025-11-16 15:15:00
        r"\b\d{4}-\d{2}-\d{2}(?:[T\s]\d{2}:\d{2}:\d{2}(?:Z|[+\-]\d{2}:?\d{2})?)?\b",
        # Month name variations: November 16, 2025 | Nov 16th, 2025
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b",
        # Day-first with month name: 16 November 2025 | 16th Nov 2025
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec),?\s+\d{4}\b",
        # Numeric formats: 16/11/2025 or 11/16/2025 (we'll parse both)
        r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b",
    ]
    
    # Known location keywords (US states, countries, major cities)
    location_keywords = {
        'alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado', 'connecticut', 'delaware',
        'florida', 'georgia', 'hawaii', 'idaho', 'illinois', 'indiana', 'iowa', 'kansas', 'kentucky',
        'louisiana', 'maine', 'maryland', 'massachusetts', 'michigan', 'minnesota', 'mississippi', 'missouri',
        'montana', 'nebraska', 'nevada', 'hampshire', 'jersey', 'mexico', 'york', 'carolina', 'dakota', 'ohio',
        'oklahoma', 'oregon', 'pennsylvania', 'island', 'tennessee', 'texas', 'utah', 'vermont', 'virginia',
        'washington', 'wisconsin', 'wyoming',
        'london', 'paris', 'tokyo', 'beijing', 'mumbai', 'delhi', 'sydney', 'toronto', 'dubai', 
        'singapore', 'hongkong', 'bangkok', 'istanbul', 'moscow', 'berlin', 'madrid', 'rome', 'amsterdam',
        'united states', 'united kingdom', 'canada', 'china', 'india', 'japan', 'australia', 'germany',
        'france', 'italy', 'spain', 'russia', 'brazil', 'mexico', 'south korea', 'korea', 'uae',
        'city', 'province', 'state', 'island', 'country', 'river', 'lake', 'mountain', 'county',
        'district', 'region', 'nation', 'continent', 'bay', 'strait', 'gulf', 'sea', 'ocean',
        'cupertino', 'seattle', 'new york', 'los angeles', 'chicago', 'san francisco', 'boston',
        'miami', 'denver', 'portland', 'austin', 'vegas', 'haryana', 'kashmir', 'new delhi', 'red fort'
    }
    
    # Org keywords
    org_keywords = ['inc', 'ltd', 'corp', 'company', 'university', 'school', 'bank', 'hospital', 
                    'foundation', 'organization', 'org', 'institute', 'agency', 'department', 'police', 'force', 'ministry', 'board']
    
    # Common nouns to skip (not entities)
    common_nouns = {
        'the', 'a', 'an', 'and', 'or', 'but', 'with', 'in', 'at', 'on', 'to', 'from', 'by', 'for',
        'arrests', 'car', 'cars', 'man', 'woman', 'people', 'persons', 'group', 'team', 'group', 'area',
        'report', 'case', 'attack', 'event', 'incident', 'issue', 'problem', 'situation', 'statement',
        'order', 'law', 'right', 'rule', 'system', 'process', 'result', 'effect', 'cause', 'reason',
        'day', 'night', 'time', 'date', 'year', 'month', 'week', 'hour', 'minute', 'second',
        'investigation', 'success', 'failure', 'action', 'activity', 'work', 'job', 'task', 'evidence',
        'finding', 'search', 'check', 'examination', 'analysis', 'review', 'campus', 'building', 'office'
    }
    
    # Try spaCy NER extraction for better named entity recognition
    spacy_persons = set()
    spacy_locations = set()
    spacy_orgs = set()
    try:
        nlp = get_spacy_model()
        if nlp:
            # Process all events with spaCy
            combined_text = "\n".join([item.get('event', '') for item in timeline if item.get('event')])
            if combined_text:
                doc = nlp(combined_text[:50000])  # Limit to first 50k chars for performance
                for ent in doc.ents:
                    if ent.label_ == "PERSON":
                        spacy_persons.add(ent.text)
                    elif ent.label_ in ("GPE", "LOC"):  # GPE=geopolitical entity, LOC=location
                        spacy_locations.add(ent.text)
                    elif ent.label_ in ("ORG", "PRODUCT"):  # ORG=organization, PRODUCT=brand
                        spacy_orgs.add(ent.text)
    except Exception:
        pass  # If spaCy fails, continue with heuristics

    for item in timeline:
        text = item.get('event', '')
        # Also consider explicit date field on the event and normalize it
        raw_date_field = item.get('date')
        if raw_date_field:
            try:
                parsed = dateparser.parse(str(raw_date_field))
                if parsed:
                    try:
                        from datetime import timezone
                        if parsed.tzinfo is None:
                            parsed = parsed.replace(tzinfo=timezone.utc)
                        else:
                            parsed = parsed.astimezone(timezone.utc)
                    except Exception:
                        pass
                    if parsed.time().hour == 0 and parsed.time().minute == 0 and parsed.time().second == 0:
                        dates.add(parsed.date().isoformat())
                    else:
                        iso = parsed.isoformat()
                        if iso.endswith('+00:00'):
                            iso = iso[:-6] + 'Z'
                        dates.add(iso)
            except Exception:
                # if parsing fails, still add the raw string
                try:
                    if str(raw_date_field).strip():
                        dates.add(str(raw_date_field).strip())
                except Exception:
                    pass
        if not text:
            continue
        
        # Extract dates first and normalize them to ISO-like strings where possible
        for pattern in date_patterns:
            for m in re.findall(pattern, text, re.IGNORECASE):
                m = m.strip()
                if not m:
                    continue
                # try to parse and normalize
                try:
                    parsed = dateparser.parse(m)
                    if parsed:
                        # normalize to UTC-aware ISO if possible
                        try:
                            from datetime import timezone
                            if parsed.tzinfo is None:
                                parsed = parsed.replace(tzinfo=timezone.utc)
                            else:
                                parsed = parsed.astimezone(timezone.utc)
                        except Exception:
                            pass
                        # prefer full ISO timestamp when time is present, otherwise date
                        if parsed.time().hour == 0 and parsed.time().minute == 0 and parsed.time().second == 0:
                            dates.add(parsed.date().isoformat())
                        else:
                            # use ISO format with Z for UTC when possible
                            iso = parsed.isoformat()
                            # normalize +00:00 to Z for readability
                            if iso.endswith('+00:00'):
                                iso = iso[:-6] + 'Z'
                            dates.add(iso)
                        continue
                except Exception:
                    pass

                # fallback: keep the raw matched string
                dates.add(m)
        
        # Extract organization/location mentions via prepositions (e.g., "in California", "at Apple Inc")
        loc_org_patterns = [r'\b(?:in|at|near|from|to|across|through|within|throughout)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b']
        for pattern in loc_org_patterns:
            for m in re.findall(pattern, text):
                m_lower = m.lower()
                # Decide if it's an org or location based on keywords
                if any(w in m_lower for w in org_keywords):
                    orgs.add(m)
                else:
                    locations.add(m)
        
        # Extract "Organization of X" patterns (e.g., "University of Toronto", "Bank of America")
        org_patterns = [
            r'\b((?:University|Bank|Institute|Department|Foundation|Company|Organization|Ministry|Agency|Police|Force)\s+of\s+[A-Z][a-z]+)\b',
            r'\b([A-Z][a-z]+\s+(?:Police|Force|Ministry|Board|Commission|Authority))\b'
        ]
        for pattern in org_patterns:
            for m in re.findall(pattern, text):
                if m not in orgs:
                    orgs.add(m)
        
        # Extract capitalized phrases (persons, locations, orgs)
        cap_re = re.compile(r"\b([A-Z][a-z]{1,}(?:\s+[A-Z][a-z]{1,}){0,3})\b")
        for m in cap_re.findall(text):
            m_lower = m.lower()
            
            # Skip month names and single letters
            if m_lower in ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december',
                          'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']:
                continue
            
            # Skip common nouns
            if m_lower in common_nouns:
                continue
            
            # Skip if already extracted via preposition or org patterns
            if m in orgs or m in locations:
                continue
            
            # Skip if it's a substring of an already extracted org (e.g., "University" in "University of Toronto")
            if any(m_lower in org_lower for org_lower in [o.lower() for o in orgs]):
                continue
            
            # Check if it's an organization keyword
            if any(w in m_lower for w in org_keywords):
                orgs.add(m)
            # Check if it's a known location
            elif any(loc in m_lower for loc in location_keywords):
                locations.add(m)
            # Check if it contains articles or common words (skip them)
            elif any(w in m_lower for w in ['the ', 'and ', 'or ', 'a ']):
                continue
            else:
                # Classification logic for ambiguous phrases
                word_count = len(m.split())
                
                # Single word: could be location or person - check context
                if word_count == 1:
                    # Single word - if it's already in locations from prepositions, skip
                    # Otherwise, default to person (e.g., "John", "Delhi" - but Delhi already caught above)
                    if m not in locations:
                        persons.add(m)
                
                # Two words: likely a person name (e.g., "John Smith")
                elif word_count == 2:
                    persons.add(m)
                
                # Three or more words: ambiguous, but if contains location keywords in location_keywords, prefer location
                else:
                    # Check if any word in the phrase is a location keyword
                    words = m.split()
                    has_location_word = any(word.lower() in location_keywords for word in words)
                    has_org_word = any(w in m_lower for w in org_keywords)
                    
                    if has_org_word:
                        orgs.add(m)
                    elif has_location_word:
                        locations.add(m)
                    else:
                        # Default to person for multi-word phrases without location/org keywords
                        # (e.g., "Amir Rashid Ali" is a 3-word person name)
                        persons.add(m)
        
        # Extract outcomes
        low = text.lower()
        for t in SUCCESS_TERMS:
            if t in low:
                outcomes.add(t)
        for t in FAILURE_TERMS:
            if t in low:
                outcomes.add(t)

    # Merge spaCy NER results (complement heuristics)
    try:
        persons.update(spacy_persons)
        locations.update(spacy_locations)
        orgs.update(spacy_orgs)
    except Exception:
        pass

    out = {
        'dates': sorted(dates),
        'persons': sorted(persons),
        'locations': sorted(locations),
        'organizations': sorted(orgs),
        'outcomes': sorted(outcomes),
        'misc': sorted(misc)
    }

    return out


def generate_timeline(events, model=None):
    """
    MAIN pipeline – no embeddings, hybrid-friendly.
    Input: list of event dicts with keys: date, event, confidence, sources
    """
    if not events:
        return {
            "timeline": [],
            "summary": "No events found.",
            "highlights": [],
            "turning_points": [],
            "discrepancies": [],
            "authenticity_score": {"overall_score": 0, "explanation": "No data available", "grade": "F"}
        }

    # 1. Fast dedupe
    events = deduplicate_events(events, threshold=0.70)

    # 2. Normalize entries and ensure fields
    normalized = []
    for e in events:
        d = e.get("date", "Unknown")
        try:
            sort_key = safe_date(d) or datetime.min.replace(tzinfo=timezone.utc)
        except:
            sort_key = datetime.min.replace(tzinfo=timezone.utc)
        normalized.append({
            "date": d if d else "Unknown",
            "event": e.get("event", "")[:800],
            "confidence": float(e.get("confidence", 0.5)) if e.get("confidence") is not None else 0.5,
            "sources": e.get("sources", []) or [],
            "_sort": sort_key
        })

    # 3. Sort chronologically (Unknown at end)
    normalized = sorted(normalized, key=lambda x: x["_sort"] or datetime.min.replace(tzinfo=timezone.utc))
    for x in normalized:
        if "_sort" in x:
            del x["_sort"]

    # 4. Summary, highlights, turning points, discrepancies (using LLM)
    summary = generate_summary_llm(normalized)
    highlights = extract_highlights(normalized)
    turning_points = find_turning_points_llm(normalized)
    discrepancies = detect_discrepancies_llm(normalized)
    # augment with cross-source inconsistency detection
    try:
        inconsist = identify_inconsistencies(normalized)
        if inconsist:
            discrepancies.extend(inconsist)
    except Exception:
        pass
    
    # 5. NEW - Advanced authenticity scoring
    authenticity = compute_advanced_authenticity(normalized)
    # 6. FACT CONSISTENCY CHECK
    try:
        consistency_issues = fact_consistency_check(normalized)
    except Exception:
        consistency_issues = []

    return {
        "timeline": normalized,
        "summary": summary,
        "highlights": highlights,
        "turning_points": turning_points,
        "discrepancies": discrepancies,
        "consistency_issues": consistency_issues,
        "authenticity_score": authenticity  # Now returns detailed breakdown
    }