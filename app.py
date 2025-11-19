import streamlit as st
import os
import time
import requests
from dotenv import load_dotenv
from backend.fetcher import aggregate, fetch_top_headlines
from backend.generator import generate_timeline


# NEW IMPORTS
from backend.utils import (
    create_interactive_timeline,
    create_confidence_distribution,
    retry_with_exponential_backoff,
    validate_timeline_data,
    check_data_quality,
    ResultCache,
    detect_subjective_tone,
    detect_clickbait_headline,
    save_timeline_history,
    load_timeline_history
)
try:
    from backend.generator import extract_entities
except Exception as e:
    import logging
    logging.warning(f"Could not import helpers from generator: {e}")
    def extract_entities(timeline):
        # fallback stub: returns empty lists so UI can still function
        return {'dates': [], 'persons': [], 'locations': [], 'organizations': [], 'misc': [], 'outcomes': []}
    def format_progressive_summary(timeline, max_events=12):
        return [], "No events to summarize.", {"overall_score": 0, "grade": "F", "breakdown": {}}

try:
    from langchain_openai import ChatOpenAI
    _HAS_LANGCHAIN_OPENAI = True
except Exception:
    _HAS_LANGCHAIN_OPENAI = False

load_dotenv()

# ---- SUMMARY TRANSLATION ENGINE ----
SUPPORTED_LANGS = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Marathi": "mr",
}

def translate_google(text: str, target_lang: str):
    try:
        if not text:
            return text
        url = "https://translate.googleapis.com/translate_a/single"
        params = {"client": "gtx", "sl": "auto", "tl": target_lang, "dt": "t", "q": text}
        resp = requests.get(url, params=params, timeout=6)
        resp.raise_for_status()
        arr = resp.json()
        return "".join([seg[0] for seg in arr[0]])
    except Exception:
        return None

def translate_groq(text: str, target_lang: str):
    try:
        if not _HAS_LANGCHAIN_OPENAI:
            return None
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            return None
        llm = ChatOpenAI(model="llama-3.1-8b-instant", api_key=groq_key, 
                        base_url="https://api.groq.com/openai/v1", temperature=0)
        prompt = f"Translate the following English text into {target_lang} (keep meaning exact, don't add commentary):\n\n{text}"
        resp = llm.invoke(prompt)
        return getattr(resp, "content", None)
    except Exception:
        return None

def translate_summary(text: str, lang_code: str):
    if not text or lang_code == "en":
        return text
    g = translate_google(text, lang_code)
    if g:
        return g
    g2 = translate_groq(text, lang_code)
    if g2:
        return g2
    return text


@st.cache_data(ttl=60 * 10)
def cached_aggregate(query, top_n, sources_tuple, use_cache_flag):
    # lightweight wrapper around fetcher.aggregate with caching
    return aggregate(query, top_n=top_n, sources=tuple(sources_tuple), use_cache=use_cache_flag)


@st.cache_resource
def get_result_cache():
    return ResultCache()

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="AI Timeline Generator",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===== CUSTOM CSS (keeping your existing styles but enhancing colors & spacing only) =====
st.markdown("""
    <style>
        body, .stApp {
            background-color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }

    /* Modern gradient (same style, better colors) */
    .main { 
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); 
        background-attachment: fixed; 
    }

    /* Slightly more breathing space (UI unchanged, only cleaner) */
    .hero { 
        text-align: center; 
        padding: 0px 20px 0; 
        margin-bottom: 0px; 
    }
    .hero-title { 
        font-size: 48px; 
        font-weight: 800; 
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
        margin-bottom: 8px; 
        letter-spacing: -1.5px; 
    }
    .hero-subtitle { 
    font-size: 18px; 
    color: #374151; 
    font-weight: 400; 
    margin-top: -6px; 
    letter-spacing: 0.2px;
    }        

    .glass-card { background: white; border-radius: 16px; padding: 20px; margin-bottom: 12px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15); border: 1px solid rgba(102, 126, 234, 0.2); }

    .timeline-item { position: relative; padding: 18px; margin: 10px 0; background: white; 
        border-left: 5px solid #4f46e5;
        border-radius: 12px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08); transition: all 0.3s ease; }
    .timeline-item:hover { 
        transform: translateX(10px); 
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3); 
        border-left-color: #7c3aed; 
    }
    .timeline-date { font-size: 12px; font-weight: 700; color: #4f46e5; text-transform: uppercase; 
        letter-spacing: 1.5px; margin-bottom: 6px; }

    .timeline-event { font-size: 15px; color: #2c3e50; line-height: 1.6; margin-bottom: 8px; }
    .timeline-meta { font-size: 11px; color: #7f8c8d; display: flex; gap: 12px; margin-top: 8px; }

    /* Stat boxes improved with modern colors */
    .stat-box { 
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); 
        color: white; padding: 18px 15px;
        border-radius: 16px; text-align: center; 
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4); 
        transition: transform 0.3s ease; 
    }

    .stat-box:hover { transform: translateY(-5px); }

    .stat-number { font-size: 36px; font-weight: 800; margin-bottom: 4px; }
    .stat-label { font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; opacity: 0.95; }

    .section-header { font-size: 24px; font-weight: 700; color: #2c3e50; margin: 20px 0 12px; 
        padding: 15px 20px; background: white; border-radius: 12px; 
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); border-left: 6px solid #4f46e5; }

    .highlight-item { background: white; padding: 15px 20px; margin: 8px 0; border-radius: 10px; 
        border-left: 4px solid #7c3aed; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08); 
        transition: all 0.3s ease; color: #2c3e50; font-size: 14px; line-height: 1.5; }
    .highlight-item:hover { box-shadow: 0 6px 20px rgba(118, 75, 162, 0.3); transform: translateX(8px); }

    .summary-box { background: white; padding: 20px; border-radius: 16px; 
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1); border: 2px solid #4f46e5; }

    .summary-text { font-size: 15px; line-height: 1.7; color: #34495e; }

    .turning-point { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white;
        padding: 18px; border-radius: 12px; margin: 10px 0; 
        box-shadow: 0 6px 20px rgba(245, 87, 108, 0.3); }

    .turning-point-title { font-size: 15px; font-weight: 700; margin-bottom: 6px; }
    .turning-point-reason { font-size: 13px; opacity: 0.95; font-style: italic; }

    .discrepancy-box { background: white; border-left: 5px solid #f39c12; padding: 18px; border-radius: 10px;
        margin: 10px 0; box-shadow: 0 4px 12px rgba(243, 156, 18, 0.2); }

    .discrepancy-title { font-size: 15px; font-weight: 700; color: #e67e22; margin-bottom: 6px; }
    .discrepancy-text { font-size: 13px; color: #555; line-height: 1.5; }

    /* Modern button color - wider */
    .stButton>button { 
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); 
        color: white; border: none;
        border-radius: 50px; padding: 15px 60px; font-size: 20px; font-weight: 700; 
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4); 
        width: 100%;
    }
    
    .stButton>button:hover { transform: translateY(-3px); box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6); }
   
    .stTextInput>div>div>input {
    border-radius: 30px;
    padding: 16px 28px;
    font-size: 18px;
    line-height: 1.0;
    border: 3px solid transparent;
    background: 
        linear-gradient(white, white) padding-box,
        linear-gradient(135deg, #4f46e5, #7c3aed) border-box;
    color: #1f2937;
    }
            
    .stTextInput > div > div:focus,
    .stTextInput > div:focus-within,
    .stTextInput:focus-within {
        box-shadow: none !important;
        border: none !important;
        outline: none !important;
    }
    
    [data-testid="stSidebar"] { display: none; }

    .main .block-container { max-width: 1200px; padding-top: 0.5rem; padding-bottom: 0.5rem; }

    /* Updated progress bar color */
    .stProgress > div > div > div { 
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%); 
    }

    .no-data { 
        text-align: center; padding: 60px 20px; background: white; 
        border-radius: 16px; color: #7f8c8d; font-size: 18px; 
    }
</style>
""", unsafe_allow_html=True)


# ===== HERO SECTION =====
st.markdown("""
<div class="hero">
    <div class="hero-title">🚀 AI News Orchestrator</div>
    <div class="hero-subtitle">Reconstruct events. Detect conflicts. Deliver authentic clarity.</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

# ===== TRENDING NEWS =====
trending_news = fetch_top_headlines(max_results=4)
if trending_news:
    # place trends on a light translucent card so black/gray text is readable with right margin
    st.markdown("<div style='background: rgba(255,255,255,0.92); padding:8px 10px; border-radius:10px; display:inline-block; margin-left:120px; margin-top:-8px;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:16px; font-weight:700; color:#111111; margin-bottom:4px;'>🌎 Recent Top Trends</div>", unsafe_allow_html=True)
    for item in trending_news:
        title = item.get("title", "No title")
        url = item.get("url", "#")
        source = item.get("source", "")
        st.markdown(
            f"<div style='font-size:13px; margin:4px 0; color:#111111;'>• <a href='{url}' target='_blank' style='color:#111111; text-decoration:none; font-weight:600;'>{title}</a> <span style='font-size:11px; color:#666666; margin-left:4px;'>({source})</span></div>",
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

# ===== SEARCH SECTION =====
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    query = st.text_input("🔍 Enter any event, topic, or news story", placeholder="Delhi Terror Attack, Chandrayaan-3 Mission, OpenAI GPT-5 Launch, COP28 Summit", label_visibility="collapsed")
    st.markdown("<div style='height:7px;'></div>", unsafe_allow_html=True)
    generate_btn = st.button("🚀 Generate Timeline", width='stretch', type="primary")

st.markdown("<div style='height:3px;'></div>", unsafe_allow_html=True)

# ===== CONFIGURATION =====
use_gdelt = True
use_gnews = True
num_articles = 10
model_choice = "llama-3.1-8b-instant"
use_cache = True

# ===== MAIN GENERATION LOGIC =====
if generate_btn:
    if not query.strip():
        st.error("⚠️ Please enter a search topic!")
        st.stop()
    
    # Initialize cache (cached resource across reruns)
    cache = get_result_cache()
    
    sources = []
    if use_gdelt:
        sources.append("gdelt")
    if use_gnews:
        sources.append("gnews")
    
    if not sources:
        st.error("⚠️ Select at least one data source!")
        st.stop()
    
    if 'results' not in st.session_state:
        st.session_state.results = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Fetch Articles with retry
        status_text.info("📡 Fetching articles from multiple sources...")
        progress_bar.progress(20)
        
        @retry_with_exponential_backoff(max_retries=3)
        def fetch_with_retry():
            # use Streamlit cached aggregator to avoid repeated network calls
            return cached_aggregate(query, num_articles, tuple(sources), use_cache)
        
        articles = fetch_with_retry()
        
        if not articles:
            st.error("❌ No articles found. Try different keywords.")
            st.stop()
        
        st.session_state.results['articles'] = articles
        progress_bar.progress(40)
        status_text.success(f"✅ Found {len(articles)} articles")
        time.sleep(0.5)
        
        # Step 2: Extract Events with caching
        status_text.info("🔎 Extracting events using AI...")
        progress_bar.progress(60)
        
        events = cache.cached_extraction(articles, model=model_choice)
        if not events:
            st.warning("⚠️ No events extracted. Try more articles.")
            st.stop()
        
        # Check data quality
        quality_check = check_data_quality(articles, events)
        
        st.session_state.results['events'] = events
        st.session_state.results['quality'] = quality_check
        progress_bar.progress(80)
        status_text.success(f"✅ Extracted {len(events)} events")
        time.sleep(0.5)
        
        # Step 3: Generate Timeline
        status_text.info("🎨 Generating comprehensive timeline...")
        progress_bar.progress(90)
        
        result = generate_timeline(events, model=model_choice)
        result = validate_timeline_data(result)
        st.session_state.results['timeline'] = result
        
        progress_bar.progress(100)
        status_text.success("✅ Timeline generated successfully!")
        
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        # Show quality warnings if any
        if quality_check['warnings']:
            with st.expander("⚠️ Data Quality Warnings", expanded=False):
                for warning in quality_check['warnings']:
                    st.warning(warning)
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.stop()

# ===== DISPLAY RESULTS =====
if 'results' in st.session_state and st.session_state.results:
    result = st.session_state.results.get('timeline', {})
    articles = st.session_state.results.get('articles', [])
    events = st.session_state.results.get('events', [])
    
    # Stats Overview
    st.markdown('<div class="section-header">📊 Overview Statistics</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'<div class="stat-box"><div class="stat-number">{len(articles)}</div><div class="stat-label">Articles</div></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'<div class="stat-box"><div class="stat-number">{len(events)}</div><div class="stat-label">Events</div></div>', unsafe_allow_html=True)
    
    with col3:
        timeline_count = len(result.get('timeline', []))
        st.markdown(f'<div class="stat-box"><div class="stat-number">{timeline_count}</div><div class="stat-label">Timeline Items</div></div>', unsafe_allow_html=True)
    
    with col4:
        auth_score_data = result.get('authenticity_score', {})
        score = auth_score_data.get('overall_score', auth_score_data.get('score', 0))
        grade = auth_score_data.get('grade', '?')
        st.markdown(f'<div class="stat-box"><div class="stat-number">{score}</div><div class="stat-label">Auth. (Grade: {grade})</div></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Key Highlights (moved before summary)
    st.markdown('<div class="section-header">🌟 Key Highlights</div>', unsafe_allow_html=True)
    
    highlights = result.get('highlights', [])
    if highlights:
        for highlight in highlights:
            st.markdown(f'<div class="highlight-item">✨ {highlight}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="no-data">💡 No highlights available</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Summary Section
    st.markdown('<div class="section-header">📝 Summary</div>', unsafe_allow_html=True)
    
    lang_choice = st.selectbox("🌐 Summary Language", list(SUPPORTED_LANGS.keys()), index=0)
    lang_code = SUPPORTED_LANGS[lang_choice]
    
    summary = result.get('summary', 'No summary available')
    translated_summary = translate_summary(summary, lang_code)
    
    # Enhanced summary with details
    timeline_items = result.get('timeline', [])
    first_event = timeline_items[0].get('event', '') if timeline_items else ''
    last_event = timeline_items[-1].get('event', '') if timeline_items else ''
    num_events = len(timeline_items)
    num_sources = len(set(a.get('url') for a in articles if a.get('url')))
    auth_score = auth_score_data.get('overall_score', 0)
    
    enhanced_summary = f"{translated_summary}\n\nKey Details: {num_events} timeline events from {num_sources} sources with {auth_score}% authenticity.\nJourney: {first_event} → ... → {last_event}"
    
    st.markdown(f'<div class="summary-box"><p class="summary-text">{enhanced_summary}</p></div>', unsafe_allow_html=True)
    
    
    # Interactive Charts - No header, just button
    if timeline_items:
        if st.button("📈 Show Interactive Charts"):
            fig = create_interactive_timeline(timeline_items)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            fig2 = create_confidence_distribution(timeline_items)
            if fig2:
                st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Timeline Display
    st.markdown('<div class="section-header">🕐 Detailed Timeline</div>', unsafe_allow_html=True)
    
    if timeline_items:
        for idx, item in enumerate(timeline_items):
            date = item.get('date', 'Unknown Date')
            event = item.get('event', 'No description')
            confidence = item.get('confidence', 0) * 100
            sources = item.get('sources', [])
            
            st.markdown(f"""
            <div class="timeline-item">
                <div class="timeline-date">📅 {date}</div>
                <div class="timeline-event">{event}</div>
                <div class="timeline-meta">
                    <span>📊 Confidence: {confidence:.1f}%</span>
                    <span>📰 Sources: {len(sources)}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="no-data">🕐 No timeline events available</div>', unsafe_allow_html=True)

    # Entities & Sources Panel - Improved UI
    st.markdown('<div class="section-header">🔎 Entities & Sources</div>', unsafe_allow_html=True)
    entities = extract_entities(timeline_items)
    
    # Create a better layout
    entity_col, source_col = st.columns([1, 1])
    
    with entity_col:
        st.markdown("<h3 style='color: #2c3e50; margin-top: 0;'>🎯 Key Entities</h3>", unsafe_allow_html=True)
        
        dates = entities.get('dates',[])
        persons = entities.get('persons',[])
        locations = entities.get('locations',[])
        orgs = entities.get('organizations',[])
        outcomes = entities.get('outcomes',[])
        
        # Display entities with better formatting
        if dates:
            st.markdown(f"<div style='margin: 12px 0;'><strong>📅 Dates</strong> ({len(dates)})<br>{', '.join(dates[:8])}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='margin: 12px 0; color: #999;'><strong>📅 Dates</strong> (0) - None extracted</div>", unsafe_allow_html=True)
            
        if persons:
            st.markdown(f"<div style='margin: 12px 0;'><strong>👥 Persons</strong> ({len(persons)})<br>{', '.join(persons[:8])}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='margin: 12px 0; color: #999;'><strong>👥 Persons</strong> (0) - None extracted</div>", unsafe_allow_html=True)
            
        if locations:
            st.markdown(f"<div style='margin: 12px 0;'><strong>🌍 Locations</strong> ({len(locations)})<br>{', '.join(locations[:8])}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='margin: 12px 0; color: #999;'><strong>🌍 Locations</strong> (0) - None extracted</div>", unsafe_allow_html=True)
            
        if orgs:
            st.markdown(f"<div style='margin: 12px 0;'><strong>🏢 Organizations</strong> ({len(orgs)})<br>{', '.join(orgs[:8])}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='margin: 12px 0; color: #999;'><strong>🏢 Organizations</strong> (0) - None extracted</div>", unsafe_allow_html=True)
            
        if outcomes:
            st.markdown(f"<div style='margin: 12px 0;'><strong>✅ Outcomes</strong> ({len(outcomes)})<br>{', '.join(outcomes[:8])}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='margin: 12px 0; color: #999;'><strong>✅ Outcomes</strong> (0) - None extracted</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with source_col:
        st.markdown("<h3 style='color: #2c3e50; margin-top: 0;'>📰 News Sources</h3>", unsafe_allow_html=True)
        
        # show unique source links and credibility
        seen = set()
        count = 0
        for a in articles:
            url = a.get('url') or ''
            title = a.get('title') or url
            src = a.get('source') or ''
            if url and url not in seen:
                seen.add(url)
                count += 1
                try:
                    from backend.utils import get_source_tier
                    tier_score = get_source_tier(url)
                    if tier_score == 3:
                        tier_badge = '<span style="background: #10b981; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold;">Tier 1</span>'
                    elif tier_score == 2:
                        tier_badge = '<span style="background: #f59e0b; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold;">Tier 2</span>'
                    elif tier_score == 1:
                        tier_badge = '<span style="background: #3b82f6; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold;">Tier 3</span>'
                    else:
                        tier_badge = '<span style="background: #999; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px;">Unrated</span>'
                except:
                    tier_badge = '<span style="background: #999; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px;">Unrated</span>'
                    
                st.markdown(f"<div style='padding: 10px 0; border-bottom: 1px solid #f0f0f0;'><div style='font-size: 13px; margin-bottom: 4px;'><a href='{url}' target='_blank' style='color: #111; text-decoration: none; font-weight: 600;'>{title[:50]}</a></div><div style='font-size: 12px; color: #666;'>{src}</div><div style='margin-top: 4px;'>{tier_badge}</div></div>", unsafe_allow_html=True)
                if count >= 8:
                    break
        
        remaining = len(set(a.get('url') for a in articles if a.get('url'))) - count
        if remaining > 0:
            st.markdown(f"<div style='padding: 10px 0; color: #999; font-size: 12px;'>+{remaining} more sources</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Authenticity Analysis with detailed metrics breakdown
    st.markdown('<div class="section-header">📊 Score Breakdown (Weighted Factors)</div>', unsafe_allow_html=True)
    auth_score_data = result.get('authenticity_score', {})
    if isinstance(auth_score_data, dict):
        score = auth_score_data.get('overall_score', 0)
        grade = auth_score_data.get('grade', '?')
        breakdown = auth_score_data.get('breakdown', {})
        
        
        # Detailed metrics breakdown
        if breakdown:
            
            metrics = [
                ('confidence', '🎯 Extraction Confidence', 'Average confidence from AI extraction', 0.25),
                ('source_diversity', '📰 Source Diversity', 'Variety of news sources consulted', 0.20),
                ('cross_verification', '✓ Cross-Verification', 'Events confirmed by multiple sources', 0.25),
                ('temporal_consistency', '⏱️ Temporal Consistency', 'Logical chronological ordering', 0.15),
                ('source_quality', '⭐ Source Quality', 'Reliability of cited sources', 0.15)
            ]
            
            cols = st.columns(5)
            for idx, (key, label, desc, weight) in enumerate(metrics):
                val = breakdown.get(key, 0)
                # Color coding: green > 75, yellow 50-75, red < 50
                if val >= 75:
                    color = '#10b981'
                    bar_color = '#d1fae5'
                elif val >= 50:
                    color = '#f59e0b'
                    bar_color = '#fef3c7'
                else:
                    color = '#ef4444'
                    bar_color = '#fee2e2'
                
                with cols[idx]:
                    st.markdown(f"""
                    <div style='background: {bar_color}; padding: 15px; border-radius: 8px; border-left: 4px solid {color};'>
                        <div style='font-size: 14px; color: #2c3e50; font-weight: 600;'>{label}</div>
                        <div style='font-size: 28px; color: {color}; font-weight: 800; margin: 8px 0;'>{val}</div>
                        <div style='font-size: 11px; color: #666;'>{desc}</div>
                        <div style='font-size: 10px; color: #999; margin-top: 4px;'>Weight: {weight*100:.0f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    # Turning Points
    st.markdown('<div class="section-header">🔄 Critical Turning Points</div>', unsafe_allow_html=True)
    
    turning_points = result.get('turning_points', [])
    if turning_points:
        for tp in turning_points:
            event = tp.get('event', 'Unknown event')
            reason = tp.get('reason', 'No reason provided')
            st.markdown(f'<div class="turning-point"><div class="turning-point-title">🎯 {event}</div><div class="turning-point-reason">{reason}</div></div>', unsafe_allow_html=True)
    else:
        st.info("ℹ️ No major turning points identified")
    
    # Discrepancies
    st.markdown('<div class="section-header">⚠️ Source Discrepancies</div>', unsafe_allow_html=True)
    
    discrepancies = result.get('discrepancies', [])
    if discrepancies:
        # show only top 4 discrepancy items to avoid overwhelming the UI
        for disc in (discrepancies[:4]):
            issue = disc.get('issue', 'Unknown issue')
            explanation = disc.get('explanation', 'No explanation')
            # compact source count (do not show full source list)
            srcs = disc.get('sources', []) or []
            try:
                src_count = len(srcs) if isinstance(srcs, (list, tuple, set)) else 0
            except Exception:
                src_count = 0

            count_html = ''
            if src_count > 0:
                count_html = f"<div style='font-size:12px; color:#666; margin-top:6px;'>Affects {src_count} source{'s' if src_count!=1 else ''}</div>"

            # Render only the issue, explanation, and compact count; source details are omitted by design
            st.markdown(f'<div class="discrepancy-box"><div class="discrepancy-title">⚠️ {issue}</div><div class="discrepancy-text">{explanation}</div>{count_html}</div>', unsafe_allow_html=True)
        remaining = max(0, len(discrepancies) - 4)
        if remaining > 0:
            st.markdown(f"<div style='font-size:12px; color:#666; margin-top:8px;'>+{remaining} more discrepancy items (hidden)</div>", unsafe_allow_html=True)
    else:
        st.success("✅ No major discrepancies detected across sources")

    # Consistency Issues (numeric/date/negation conflicts)
    st.markdown('<div class="section-header">🔍 Fact Consistency Checker</div>', unsafe_allow_html=True)
    consistency_issues = result.get('consistency_issues', [])
    if consistency_issues:
        for ci in consistency_issues:
            tp = ci.get('type', 'issue')
            ex = ci.get('examples', [])
            details = ci.get('details', {})
            st.markdown(f"<div style='background:#fff7ed; padding:12px; border-radius:8px; margin-bottom:8px;'><strong>{tp.replace('_',' ').title()}</strong><div style='color:#333; margin-top:6px;'>{ex[0] if ex else ''}</div><div style='color:#555; font-size:13px; margin-top:6px;'>{details}</div></div>", unsafe_allow_html=True)
    else:
        st.success("✅ No obvious factual contradictions detected")

    # Bias / Clickbait Flags
    st.markdown('<div class="section-header">⚖️ Bias & Clickbait Detection</div>', unsafe_allow_html=True)
    # run detectors on the combined summary and top headlines
    combined_text_for_check = result.get('summary', '') + '\n' + '\n'.join([a.get('title','') for a in articles[:6]])
    subj = detect_subjective_tone(combined_text_for_check)
    click = None
    if articles:
        top_title = articles[0].get('title','')
        click = detect_clickbait_headline(top_title)

    st.markdown(f"<div style='background:white; padding:12px; border-radius:8px;'><strong>Subjectivity Score:</strong> {subj.get('score',0)}<br><strong>Flags:</strong> {', '.join(subj.get('flags',[]))}</div>", unsafe_allow_html=True)
    if click:
        st.markdown(f"<div style='background:white; padding:12px; border-radius:8px; margin-top:8px;'><strong>Top Headline Clickbait Score:</strong> {click.get('score',0)}<br><strong>Patterns:</strong> {', '.join(click.get('flags',[]))}</div>", unsafe_allow_html=True)

    # Save / Dashboard controls
    st.markdown('<div class="section-header">📅 Daily Event Tracker Dashboard</div>', unsafe_allow_html=True)
    dash_col1, dash_col2 = st.columns([1,1])
    with dash_col1:
        if st.button('💾 Save Timeline to History'):
            path = save_timeline_history(result, query=query)
            if path:
                st.success(f"Saved to history: {path}")
            else:
                st.error("Failed to save timeline to history")
    with dash_col2:
        if st.button('🔄 Auto-save Daily (save now)'):
            path = save_timeline_history(result, query=query)
            if path:
                st.success(f"Auto-saved: {path}")
            else:
                st.error("Auto-save failed")

    # Show recent history entries
    history = load_timeline_history(limit=20)
    if history:
        st.markdown('<div style="background:white; padding:12px; border-radius:8px;">', unsafe_allow_html=True)
        st.markdown('<strong>Recent Saved Timelines</strong>', unsafe_allow_html=True)
        for h in history[:10]:
            saved_at = h.get('saved_at') or h.get('saved_at', '')
            q = h.get('query') or ''
            title = f"{saved_at} — {q[:60]}"
            with st.expander(title):
                st.json(h.get('result', {}))
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info('No saved timelines yet — use "Save Timeline to History" to build your daily dashboard')
    
    # Raw Data Expanders
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🔍 Raw Data (Advanced)</div>', unsafe_allow_html=True)
    
    with st.expander("📄 View Articles Data"):
        st.json(articles)
    
    with st.expander("📋 View Extracted Events"):
        st.json(events)
    
    with st.expander("🗂️ View Complete Timeline"):
        st.json(result)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:gray; font-size:14px;">
    Made with ❤️ by <b>Niladri Giri</b><br>
    <span style="font-size:12px;">Powered by Streamlit & AI</span>
</div>
""", unsafe_allow_html=True)