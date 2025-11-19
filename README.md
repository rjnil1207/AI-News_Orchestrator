# 🚀 AI News Orchestrator

A Streamlit app that aggregates news articles, extracts events using AI heuristics/LLMs, and generates an interactive timeline with authenticity scoring and daily tracking.

## 📌 Overview

**AI News Orchestrator** is a Streamlit-based application that transforms news articles into a **chronological, AI-generated event timeline** with summaries, highlights, turning points, discrepancies, and authenticity scoring.

## ✨ Key Features

### 🔍 Topic Search
Search any ongoing event/topic and fetch curated multi-source news.

### 📰 Trending News
Displays real-time global headlines using `fetch_top_headlines()`.

### 📡 Multi-Source Aggregation
Fetches articles from GDELT & GNews to ensure diversity and reduce bias.

### 🧠 AI Event Extraction
Extracts structured events using LLMs (default: `llama-3.1-8b-instant`).

### 🕒 Timeline Generation
Creates a clean chronological timeline with key metadata.

### 🌐 Multi-Language Summary
Supports translation (English, Hindi, Bengali, Tamil, Telugu, Marathi).

### 🖼 Modern UI
Glassmorphic interface with smooth animations and rounded input components.

### 🧠 Workflow
1. User enters a topic
↓
2. App fetches news from multiple APIs
↓
3. Events extracted using LLM
↓
4. Timeline constructed & validated
↓
5. Summary generated + translated
↓
6. UI displays:

## ⚙️ Configuration

| Parameter | Purpose | Default |
|----------|---------|---------|
| num_articles | Max articles | 10 |
| model_choice | LLM used | llama-3.1-8b-instant |
| use_cache | Cache toggle | True |
| batch_delay | Delay per call | 2 sec |
| use_gdelt | Enable GDELT | True |
| use_gnews | Enable GNews | True |

## 🔑 Environment Variables

GNEWS_API_KEY=your_key_here
GROQ_API_KEY=your_key_here

## 🧪 Running the App

pip install -r requirements.txt
streamlit run main_app.py

## ✨ Credits
Developed by: Niladri Giri
Powered by: Streamlit, Groq LLM, GDELT, GNews