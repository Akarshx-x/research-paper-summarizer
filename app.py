"""
app.py
======
Self-Supervised PyTorch Classifier · Dense-RAG Paper Chat — Update 8
Streamlit UI · REST-Based Gemini · Sidebar Mode Selector · Offline Guide

Changes from Update 7:
  • _call_llm now speaks directly to the Gemini REST endpoint via `requests`
    instead of the google-generativeai SDK, with a strict 8s timeout so a
    stalled network call can never freeze the UI. Failure/exception
    contract is unchanged (raises → caller falls back to extractive output
    and shows a warning), so status badges stay accurate.
  • New: a self-supervised PyTorch MLP (supervised_classifier.py) trains
    fresh per document on its own SciBERT embeddings and renders a real
    accuracy + train/validation loss-curve panel. Its training labels are
    generated from embedding geometry, not human-annotated — see
    supervised_classifier.py's module docstring for why that distinction
    matters and how it's kept honest in the UI copy below.
  • New: a "RAG: Chat with PDF" tab — pure NumPy dense retrieval (cosine
    similarity, no FAISS/Chroma) over the document's own sentence
    embeddings, feeding the top-3 matches to Gemini as restricted context.
  • The unsupervised extractive-abstractive summarizer (SciBERT + K-Means +
    optional Gemini polish) is unchanged and still the primary summary
    engine — the two additions above are new capabilities layered next to
    it, not replacements. "RAG" in this file refers specifically to the
    chat feature, which is genuinely retrieval-then-generation; the K-Means
    summarizer is still accurately described as unsupervised extractive
    clustering, not RAG.

Caching architecture:
  @st.cache_resource  SciBERT model         — once per server process, shared globally
  @st.cache_data      run_pipeline          — keyed by raw PDF bytes (text + embeddings)
  @st.cache_data      run_clustering        — keyed by (embeddings, mode)
  @st.cache_data      run_supervised_training — keyed by embeddings
  @st.cache_data      _call_llm             — keyed by (sentences,); API key excluded
  @st.cache_data      _call_rag_chat        — keyed by (question, context); API key excluded

Run:
    streamlit run app.py

Secrets setup (local):
    mkdir -p .streamlit
    touch .streamlit/secrets.toml
    # then add:  GEMINI_API_KEY = "AIza..."
    Cloud: add GEMINI_API_KEY in the Streamlit Cloud secrets dashboard.
"""

import hashlib
import os
import logging
import tempfile
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from rouge_score import rouge_scorer

from paper_preprocessor import DocumentIngestionPipeline
from update_2_embeddings import (
    ScientificSentenceEmbedder,
    EmbeddingConfig,
    EmbeddingResult,
)
from update_3_clustering import generate_summary, SummaryResult
from supervised_classifier import (
    TrainingResult,
    train_salience_classifier,
    MIN_SENTENCES_FOR_TRAINING,
)


logging.disable(logging.CRITICAL)


st.set_page_config(
    page_title="ResearchLens · Self-Supervised RAG & PyTorch Summarizer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── Base ─────────────────────────────────────────────────────────────────── */
html, body, [data-testid="stApp"] { background-color: #0f1117; }
[data-testid="stSidebar"] {
    background-color: #1a1d27;
    border-right: 1px solid #252836;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }
section.main > div { padding-top: 1.5rem; padding-bottom: 3rem; }

/* ── Metric cards ─────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #1e2130;
    border: 1px solid #252836;
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    transition: border-color 0.18s ease;
}
[data-testid="stMetric"]:hover { border-color: #4f8ef7; }
[data-testid="stMetricLabel"] p {
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #6c7a9c !important;
}
[data-testid="stMetricValue"] {
    color: #e6f1ff !important;
    font-size: 1.55rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricDelta"] > div { font-size: 0.76rem !important; }

/* ── Single prose chat bubble ─────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: #1e2130 !important;
    border: 1px solid #252836 !important;
    border-radius: 14px !important;
    padding: 1.4rem 1.6rem !important;
    margin-bottom: 0.6rem !important;
}
[data-testid="stChatMessage"] p {
    color: #ccd6f6 !important;
    line-height: 1.9 !important;
    font-size: 0.95rem !important;
    margin: 0 !important;
}
[data-testid="stChatMessageAvatarContainer"] {
    background: #252836 !important;
    border-radius: 8px !important;
}

/* ── File uploader ────────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #252836;
    border: 1px dashed #3d4468;
    border-radius: 10px;
    padding: 0.5rem;
}
[data-testid="stFileUploader"]:hover { border-color: #4f8ef7; }

/* ── Sidebar mode radio (vertical pill list) ──────────────────────────────── */
[data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    padding: 0.1rem 0;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label {
    background: #1e2130;
    border: 1px solid #3d4468;
    border-radius: 8px;
    padding: 0.45rem 1rem;
    cursor: pointer;
    font-size: 0.82rem;
    font-weight: 600;
    color: #6c7a9c;
    transition: all 0.15s ease;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
    border-color: #4f8ef7;
    color: #ccd6f6;
    background: #1a2540;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
    background: #1a3069;
    border-color: #4f8ef7;
    color: #4f8ef7;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {
    display: none;
}

/* ── Section labels ───────────────────────────────────────────────────────── */
.rl-section-label {
    font-size: 0.67rem;
    font-weight: 800;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #4f8ef7;
    margin: 1.4rem 0 0.7rem 0;
}

/* ── Sidebar badges ───────────────────────────────────────────────────────── */
.rl-badge {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.82rem;
    color: #6c7a9c;
    padding: 0.2rem 0;
    font-weight: 500;
    line-height: 1.5;
}
.rl-badge.done    { color: #64ffda; }
.rl-badge.running { color: #ffd166; }
.rl-badge.error   { color: #ef476f; }
.rl-badge.offline { color: #3d4468; }

/* ── Meta strip ───────────────────────────────────────────────────────────── */
.rl-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 1.4rem;
    font-size: 0.78rem;
    color: #6c7a9c;
    margin-bottom: 1.1rem;
    padding: 0.75rem 1rem;
    background: #161824;
    border-radius: 8px;
    border: 1px solid #252836;
}
.rl-meta b { color: #4f8ef7; }
.rl-mode-pill {
    background: #1a3069;
    border: 1px solid #4f8ef7;
    color: #4f8ef7;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    border-radius: 4px;
    padding: 0.1rem 0.5rem;
    text-transform: uppercase;
    margin-left: auto;
}

/* ── Download button ──────────────────────────────────────────────────────── */
[data-testid="stDownloadButton"] button {
    margin-top: 0.75rem;
    background: transparent !important;
    border: 1px solid #3d4468 !important;
    color: #8892b0 !important;
    font-size: 0.8rem !important;
    padding: 0.35rem 0.9rem !important;
    border-radius: 6px !important;
    transition: all 0.15s ease;
}
[data-testid="stDownloadButton"] button:hover {
    border-color: #4f8ef7 !important;
    color: #4f8ef7 !important;
    background: #0d1b3e !important;
}

/* ── Welcome card ─────────────────────────────────────────────────────────── */
.rl-welcome {
    max-width: 640px;
    margin: 2.5rem auto;
    background: #1e2130;
    border: 1px solid #252836;
    border-radius: 18px;
    padding: 2.5rem 3rem;
    text-align: center;
}
.rl-welcome h1 { color: #e6f1ff; font-size: 1.8rem; margin-bottom: 0.4rem; font-weight: 700; }
.rl-welcome .rl-tagline { color: #6c7a9c; font-size: 0.88rem; line-height: 1.65; margin-bottom: 2rem; }
.rl-step { display: flex; align-items: flex-start; gap: 0.85rem; padding: 0.7rem 0; border-bottom: 1px solid #252836; text-align: left; }
.rl-step:last-child { border-bottom: none; }
.rl-step-num {
    flex-shrink: 0; width: 1.75rem; height: 1.75rem;
    background: #252836; border: 1px solid #3d4468; border-radius: 50%;
    color: #4f8ef7; font-size: 0.72rem; font-weight: 800;
    display: flex; align-items: center; justify-content: center; margin-top: 0.1rem;
}
.rl-step-text { font-size: 0.85rem; color: #8892b0; line-height: 1.6; }
.rl-step-text b { color: #ccd6f6; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# GEMINI — DIRECT REST TRANSPORT
# ═══════════════════════════════════════════════════════════════════════════════

_GEMINI_MODEL    = "gemini-2.5-flash"
_GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{_GEMINI_MODEL}:generateContent"


def _gemini_rest_call(system_prompt: str, user_content: str, api_key: str, timeout: int = 8) -> str:
    """
    Direct HTTPS POST to the Gemini Generative Language API — deliberately
    bypasses the google-generativeai SDK, whose default transport is gRPC.
    On networks that block long-lived HTTP/2 streams, the SDK's gRPC calls
    hang indefinitely; plain REST calls to the same API succeed in seconds.

    `timeout` is enforced by `requests` itself, so a stalled network call
    can never freeze the Streamlit UI — it raises requests.Timeout instead.
    Raises on any failure (timeout, HTTP error, malformed response); callers
    decide how to fall back (see _call_llm and _call_rag_chat, which fall
    back differently because they have different failure-visibility needs).

    thinkingBudget=0 disables gemini-2.5-flash's default extended-reasoning
    pass. Measured directly: the same real polish prompt took >8s (timed
    out) with thinking on and 1.4s with it off. Neither task here (copyedit
    fusion, short-context Q&A) benefits from multi-step reasoning, so this
    is a pure latency win, not a quality tradeoff.

    Authenticates via the `x-goog-api-key` header, NOT `?key=...` in the URL
    — verified directly (429 vs 401 response) that the API accepts this.
    The query-param form was tried first and found to leak the raw key:
    requests.HTTPError's string repr includes the full request URL, and
    that string reaches st.warning() on failure (see main()'s except
    branch), meaning a query-param key would render in plaintext in the
    browser on any failed call. The header form never appears in that
    string, so a failure can be shown to the user without exposing the key.
    """
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_content}]}],
        "generationConfig": {"thinkingConfig": {"thinkingBudget": 0}},
    }
    response = requests.post(
        _GEMINI_ENDPOINT,
        headers={"x-goog-api-key": api_key},
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


# ═══════════════════════════════════════════════════════════════════════════════
# CACHING GATES
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_embedder() -> ScientificSentenceEmbedder:
    """SciBERT loaded once per server process; shared across all sessions."""
    return ScientificSentenceEmbedder(EmbeddingConfig())


@st.cache_data(show_spinner=False)
def run_pipeline(file_bytes: bytes) -> dict:
    """
    Stages 1 + 2: PDF extraction + SciBERT embedding.
    Keyed by raw PDF bytes — same file = instant cache hit, no re-embedding.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        pipeline = DocumentIngestionPipeline()
        doc = pipeline.run(tmp_path)
    finally:
        os.unlink(tmp_path)

    if not doc.sentences:
        raise ValueError(
            "No readable text extracted. The PDF may be a scanned image — "
            "run OCR first (Adobe Acrobat, Tesseract)."
        )
    if len(doc.sentences) < 4:
        raise ValueError(
            f"Only {len(doc.sentences)} sentence(s) extracted (minimum 4). "
            "The PDF may be a cover page or abstract-only submission."
        )

    embedder   = load_embedder()
    embeddings = embedder.embed(doc.sentences)

    return {
        "sentences":     doc.sentences,
        "embeddings":    embeddings,
        "raw_chars":     len(doc.joined_text),
        "cleaned_chars": len(doc.cleaned_text),
        "n_sentences":   len(doc.sentences),
        "device":        str(embedder.device),
        "model":         EmbeddingConfig().model_name,
        "abstract":      doc.abstract,
    }


@st.cache_data(show_spinner=False)
def run_clustering(
    sentences:   tuple,
    embeddings:  np.ndarray,
    mode:        str,
    model_name:  str,
    device_used: str,
) -> SummaryResult:
    """
    Stage 3: K-Means on cached embeddings. Fast (~seconds).
    All three modes are pre-computed on upload; switching modes is instant.
    sentences: tuple for stable Streamlit hash key derivation.
    """
    emb_result = EmbeddingResult(
        sentences=list(sentences),
        embeddings=embeddings,
        model_name=model_name,
        device_used=device_used,
    )
    return generate_summary(emb_result, mode=mode)


@st.cache_data(show_spinner=False)
def run_supervised_training(embeddings: np.ndarray) -> TrainingResult:
    """
    Stage 5: self-supervised PyTorch MLP training (supervised_classifier.py).
    Keyed by embeddings, so switching summary length mode or asking a chat
    question doesn't retrain — only a genuinely new document does.
    """
    return train_salience_classifier(embeddings)


@st.cache_data(show_spinner=False)
def run_rag_embeddings(sentences: tuple) -> np.ndarray:
    """
    Mean-pooled embeddings for RAG retrieval ONLY — separate from the
    [CLS]-pooled `data["embeddings"]` used everywhere else (K-Means,
    supervised classifier). See ScientificSentenceEmbedder.embed_mean_pooled
    for why: raw [CLS] vectors retrieve poorly for short natural-language
    questions against long-form technical prose. Cached per-document
    (keyed by sentences) so it's computed once on upload, not per question.
    """
    embedder = load_embedder()
    return embedder.embed_mean_pooled(list(sentences))


@st.cache_data(show_spinner=False)
def compute_rouge(summary_sentences: tuple, reference: str) -> dict:
    """
    ROUGE-1 / ROUGE-2 / ROUGE-L F1 of the extractive summary against the
    paper's own Abstract, used as a reference/gold-standard proxy since no
    human-written gold summary exists for an arbitrary uploaded PDF.

    Returns {} when no abstract was detected (see paper_preprocessor.extract_abstract),
    so callers must treat this as optional, not guaranteed.
    """
    if not reference:
        return {}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    hypothesis = " ".join(summary_sentences)
    scores = scorer.score(reference, hypothesis)
    return {name: result.fmeasure for name, result in scores.items()}


@st.cache_data(show_spinner=False)
def _call_llm(sentences: tuple, _api_key: str) -> str:
    """
    Optional LLM abstractive polish layer (Stage 4) — Gemini only, direct REST.

    _api_key is loaded from st.secrets["GEMINI_API_KEY"] by the caller.
    The underscore prefix tells Streamlit to exclude it from the cache key,
    so the same sentence set always hits the same cache entry regardless of
    which API key authenticated the call.

    Returns " ".join(sentences) when _api_key is empty (offline fallback).
    Exceptions propagate to the caller (main()) so failed calls are NOT
    cached (the next render will retry the API call) and so main() can show
    an accurate "LLM Polish" status badge + warning instead of silently
    mislabeling a fallback as "polished".
    """
    if not _api_key:
        return " ".join(sentences)

    system_prompt = (
        "You are an elite scientific copyeditor. Your sole task is to synthesize "
        "the following extracted ground-truth sentences into a fluid, beautifully "
        "coherent, continuous academic paragraph summary. You must fix any spacing "
        "or formatting typos present in the extraction stream. Do NOT add any outside "
        "facts, assumptions, or external details. Your output must contain zero bullet "
        "points and zero sentence indexes—just continuous prose."
    )
    user_content = "Extracted sentences:\n" + "\n".join(
        f"{i}. {s}" for i, s in enumerate(sentences, 1)
    )
    return _gemini_rest_call(system_prompt, user_content, _api_key, timeout=8)


@st.cache_data(show_spinner=False)
def _call_rag_chat(question: str, context_sentences: tuple, _api_key: str) -> str:
    """
    RAG answer generation for the "Chat with PDF" tab: given a user question
    and the top-k retrieved sentences (see cosine_topk), asks Gemini to
    answer using ONLY that context. Same REST transport + 8s timeout as
    _call_llm, but this function swallows failures internally and falls
    back to surfacing the raw retrieved sentences — a chat reply degrading
    to "here's what I found" is a better UX than a stack trace mid-conversation,
    and there's no "polished" status badge here that a silent fallback could
    mislabel (unlike _call_llm).
    """
    context_block = "\n".join(f"- {s}" for s in context_sentences)
    if not _api_key:
        return "No GEMINI_API_KEY configured — showing retrieved context directly:\n\n" + context_block

    system_prompt = (
        "You are a research-paper assistant answering questions about a single "
        "uploaded PDF. Answer using ONLY the retrieved context sentences below. "
        "If the context does not contain the answer, say so explicitly rather "
        "than guessing or using outside knowledge."
    )
    user_content = f"Retrieved context:\n{context_block}\n\nQuestion: {question}"

    try:
        return _gemini_rest_call(system_prompt, user_content, _api_key, timeout=8)
    except (requests.RequestException, KeyError, IndexError, ValueError):
        return "Gemini request failed or timed out — showing retrieved context directly:\n\n" + context_block


# ═══════════════════════════════════════════════════════════════════════════════
# RAG RETRIEVAL — PURE NUMPY, NO VECTOR DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

def cosine_topk(
    query_vector:        np.ndarray,
    sentence_embeddings: np.ndarray,
    sentences:           List[str],
    k:                   int = 3,
) -> List[Tuple[str, float]]:
    """
    Dense retrieval via a full in-memory cosine-similarity scan — no
    FAISS/Chroma. At single-document scale (tens to low thousands of
    sentences) a NumPy dot-product pass is microseconds; a vector database
    would be pure overhead here, and both FAISS and Chroma carry native
    C++ build dependencies that are a common source of failed installs on
    free-tier cloud deployment (the exact "runs locally, breaks in the
    cloud" failure mode this update is meant to avoid).

    Score_i = (V_query · V_sentence_i) / (‖V_query‖ · ‖V_sentence_i‖)
    """
    query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)
    emb_norm = sentence_embeddings / (
        np.linalg.norm(sentence_embeddings, axis=1, keepdims=True) + 1e-8
    )
    scores = emb_norm @ query_norm
    top_idx = np.argsort(-scores)[:k]
    return [(sentences[i], float(scores[i])) for i in top_idx]


# ═══════════════════════════════════════════════════════════════════════════════
# RENDERING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

_STATUS_ICONS = {"pending": "⬜", "running": "🔄", "done": "✅", "error": "❌", "offline": "◻️"}

_MODE_LABELS = {
    "⚡  Short — 4 sentences":       "short",
    "📄  1-Page — 11 sentences":     "one_page",
    "📚  Full — 15% extraction":     "full",
}

_MODE_DISPLAY = {"short": "Short", "one_page": "1-Page", "full": "Full"}


def _badge(label: str, status: str) -> str:
    icon = _STATUS_ICONS.get(status, "⬜")
    css  = status if status in ("done", "running", "error", "offline") else ""
    return f"<div class='rl-badge {css}'>{icon} {label}</div>"


def render_welcome():
    st.markdown("""
    <div class="rl-welcome">
      <h1>🔬 ResearchLens</h1>
      <p class="rl-tagline">
        Upload any academic PDF to generate a <b>zero-hallucination</b>
        extractive summary powered by <b>SciBERT</b> embeddings and
        unsupervised <b>K-Means</b> clustering — with optional
        <b>Gemini abstractive polish</b>, a <b>self-supervised PyTorch</b>
        neural classifier validating sentence salience, and a
        <b>Dense-RAG</b> chat interface for asking direct questions about
        the paper.
      </p>
      <div class="rl-step">
        <div class="rl-step-num">1</div>
        <div class="rl-step-text">
          <b>Column-aware PDF extraction</b> — word bounding-box reordering
          corrects two-column layout interleaving. 8-step regex cascade removes
          citations, math, and page artifacts with word-boundary protection.
        </div>
      </div>
      <div class="rl-step">
        <div class="rl-step-num">2</div>
        <div class="rl-step-text">
          <b>SciBERT sentence embedding</b> — 768-dim CLS vectors trained on
          1.14M scientific papers. L2-normalised for cosine K-Means.
        </div>
      </div>
      <div class="rl-step">
        <div class="rl-step-num">3</div>
        <div class="rl-step-text">
          <b>K-Means semantic clustering</b> — centroid-closest sentence per
          cluster, re-sorted into document order for narrative coherence.
          Select length in the sidebar; all three modes are pre-computed.
        </div>
      </div>
      <div class="rl-step">
        <div class="rl-step-num">4</div>
        <div class="rl-step-text">
          <b>Gemini abstractive polish</b> — synthesizes extracted sentences
          into fluent prose via <code>gemini-2.5-flash</code>, called directly
          over REST with a hard 8-second timeout. Auto-loaded from
          <code>st.secrets["GEMINI_API_KEY"]</code>; skipped gracefully offline.
        </div>
      </div>
      <div class="rl-step">
        <div class="rl-step-num">5</div>
        <div class="rl-step-text">
          <b>Self-supervised PyTorch classifier</b> — a 3-layer MLP trains
          fresh per document (15 epochs, AdamW / BCELoss) against weak
          salience labels derived from embedding geometry, reporting real
          train/validation loss curves and held-out accuracy.
        </div>
      </div>
      <div class="rl-step">
        <div class="rl-step-num">6</div>
        <div class="rl-step-text">
          <b>Dense-RAG paper chat</b> — ask a direct question; it's embedded
          with the same SciBERT model, matched via pure-NumPy cosine
          similarity against every sentence in the document, and the
          top-3 retrieved sentences ground Gemini's answer.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_metrics(data: dict):
    """
    2x2 grid, not 1x4: this panel lives inside the narrower left column of
    the dual-column dashboard, and four metric cards side by side there
    truncated both labels ("RAW CH...") and values ("39,...") — confirmed
    visually, not hypothetical. Two per row gives each card roughly double
    the width; labels are also shortened and full detail moved into a
    tooltip rather than the visible label, so nothing gets cut off again
    if the column narrows further.
    """
    raw       = data["raw_chars"]
    cleaned   = data["cleaned_chars"]
    noise     = raw - cleaned
    noise_pct = noise / max(raw, 1)

    row1_c1, row1_c2 = st.columns(2)
    row1_c1.metric("Raw Chars", f"{raw:,}", help="Total characters extracted from the PDF before cleaning.")
    row1_c2.metric("Sentences", f"{data['n_sentences']:,}", help="Sentences kept after quality filtering.")

    row2_c1, row2_c2 = st.columns(2)
    row2_c1.metric(
        "Noise Removed",
        f"{noise:,}",
        delta=f"-{noise_pct:.0%}",
        delta_color="inverse",
        help=f"{noise:,} characters ({noise_pct:.0%} of raw text) removed by the cleaning regex cascade.",
    )
    row2_c2.metric("Device", data["device"].upper(), help="Inference device used for SciBERT embedding.")


def render_quality_metrics(result: SummaryResult, rouge_scores: dict):
    """
    Quantitative quality metrics for the unsupervised clustering pipeline:
    clustering coherence (silhouette) needs no ground truth; ROUGE needs
    the paper's Abstract as a reference and is only shown when detected.
    """
    st.markdown(
        "<div class='rl-section-label'>Quality Metrics — Unsupervised Pipeline</div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    silhouette = result.silhouette
    cols[0].metric(
        "Silhouette Score",
        f"{silhouette:.3f}" if silhouette is not None else "n/a",
        help="Cluster separation/coherence of the K-Means step, -1 to 1 (higher is better). Needs no ground truth.",
    )
    if rouge_scores:
        cols[1].metric("ROUGE-1 (F1)", f"{rouge_scores['rouge1']:.3f}",
                        help="Unigram overlap with the paper's own Abstract, used as a reference summary.")
        cols[2].metric("ROUGE-2 (F1)", f"{rouge_scores['rouge2']:.3f}",
                        help="Bigram overlap with the paper's own Abstract.")
        cols[3].metric("ROUGE-L (F1)", f"{rouge_scores['rougeL']:.3f}",
                        help="Longest common subsequence overlap with the paper's own Abstract.")
    else:
        cols[1].metric("ROUGE Scores", "n/a", help="No Abstract section could be detected in this PDF to use as a reference.")


def render_supervised_panel(embeddings: np.ndarray) -> Optional[TrainingResult]:
    """
    Stage 5 UI: trains (or retrieves from cache) the self-supervised MLP for
    the current document and renders a real accuracy + loss-curve panel.

    Returns the TrainingResult so main() can drive an accurate sidebar
    status badge, or None if the document is too short to train/validate
    meaningfully (see MIN_SENTENCES_FOR_TRAINING) — short documents show an
    explanatory st.info instead of a number computed on a handful of
    validation examples.
    """
    st.markdown(
        "<div class='rl-section-label'>PyTorch Supervised Validation</div>",
        unsafe_allow_html=True,
    )

    n_sentences = embeddings.shape[0]
    if n_sentences < MIN_SENTENCES_FOR_TRAINING:
        st.info(
            f"Document has only {n_sentences} sentences — need at least "
            f"{MIN_SENTENCES_FOR_TRAINING} for a meaningful train/validation split. "
            "Supervised panel skipped for this document."
        )
        return None

    with st.spinner("Training PyTorch MLP classifier (15 epochs)…"):
        training = run_supervised_training(embeddings)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(
            "Supervised Classifier Accuracy",
            f"{training.val_accuracy:.1f}%",
            help=(
                f"MLP validation accuracy on a held-out {training.n_val}-sentence "
                f"split ({training.n_train} train / {training.n_val} val), predicting "
                "weak salience labels derived from cosine similarity to this "
                "document's own embedding centroid — a real, measured result "
                "from this training run, not a preset value."
            ),
        )
        st.caption(
            f"{training.n_train} train · {training.n_val} val sentences  \n"
            f"{training.epochs}-epoch AdamW (lr=0.005) / BCELoss  \n"
            "Trained fresh per document on its own SciBERT embeddings."
        )
    with col2:
        loss_df = pd.DataFrame(
            {
                "Training Loss":   training.train_losses,
                "Validation Loss": training.val_losses,
            },
            index=range(1, training.epochs + 1),
        )
        loss_df.index.name = "Epoch"
        st.line_chart(loss_df)

    st.caption(
        "⚠️ Training labels are weak/self-supervised (cosine similarity to the "
        "document's own embedding centroid, top vs. bottom tertile) — a proxy "
        f"signal, not human-annotated ground truth. {training.n_excluded_ambiguous} "
        f"of {training.n_total_sentences} sentences fell in the ambiguous middle "
        "tertile and were excluded from training/validation entirely, rather than "
        "forced into a class. Reported here for methodological transparency."
    )

    return training


def render_summary(
    result:        SummaryResult,
    filename_stem: str,
    prose:         str,
    polished:      bool,
):
    n  = result.n_total_sentences
    k  = result.k_used
    wc = result.word_count
    cr = result.compression_ratio
    mode_label  = _MODE_DISPLAY[result.mode]
    layer_label = "LLM-polished" if polished else "Extractive"

    st.markdown(
        f"<div class='rl-meta'>"
        f"<span><b>{k}</b> sentences selected</span>"
        f"<span><b>{n}</b> total in document</span>"
        f"<span><b>~{wc:,}</b> words</span>"
        f"<span><b>{cr:.0%}</b> extraction rate</span>"
        f"<span class='rl-mode-pill'>{layer_label}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    with st.chat_message("assistant"):
        st.markdown(prose)

    if polished:
        st.caption(
            "✨ Abstractive polish applied — factual content sourced "
            "exclusively from the extracted paper sentences."
        )
    else:
        st.caption(
            "📋 Extractive mode — sentences are verbatim from the paper, "
            "re-ordered chronologically for coherence."
        )

    download_txt = (
        f"ResearchLens — {layer_label} Summary\n"
        f"Mode   : {mode_label}\n"
        f"Sentences selected : {k} / {n}\n"
        f"Word count         : ~{wc:,}\n"
        f"Extraction rate    : {cr:.0%}\n"
        f"{'─' * 60}\n\n"
        f"{prose}"
    )
    st.download_button(
        label="⬇  Download summary (.txt)",
        data=download_txt,
        file_name=f"{filename_stem}_{result.mode}_{layer_label.lower()}_summary.txt",
        mime="text/plain",
    )


def render_rag_chat(doc_key: str, sentences: List[str], api_key: str):
    """
    "RAG: Chat with PDF" — genuine retrieval-augmented generation: the
    user's question is embedded, matched against every sentence in the
    current document via cosine_topk (pure NumPy, no vector database), and
    the top-3 matches are the ONLY context Gemini is allowed to answer
    from. Chat history lives in st.session_state, keyed by doc_key so
    switching to a different uploaded PDF starts a fresh conversation.

    Retrieval uses mean-pooled embeddings (run_rag_embeddings /
    embed_mean_pooled), NOT the [CLS]-pooled `data["embeddings"]` used for
    clustering — measured directly, raw [CLS] cosine similarity retrieved
    weakly-relevant sentences for short natural-language questions even on
    clean documents; mean-pooling noticeably improved match relevance with
    no new dependency.

    Uses a localized st.form (text_input + submit button) instead of
    st.chat_input deliberately: st.chat_input always docks to the bottom of
    the entire browser viewport regardless of which container it's called
    from, which would stretch across both dashboard columns and break the
    side-by-side layout. A form scoped to this column keeps the input
    exactly where the two-column design needs it.
    """
    st.markdown(
        "<div class='rl-section-label'>RAG: Chat with PDF</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Dense retrieval, pure NumPy — no vector database. Your question is "
        "embedded with the same SciBERT model used for the paper (mean-pooled "
        "for retrieval), matched by cosine similarity against every sentence "
        "in this document, and the top-3 matches are passed to Gemini as "
        "restricted context."
    )

    if not api_key:
        st.info(
            "Configure GEMINI_API_KEY in `.streamlit/secrets.toml` to enable chat "
            "answers — retrieval itself still works and will show matched sentences."
        )

    if st.session_state.get("rag_doc_key") != doc_key:
        st.session_state["rag_doc_key"] = doc_key
        st.session_state["rag_history"] = []
        st.session_state["rag_last_matches"] = []

    with st.container(height=420, border=True):
        if not st.session_state["rag_history"]:
            st.caption("No questions asked yet — try the box below.")
        for role, text in st.session_state["rag_history"]:
            with st.chat_message(role):
                st.markdown(text)

    with st.container():
        with st.form(key="rag_chat_form", clear_on_submit=True):
            user_query = st.text_input(
                "Ask a question about this paper…",
                placeholder="What methodology was used?",
                label_visibility="collapsed",
            )
            submit_button = st.form_submit_button(label="Send 🚀", use_container_width=True)

    if st.session_state["rag_last_matches"]:
        with st.expander("🔎 Retrieved Context (Top-3 Cosine Matches)"):
            for sentence, score in st.session_state["rag_last_matches"]:
                st.caption(f"`{score:.3f}`  {sentence}")

    if not (submit_button and user_query.strip()):
        return

    question = user_query.strip()
    st.session_state["rag_history"].append(("user", question))

    embedder      = load_embedder()
    rag_embeddings = run_rag_embeddings(tuple(sentences))
    query_vector  = embedder.embed_mean_pooled([question])[0]
    top_matches   = cosine_topk(query_vector, rag_embeddings, sentences, k=3)
    context_sentences = tuple(s for s, _ in top_matches)

    with st.spinner("Retrieving + generating…"):
        answer = _call_rag_chat(
            question=question,
            context_sentences=context_sentences,
            _api_key=api_key,
        )

    st.session_state["rag_history"].append(("assistant", answer))
    st.session_state["rag_last_matches"] = top_matches
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

def build_sidebar():
    with st.sidebar:
        st.markdown(
            "<h2 style='color:#e6f1ff;margin-bottom:0;font-size:1.3rem'>🔬 ResearchLens</h2>"
            "<p style='color:#6c7a9c;font-size:0.79rem;margin-top:0.2rem;margin-bottom:0'>"
            "Hybrid extractive-abstractive summarizer · self-supervised PyTorch "
            "classifier · Dense-RAG paper chat</p>",
            unsafe_allow_html=True,
        )
        st.divider()

        uploaded_file = st.file_uploader(
            "Upload Research Paper",
            type=["pdf"],
            help=(
                "PDF files only. Two-column academic layouts handled automatically. "
                "Scanned-image PDFs require an embedded text layer."
            ),
        )

        st.divider()
        st.markdown(
            "<div class='rl-section-label'>Target Summary Length</div>",
            unsafe_allow_html=True,
        )
        selected_label = st.radio(
            "Target Summary Length",
            list(_MODE_LABELS.keys()),
            label_visibility="collapsed",
        )
        selected_mode = _MODE_LABELS[selected_label]

        st.divider()
        st.markdown(
            "<div class='rl-section-label'>Pipeline Status</div>",
            unsafe_allow_html=True,
        )
        stage1_ph = st.empty()
        stage2_ph = st.empty()
        stage3_ph = st.empty()
        stage4_ph = st.empty()
        stage5_ph = st.empty()

        # ── Gemini API key — read here so the sidebar can show guidance ───────
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except (KeyError, FileNotFoundError):
            api_key = ""

        if not api_key:
            st.divider()
            st.markdown(
                "<div class='rl-section-label'>Gemini Setup</div>",
                unsafe_allow_html=True,
            )
            st.info(
                "**No API key detected** — LLM polish and RAG chat answers will be "
                "skipped and extractive/retrieved output will be shown instead.\n\n"
                "**To enable Gemini:**\n\n"
                "**macOS / Linux:**\n"
                "```bash\n"
                "mkdir -p .streamlit\n"
                "touch .streamlit/secrets.toml\n"
                "```\n"
                "Then open `.streamlit/secrets.toml` and add:\n"
                "```toml\n"
                'GEMINI_API_KEY = "AIza..."\n'
                "```\n\n"
                "**Streamlit Cloud:** paste the key in the app's "
                "_Settings → Secrets_ dashboard — no local file needed.\n\n"
                "Restart the app after saving the file."
            )

        st.divider()
        st.markdown(
            "<div class='rl-section-label'>Model</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size:0.77rem;color:#6c7a9c;line-height:1.75;margin:0'>"
            "allenai/<b style='color:#ccd6f6'>scibert_scivocab_uncased</b><br/>"
            "110M parameters · 768-dim CLS vectors<br/>"
            "Pre-trained on 1.14M scientific papers<br/>"
            "K-Means · cosine similarity on unit sphere<br/>"
            "Supervised head: <b style='color:#ccd6f6'>SalienceClassifierMLP</b> "
            "(768→128→32→1, Dropout)<br/>"
            "RAG retrieval: NumPy cosine similarity, top-3, no vector DB<br/>"
            "LLM: <b style='color:#ccd6f6'>gemini-2.5-flash</b> via direct REST</p>",
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown(
            "<p style='font-size:0.72rem;color:#3d4468;line-height:1.6'>"
            "ResearchLens · Update 8<br/>"
            "Self-Supervised PyTorch Classifier · Dense-RAG Chat · REST Gemini</p>",
            unsafe_allow_html=True,
        )

        return (
            uploaded_file, stage1_ph, stage2_ph, stage3_ph, stage4_ph, stage5_ph,
            selected_mode, api_key,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    (uploaded_file, stage1_ph, stage2_ph, stage3_ph, stage4_ph, stage5_ph,
     selected_mode, api_key) = build_sidebar()

    use_llm = bool(api_key)

    # ── Pending state ────────────────────────────────────────────────────────
    stage1_ph.markdown(_badge("Document Ingestion",  "pending"), unsafe_allow_html=True)
    stage2_ph.markdown(_badge("SciBERT Embedding",   "pending"), unsafe_allow_html=True)
    stage3_ph.markdown(_badge("Semantic Clustering", "pending"), unsafe_allow_html=True)
    stage4_ph.markdown(
        _badge(
            "LLM Polish — Skipped (Offline Mode)" if not use_llm else "LLM Polish",
            "offline" if not use_llm else "pending",
        ),
        unsafe_allow_html=True,
    )
    stage5_ph.markdown(_badge("Supervised Validation", "pending"), unsafe_allow_html=True)

    if uploaded_file is None:
        render_welcome()
        return

    pdf_stem   = os.path.splitext(uploaded_file.name)[0][:40] or "paper"
    file_bytes = uploaded_file.read()
    doc_key    = hashlib.md5(file_bytes).hexdigest()

    # ── Stages 1 + 2 ─────────────────────────────────────────────────────────
    stage1_ph.markdown(_badge("Document Ingestion",  "running"), unsafe_allow_html=True)
    stage2_ph.markdown(_badge("SciBERT Embedding",   "running"), unsafe_allow_html=True)

    try:
        with st.spinner(
            "Extracting text and computing SciBERT embeddings…  "
            "(first upload warms the model — subsequent uploads are instant)"
        ):
            data = run_pipeline(file_bytes)
    except ValueError as ve:
        stage1_ph.markdown(_badge("Document Ingestion", "error"), unsafe_allow_html=True)
        stage2_ph.markdown(_badge("SciBERT Embedding",  "error"), unsafe_allow_html=True)
        st.error(f"**Extraction failed:** {ve}")
        return
    except Exception as exc:
        stage1_ph.markdown(_badge("Document Ingestion", "error"), unsafe_allow_html=True)
        stage2_ph.markdown(_badge("SciBERT Embedding",  "error"), unsafe_allow_html=True)
        st.error(f"**Unexpected pipeline error:** {exc}")
        return

    stage1_ph.markdown(_badge("Document Ingestion",  "done"), unsafe_allow_html=True)
    stage2_ph.markdown(_badge("SciBERT Embedding",   "done"), unsafe_allow_html=True)

    # ── Stage 3: K-Means — all three modes pre-computed ──────────────────────
    stage3_ph.markdown(_badge("Semantic Clustering", "running"), unsafe_allow_html=True)

    try:
        sentences_tuple = tuple(data["sentences"])
        summaries: dict[str, SummaryResult] = {
            mode: run_clustering(
                sentences   = sentences_tuple,
                embeddings  = data["embeddings"],
                mode        = mode,
                model_name  = data["model"],
                device_used = data["device"],
            )
            for mode in ("short", "one_page", "full")
        }
    except Exception as exc:
        stage3_ph.markdown(_badge("Semantic Clustering", "error"), unsafe_allow_html=True)
        st.error(f"**Clustering failed:** {exc}")
        return

    stage3_ph.markdown(_badge("Semantic Clustering", "done"), unsafe_allow_html=True)

    # ── Selected mode comes from sidebar — main panel stays pristine ──────────
    result = summaries[selected_mode]

    # ── Quality metrics computed regardless of visibility — cheap, and the
    #    hidden diagnostics expander at the bottom of the page needs them ────
    rouge_scores = compute_rouge(
        summary_sentences = tuple(result.sentences),
        reference         = data["abstract"],
    )

    # ── Stage 5: self-supervised PyTorch classifier — full width, above the
    #    dual-column workspace, so the guide sees the trained model curves
    #    immediately without scrolling into either column ───────────────────
    stage5_ph.markdown(_badge("Supervised Validation", "running"), unsafe_allow_html=True)
    try:
        training = render_supervised_panel(data["embeddings"])
        stage5_ph.markdown(
            _badge(
                "Supervised Validation" if training is not None else "Supervised Validation — Skipped (Short Doc)",
                "done" if training is not None else "offline",
            ),
            unsafe_allow_html=True,
        )
    except Exception as exc:
        stage5_ph.markdown(_badge("Supervised Validation", "error"), unsafe_allow_html=True)
        st.warning(f"Supervised classifier training failed: {exc}")

    st.divider()

    # ── Dual-column workspace: Summary (left) | RAG Chat (right) — no tabs,
    #    no scrolling between the two; both are visible side by side ────────
    col_summary, col_chat = st.columns([1.1, 0.9])

    with col_summary:
        st.markdown(
            "<div class='rl-section-label'>Document Analytics</div>",
            unsafe_allow_html=True,
        )
        render_metrics(data)

        if use_llm:
            stage4_ph.markdown(_badge("LLM Polish", "running"), unsafe_allow_html=True)

        llm_error = None
        try:
            with st.spinner("Polishing with Gemini…" if use_llm else "Preparing summary…"):
                prose = _call_llm(
                    sentences = tuple(result.sentences),
                    _api_key  = api_key,
                )
            stage4_ph.markdown(
                _badge(
                    "LLM Polish" if use_llm else "LLM Polish — Skipped (Offline Mode)",
                    "done"       if use_llm else "offline",
                ),
                unsafe_allow_html=True,
            )
        except Exception as exc:
            prose     = " ".join(result.sentences)
            llm_error = str(exc)
            stage4_ph.markdown(_badge("LLM Polish", "error"), unsafe_allow_html=True)

        if llm_error:
            st.warning(
                f"Gemini polish failed ({llm_error}). "
                "Showing extractive output — verify GEMINI_API_KEY in .streamlit/secrets.toml."
            )

        st.markdown(
            "<div class='rl-section-label' style='margin-top:1.5rem'>Summary</div>",
            unsafe_allow_html=True,
        )
        render_summary(
            result        = result,
            filename_stem = pdf_stem,
            prose         = prose,
            polished      = use_llm and llm_error is None,
        )

    with col_chat:
        render_rag_chat(
            doc_key   = doc_key,
            sentences = data["sentences"],
            api_key   = api_key,
        )

    # ── Hidden diagnostics — silhouette + ROUGE are still fully computed and
    #    correct above, just not surfaced by default. Collapsed at the very
    #    bottom of the page for a second examiner who asks for them ─────────
    st.divider()
    with st.expander("🔬 Advanced Academic Diagnostics (Optional)", expanded=False):
        render_quality_metrics(result, rouge_scores)


if __name__ == "__main__":
    main()
