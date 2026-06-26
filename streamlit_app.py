import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sentence_transformers import SentenceTransformer

# ============================================================
# BURNOUTSENSE — Streamlit Web Application
# Nur Fathihah Binti Salmizi | UiTM MSc Data Science
# Detection of Multilingual Work Burnout Expressions
# Using Sentence-BERT and Machine Learning (MBI Framework)
# ============================================================
# HOW TO RUN:
#   1. Make sure these files are in the same folder:
#      - app.py (this file)
#      - burnout_mlp_model.pkl
#      - burnout_label_encoder.pkl
#   2. Install: pip install streamlit sentence-transformers scikit-learn pandas matplotlib seaborn
#   3. Run: streamlit run app.py
# ============================================================

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="BurnoutSense",
    page_icon="🔵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Professional blue theme CSS ───────────────────────────────
st.markdown("""
<style>
  /* ── Google Fonts ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

  /* ── Global reset ── */
  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #F0F4FA;
  }

  /* ── Hide Streamlit default elements ── */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 2rem 3rem 3rem 3rem; max-width: 1200px; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A2463 0%, #1B4F9E 100%);
    border-right: none;
  }
  [data-testid="stSidebar"] * { color: #E8F0FE !important; }
  [data-testid="stSidebar"] .stRadio label { color: #E8F0FE !important; }
  [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15) !important; }

  /* ── Header banner ── */
  .burnout-header {
    background: linear-gradient(135deg, #0A2463 0%, #1B4F9E 60%, #2E7BC4 100%);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    color: white;
    position: relative;
    overflow: hidden;
  }
  .burnout-header::before {
    content: "";
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: rgba(255,255,255,0.05);
    border-radius: 50%;
  }
  .burnout-header::after {
    content: "";
    position: absolute;
    bottom: -60px; right: 80px;
    width: 300px; height: 300px;
    background: rgba(255,255,255,0.03);
    border-radius: 50%;
  }
  .header-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
  }
  .header-subtitle {
    font-size: 1rem;
    opacity: 0.85;
    margin: 0.4rem 0 0 0;
    font-weight: 400;
  }
  .header-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    margin-top: 1rem;
    font-weight: 500;
  }

  /* ── Metric cards ── */
  .metric-card {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    border-left: 4px solid #1B4F9E;
    box-shadow: 0 2px 12px rgba(10,36,99,0.08);
  }
  .metric-number {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #0A2463;
    margin: 0;
  }
  .metric-label {
    font-size: 0.8rem;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin: 0.25rem 0 0 0;
    font-weight: 500;
  }

  /* ── Result cards ── */
  .result-card {
    background: white;
    border-radius: 16px;
    padding: 2rem;
    box-shadow: 0 4px 20px rgba(10,36,99,0.10);
    border: 1px solid #E5EBF5;
  }
  .result-dimension {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #0A2463;
    margin: 0.5rem 0;
  }
  .result-confidence {
    font-size: 0.95rem;
    color: #6B7280;
  }

  /* ── Dimension badges ── */
  .badge-ee  { background:#FEE2E2; color:#991B1B; padding:0.3rem 0.8rem; border-radius:20px; font-size:0.8rem; font-weight:600; }
  .badge-dp  { background:#FEF3C7; color:#92400E; padding:0.3rem 0.8rem; border-radius:20px; font-size:0.8rem; font-weight:600; }
  .badge-ra  { background:#DBEAFE; color:#1E40AF; padding:0.3rem 0.8rem; border-radius:20px; font-size:0.8rem; font-weight:600; }

  /* ── Progress bars ── */
  .prob-bar-container { margin: 0.6rem 0; }
  .prob-label { font-size: 0.82rem; color: #374151; font-weight: 500; margin-bottom: 0.2rem; }
  .prob-bar-bg { background: #EEF2FF; border-radius: 6px; height: 10px; }
  .prob-bar-fill-ee { background: linear-gradient(90deg, #EF4444, #F87171); height: 10px; border-radius: 6px; }
  .prob-bar-fill-dp { background: linear-gradient(90deg, #F59E0B, #FCD34D); height: 10px; border-radius: 6px; }
  .prob-bar-fill-ra { background: linear-gradient(90deg, #3B82F6, #60A5FA); height: 10px; border-radius: 6px; }

  /* ── Section headers ── */
  .section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #0A2463;
    margin: 0 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #E5EBF5;
  }

  /* ── Input area ── */
  .stTextArea textarea {
    border: 2px solid #C7D7F0 !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s;
  }
  .stTextArea textarea:focus {
    border-color: #1B4F9E !important;
    box-shadow: 0 0 0 3px rgba(27,79,158,0.1) !important;
  }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, #0A2463, #1B4F9E) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 2rem !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 12px rgba(10,36,99,0.25) !important;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #0D2E7A, #2560C0) !important;
    box-shadow: 0 6px 18px rgba(10,36,99,0.35) !important;
    transform: translateY(-1px) !important;
  }

  /* ── Info box ── */
  .info-box {
    background: #EEF2FF;
    border-left: 4px solid #1B4F9E;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
    font-size: 0.9rem;
    color: #1E3A6E;
  }

  /* ── Footer ── */
  .app-footer {
    text-align: center;
    color: #9CA3AF;
    font-size: 0.78rem;
    padding: 2rem 0 1rem 0;
    border-top: 1px solid #E5EBF5;
    margin-top: 3rem;
  }

  /* ── Table styling ── */
  .dataframe { font-size: 0.85rem !important; }
  .stDataFrame { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ── MBI dimension config ──────────────────────────────────────
MBI_CONFIG = {
    "emotional_exhaustion": {
        "label":       "Emotional Exhaustion",
        "short":       "EE",
        "emoji":       "🔴",
        "badge_class": "badge-ee",
        "bar_class":   "prob-bar-fill-ee",
        "color":       "#EF4444",
        "description": (
            "Feelings of being emotionally overextended and depleted "
            "of emotional resources due to work demands."
        ),
        "examples": [
            "I am completely burned out from work",
            "Penat gila kerja tak larat dah",
            "So burnt out lah, work is draining me",
        ],
        "advice": (
            "Consider speaking to HR about workload management. "
            "Rest, boundaries, and support are important."
        ),
    },
    "depersonalization": {
        "label":       "Depersonalization / Cynicism",
        "short":       "DP",
        "emoji":       "🟡",
        "badge_class": "badge-dp",
        "bar_class":   "prob-bar-fill-dp",
        "color":       "#F59E0B",
        "description": (
            "Development of negative, detached, or cynical attitudes "
            "toward work, job responsibilities, or colleagues."
        ),
        "examples": [
            "I don't care about my job anymore",
            "Tak kisah dah pasal kerja, whatever lah",
            "Work is meaningless, just going through the motions",
        ],
        "advice": (
            "Reconnecting with purpose and meaningful tasks at work "
            "may help. Consider discussing job role with your manager."
        ),
    },
    "reduced_accomplishment": {
        "label":       "Reduced Personal Accomplishment",
        "short":       "RA",
        "emoji":       "🔵",
        "badge_class": "badge-ra",
        "bar_class":   "prob-bar-fill-ra",
        "color":       "#3B82F6",
        "description": (
            "Feelings of incompetence, reduced achievement, "
            "and declining sense of professional efficacy."
        ),
        "examples": [
            "I feel useless at work, nothing I do matters",
            "Rasa tak berguna sangat kat tempat kerja",
            "Imposter syndrome gila, everyone is better than me lah",
        ],
        "advice": (
            "Recognising small wins and seeking mentorship can help "
            "rebuild confidence. You are more capable than you feel."
        ),
    },
}

# ── Model results (UPDATE THESE with your real Colab results) ──
MODEL_RESULTS = {
    "TF-IDF + Logistic Regression": {
        "Accuracy":  0.9120,
        "Precision": 0.9276,
        "Recall":    0.9120,
        "F1-Score":  0.9139,
    },
    "TF-IDF + SVM": {
        "Accuracy":  0.8910,
        "Precision": 0.9119,
        "Recall":    0.8910,
        "F1-Score":  0.8934,
    },
    "TF-IDF + Random Forest": {
        "Accuracy":  0.9116,
        "Precision": 0.9161,
        "Recall":    0.9116,
        "F1-Score":  0.9127,
    },
    "SBERT + MLP (Proposed)": {
        "Accuracy":  0.9694,
        "Precision": 0.9696,
        "Recall":    0.9694,
        "F1-Score":  0.9694,
    },
}

DATASET_STATS = {
    "total":     13069,
    "english":   1535,
    "malay":     4424,
    "manglish":  7110,
    "ee_count":  8206,
    "dp_count":  3040,
    "ra_count":  1823,
}


# ── Helper functions ──────────────────────────────────────────

def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
        "\U00002500-\U00002BEF\U00002702-\U000027B0"
        "\U000024C2-\U0001F251\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff\u2640-\u2642"
        "\u2600-\u2B55\u200d\u23cf\u23e9\u231a\ufe0f\u3030"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)


def clean_text(text):
    text = str(text)
    text = remove_emojis(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


@st.cache_resource(show_spinner="Loading BurnoutSense AI model...")
def load_model():
    """Load SBERT + MLP model. Cached so it only loads once."""
    try:
        with open("burnout_mlp_model.pkl", "rb") as f:
            mlp = pickle.load(f)
        with open("burnout_label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
        sbert = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
        return mlp, le, sbert, True
    except FileNotFoundError:
        return None, None, None, False


def predict(text, mlp, le, sbert):
    """Predict MBI burnout dimension for input text."""
    cleaned   = clean_text(text)
    embedding = sbert.encode([cleaned], convert_to_numpy=True)
    pred_idx  = mlp.predict(embedding)[0]
    probas    = mlp.predict_proba(embedding)[0]
    label     = le.classes_[pred_idx]
    all_probs = {le.classes_[i]: float(p) for i, p in enumerate(probas)}
    return label, float(probas[pred_idx]), all_probs, cleaned


def render_probability_bars(all_probs):
    """Render custom HTML probability bars."""
    html = ""
    for dim, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
        cfg   = MBI_CONFIG[dim]
        width = int(prob * 100)
        html += f"""
        <div class="prob-bar-container">
          <div class="prob-label">
            {cfg['emoji']} {cfg['label']} &nbsp;
            <strong>{prob*100:.1f}%</strong>
          </div>
          <div class="prob-bar-bg">
            <div class="{cfg['bar_class']}" style="width:{width}%"></div>
          </div>
        </div>"""
    return html


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem 0;'>
      <div style='font-family:Space Grotesk; font-size:1.6rem;
                  font-weight:700; letter-spacing:-0.5px;'>
        🔵 BurnoutSense
      </div>
      <div style='font-size:0.75rem; opacity:0.7; margin-top:0.25rem;'>
        Multilingual Burnout Detection
      </div>
    </div>
    <hr/>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠 Home",
         "🔍 Predict Single Post",
         "📂 Batch Prediction",
         "📊 Model Performance",
         "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.75rem; opacity:0.65; line-height:1.6;'>
      <strong>MBI Framework</strong><br/>
      Maslach, Jackson & Leiter (1996)<br/><br/>
      <strong>Model</strong><br/>
      SBERT + MLP Classifier<br/><br/>
      <strong>Languages</strong><br/>
      English · Bahasa Malaysia · Manglish<br/><br/>
      <strong>⚠️ Disclaimer</strong><br/>
      Non-diagnostic tool for awareness only.
    </div>
    """, unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────
mlp, le, sbert, model_loaded = load_model()


# ══════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════
if page == "🏠 Home":

    # Header
    st.markdown("""
    <div class="burnout-header">
      <p class="header-title">BurnoutSense</p>
      <p class="header-subtitle">
        AI-powered multilingual work burnout detection using<br/>
        Sentence-BERT and MBI framework
      </p>
      <span class="header-badge">
        🎓 UiTM MSc Data Science Research · Nur Fathihah Binti Salmizi
      </span>
    </div>
    """, unsafe_allow_html=True)

    # Model status
    if model_loaded:
        st.success("✅ AI model loaded and ready.")
    else:
        st.warning(
            "⚠️ Model files not found. Please ensure "
            "`burnout_mlp_model.pkl` and `burnout_label_encoder.pkl` "
            "are in the same folder as this app."
        )

    # Dataset stats
    st.markdown('<p class="section-title">Dataset Overview</p>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
          <p class="metric-number">{DATASET_STATS['total']:,}</p>
          <p class="metric-label">Total Posts</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color:#9B59B6">
          <p class="metric-number" style="color:#6B21A8">{DATASET_STATS['manglish']:,}</p>
          <p class="metric-label">Manglish Posts</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color:#F59E0B">
          <p class="metric-number" style="color:#92400E">{DATASET_STATS['malay']:,}</p>
          <p class="metric-label">Malay Posts</p>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color:#10B981">
          <p class="metric-number" style="color:#065F46">{DATASET_STATS['english']:,}</p>
          <p class="metric-label">English Posts</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # MBI dimensions explained
    st.markdown('<p class="section-title">MBI Burnout Dimensions Detected</p>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    for col, (dim, cfg) in zip(
        [col1, col2, col3], MBI_CONFIG.items()
    ):
        with col:
            st.markdown(f"""
            <div class="result-card" style="border-top: 4px solid {cfg['color']};">
              <div style="font-size:2rem">{cfg['emoji']}</div>
              <div class="result-dimension" style="font-size:1.1rem; color:{cfg['color']}">
                {cfg['label']}
              </div>
              <p style="font-size:0.85rem; color:#6B7280; margin-top:0.5rem; line-height:1.5;">
                {cfg['description']}
              </p>
              <p style="font-size:0.78rem; color:#9CA3AF; margin-top:0.75rem;">
                <strong>Examples:</strong><br/>
                {"<br/>".join([f"• {e}" for e in cfg['examples']])}
              </p>
            </div>""", unsafe_allow_html=True)

    # How it works
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">How BurnoutSense Works</p>',
                unsafe_allow_html=True)

    steps = [
        ("📝", "Input Text", "Type or upload any post in English, Malay, or Manglish"),
        ("🧹", "Preprocessing", "Clean text — remove URLs, emojis, duplicates, detect language"),
        ("🧠", "SBERT Encoding", "paraphrase-multilingual-mpnet-base-v2 converts text to 768-dim semantic vectors"),
        ("🎯", "MLP Classification", "3-layer neural network classifies into MBI burnout dimensions"),
        ("📊", "Result", "Predicted dimension with confidence score and recommended action"),
    ]

    cols = st.columns(5)
    for col, (icon, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div style="text-align:center; padding:1rem; background:white;
                        border-radius:12px; box-shadow:0 2px 10px rgba(10,36,99,0.07);
                        height:180px;">
              <div style="font-size:1.8rem">{icon}</div>
              <div style="font-family:Space Grotesk; font-weight:700;
                          color:#0A2463; font-size:0.85rem; margin:0.4rem 0;">
                {title}
              </div>
              <div style="font-size:0.75rem; color:#6B7280; line-height:1.4;">
                {desc}
              </div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE: PREDICT SINGLE POST
# ══════════════════════════════════════════════════════════════
elif page == "🔍 Predict Single Post":

    st.markdown("""
    <div class="burnout-header">
      <p class="header-title">Predict Burnout Expression</p>
      <p class="header-subtitle">
        Enter any text in English, Bahasa Malaysia, or Manglish
      </p>
    </div>""", unsafe_allow_html=True)

    if not model_loaded:
        st.error("Model not loaded. Please check model files.")
        st.stop()

    # Example buttons
    st.markdown('<p class="section-title">Try an Example</p>',
                unsafe_allow_html=True)

    examples = {
        "🔴 Emotional Exhaustion (EN)":
            "I am completely burned out from work. Cannot even get out of bed anymore.",
        "🔴 Emotional Exhaustion (MS)":
            "Penat sangat kerja hari ni. Rasa nak berhenti je dah tak larat.",
        "🔴 Emotional Exhaustion (Manglish)":
            "So burnt out lah. Work is literally draining all my energy cannot already.",
        "🟡 Depersonalization (EN)":
            "I don't care about my job anymore. Just going through the motions every day.",
        "🟡 Depersonalization (MS)":
            "Tak kisah dah pasal kerja. Buat je lah apa-apa, dah hilang semangat.",
        "🟡 Depersonalization (Manglish)":
            "Work dah meaningless lah. Cannot care anymore, kerja whatever je.",
        "🔵 Reduced Accomplishment (EN)":
            "I feel completely useless at work. Everyone else is so much better than me.",
        "🔵 Reduced Accomplishment (MS)":
            "Rasa tak berguna sangat kat tempat kerja. Semua orang lagi pandai dari aku.",
        "🔵 Reduced Accomplishment (Manglish)":
            "Imposter syndrome gila. Work hard tapi nothing lah, rasa loser je kat office.",
    }

    cols = st.columns(3)
    selected_example = None
    example_items    = list(examples.items())

    for i, col in enumerate(cols):
        with col:
            for j in range(3):
                idx = i * 3 + j
                if idx < len(example_items):
                    label, text = example_items[idx]
                    if st.button(label, key=f"ex_{idx}",
                                 use_container_width=True):
                        selected_example = text

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">Enter Your Text</p>',
                unsafe_allow_html=True)

    default_text = selected_example if selected_example else ""
    user_input   = st.text_area(
        "Post text",
        value=default_text,
        placeholder="Type any social media post here...\n\nExample: Penat gila kerja tak larat dah, so burnt out lah",
        height=130,
        label_visibility="collapsed",
    )

    col_btn, col_clear = st.columns([1, 5])
    with col_btn:
        predict_btn = st.button("🔍 Analyse Post", use_container_width=True)

    if predict_btn:
        if not user_input.strip():
            st.warning("Please enter some text first.")
        elif len(user_input.strip()) < 5:
            st.warning("Text is too short. Please enter at least one sentence.")
        else:
            with st.spinner("Analysing..."):
                label, confidence, all_probs, cleaned = predict(
                    user_input, mlp, le, sbert)

            cfg = MBI_CONFIG[label]

            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown('<p class="section-title">Analysis Result</p>',
                        unsafe_allow_html=True)

            col_result, col_detail = st.columns([1, 1])

            with col_result:
                st.markdown(f"""
                <div class="result-card" style="border-top: 5px solid {cfg['color']};">
                  <div style="font-size:3rem; margin-bottom:0.5rem">{cfg['emoji']}</div>
                  <span class="{cfg['badge_class']}">MBI DIMENSION DETECTED</span>
                  <div class="result-dimension">{cfg['label']}</div>
                  <div class="result-confidence">
                    Confidence: <strong style="color:{cfg['color']}">{confidence*100:.1f}%</strong>
                  </div>
                  <hr style="border:none; border-top:1px solid #F0F4FA; margin:1rem 0"/>
                  <p style="font-size:0.85rem; color:#4B5563; line-height:1.6;">
                    {cfg['description']}
                  </p>
                  <div class="info-box">
                    💡 <strong>Recommendation:</strong> {cfg['advice']}
                  </div>
                  <p style="font-size:0.78rem; color:#9CA3AF; margin-top:1rem;">
                    ⚠️ This is a non-diagnostic tool for awareness only.
                    Not a clinical assessment.
                  </p>
                </div>""", unsafe_allow_html=True)

            with col_detail:
                st.markdown(f"""
                <div class="result-card">
                  <div class="section-title" style="margin-top:0">
                    All Dimension Scores
                  </div>
                  {render_probability_bars(all_probs)}
                  <hr style="border:none; border-top:1px solid #F0F4FA; margin:1.5rem 0"/>
                  <div style="font-size:0.82rem; color:#6B7280;">
                    <strong>Input text</strong><br/>
                    <span style="color:#374151;">{user_input[:120]}
                    {"..." if len(user_input) > 120 else ""}</span>
                    <br/><br/>
                    <strong>After preprocessing</strong><br/>
                    <span style="color:#374151; font-family:monospace;">
                    {cleaned[:120]}{"..." if len(cleaned) > 120 else ""}</span>
                  </div>
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE: BATCH PREDICTION
# ══════════════════════════════════════════════════════════════
elif page == "📂 Batch Prediction":

    st.markdown("""
    <div class="burnout-header">
      <p class="header-title">Batch Prediction</p>
      <p class="header-subtitle">
        Upload a CSV file to classify multiple posts at once
      </p>
    </div>""", unsafe_allow_html=True)

    if not model_loaded:
        st.error("Model not loaded. Please check model files.")
        st.stop()

    st.markdown('<p class="section-title">Upload CSV File</p>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
      📋 <strong>CSV Requirements:</strong>
      Your file must have a column containing the post text.
      Common column names: <code>tweet</code>, <code>text</code>,
      <code>post</code>, <code>content</code>
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload CSV", type=["csv"],
        label_visibility="collapsed")

    if uploaded:
        df_up = pd.read_csv(uploaded)
        st.success(f"File uploaded: {len(df_up):,} rows, {len(df_up.columns)} columns")
        st.dataframe(df_up.head(3), use_container_width=True)

        text_col = st.selectbox(
            "Select the column containing post text:",
            options=list(df_up.columns),
        )

        if st.button("🚀 Run Batch Prediction", use_container_width=False):
            progress = st.progress(0, text="Preprocessing...")
            df_up["text_clean"] = df_up[text_col].apply(clean_text)
            progress.progress(25, text="Generating SBERT embeddings...")

            embeddings = sbert.encode(
                df_up["text_clean"].tolist(),
                batch_size=64, show_progress_bar=False,
                convert_to_numpy=True,
            )
            progress.progress(70, text="Classifying...")

            pred_idx = mlp.predict(embeddings)
            probas   = mlp.predict_proba(embeddings)

            df_up["predicted_dimension"] = le.inverse_transform(pred_idx)
            df_up["confidence"]          = probas.max(axis=1).round(4)
            for i, cls in enumerate(le.classes_):
                df_up[f"prob_{cls}"] = probas[:, i].round(4)

            progress.progress(100, text="Done!")

            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown('<p class="section-title">Prediction Results</p>',
                        unsafe_allow_html=True)

            r1, r2, r3 = st.columns(3)
            vc = df_up["predicted_dimension"].value_counts()
            for col, (dim, cfg) in zip([r1,r2,r3], MBI_CONFIG.items()):
                with col:
                    count = vc.get(dim, 0)
                    pct   = count / len(df_up) * 100
                    st.markdown(f"""
                    <div class="metric-card"
                         style="border-left-color:{cfg['color']}; text-align:center">
                      <div style="font-size:1.5rem">{cfg['emoji']}</div>
                      <p class="metric-number" style="color:{cfg['color']}">{count:,}</p>
                      <p class="metric-label">{cfg['short']} — {pct:.1f}%</p>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br/>", unsafe_allow_html=True)
            display_cols = [text_col, "predicted_dimension", "confidence"]
            st.dataframe(
                df_up[display_cols].head(20),
                use_container_width=True,
            )

            csv_out = df_up.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ Download Full Results CSV",
                data=csv_out,
                file_name="burnoutsense_predictions.csv",
                mime="text/csv",
            )


# ══════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":

    st.markdown("""
    <div class="burnout-header">
      <p class="header-title">Model Performance</p>
      <p class="header-subtitle">
        Comparison of baseline and proposed models
      </p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
      ℹ️ Update the <code>MODEL_RESULTS</code> dictionary in
      <code>app.py</code> with your real Colab results
      once training is complete.
    </div>""", unsafe_allow_html=True)

    # Results table
    st.markdown('<p class="section-title">Model Comparison Table</p>',
                unsafe_allow_html=True)

    df_results = pd.DataFrame(MODEL_RESULTS).T.reset_index()
    df_results.columns = ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]

    # Highlight best model
    best_idx = df_results["F1-Score"].idxmax()

    st.dataframe(
        df_results.style.highlight_max(
            subset=["Accuracy","Precision","Recall","F1-Score"],
            color="#DBEAFE",
        ).format({
            "Accuracy":  "{:.4f}",
            "Precision": "{:.4f}",
            "Recall":    "{:.4f}",
            "F1-Score":  "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Bar chart
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">Performance Visualisation</p>',
                unsafe_allow_html=True)

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    x       = np.arange(len(df_results))
    width   = 0.18
    colors  = ["#3B82F6","#EF4444","#10B981","#F59E0B"]

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("#F0F4FA")
    ax.set_facecolor("#F0F4FA")

    for i, (m, c) in enumerate(zip(metrics, colors)):
        bars = ax.bar(x + i*width, df_results[m], width,
                      label=m, color=c, edgecolor="white",
                      alpha=0.9, linewidth=0.8)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2,
                        h + 0.005, f"{h:.3f}",
                        ha="center", va="bottom", fontsize=7.5,
                        color="#374151")

    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(df_results["Model"], rotation=15,
                       ha="right", fontsize=9, color="#374151")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", color="#374151")
    ax.set_title("Model Comparison — Baseline vs Proposed (SBERT + MLP)",
                 fontsize=12, fontweight="bold", color="#0A2463", pad=15)
    ax.legend(loc="upper right", fontsize=9)
    ax.axhline(y=0.8, color="#94A3B8", linestyle="--",
               alpha=0.6, linewidth=1)
    ax.text(len(x)-0.3, 0.81, "80% threshold",
            fontsize=8, color="#94A3B8")
    ax.spines[["top","right","left","bottom"]].set_visible(False)
    ax.yaxis.set_tick_params(labelcolor="#374151")
    ax.grid(axis="y", alpha=0.3, color="#CBD5E1")
    plt.tight_layout()
    st.pyplot(fig)

    # Research info
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">Research Information</p>',
                unsafe_allow_html=True)

    info_cols = st.columns(3)
    infos = [
        ("Framework", "Maslach Burnout Inventory (MBI)\nMaslach et al., 1996"),
        ("Proposed Model", "SBERT + MLP\nparaphrase-multilingual-mpnet-base-v2"),
        ("Dataset", f"{DATASET_STATS['total']:,} multilingual posts\nEN · MS · Manglish"),
    ]
    for col, (title, content) in zip(info_cols, infos):
        with col:
            st.markdown(f"""
            <div class="result-card">
              <div style="font-weight:700; color:#0A2463;
                          font-size:0.9rem; margin-bottom:0.5rem">
                {title}
              </div>
              <div style="font-size:0.85rem; color:#4B5563;
                          white-space:pre-line; line-height:1.6">
                {content}
              </div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════
elif page == "ℹ️ About":

    st.markdown("""
    <div class="burnout-header">
      <p class="header-title">About BurnoutSense</p>
      <p class="header-subtitle">
        Research background and technical details
      </p>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown('<p class="section-title">Research Overview</p>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="result-card">
          <p style="font-size:0.92rem; color:#374151; line-height:1.8;">
            <strong>BurnoutSense</strong> is developed as part of an MSc Data Science
            thesis at Universiti Teknologi MARA (UiTM), Malaysia.
            <br/><br/>
            The study addresses a critical gap in occupational health research:
            the lack of multilingual, theory-driven tools for early detection of
            work burnout expressions in Malaysian social media, where users
            frequently code-switch between English, Bahasa Malaysia, and Manglish.
            <br/><br/>
            The system maps social media posts to three dimensions of the
            Maslach Burnout Inventory (MBI): Emotional Exhaustion,
            Depersonalization, and Reduced Personal Accomplishment.
            <br/><br/>
            Sentence-BERT (SBERT) is used to generate multilingual sentence
            embeddings, which are then classified by a Multi-Layer Perceptron
            (MLP). This proposed architecture is compared against TF-IDF
            baseline models to demonstrate the superiority of semantic
            representations for multilingual burnout detection.
          </p>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown('<p class="section-title">Research Details</p>',
                    unsafe_allow_html=True)
        details = [
            ("Researcher",    "Nur Fathihah Binti Salmizi"),
            ("Student ID",    "2024875526"),
            ("Programme",     "Master of Data Science (CDCS779)"),
            ("Faculty",       "Computer & Mathematical Sciences"),
            ("University",    "Universiti Teknologi MARA (UiTM)"),
            ("Supervisor",    "Dr. Ruhaila Maskat"),
            ("Year",          "2026"),
            ("Platform",      "Threads (Meta)"),
            ("Languages",     "English · Malay · Manglish"),
        ]
        for key, val in details:
            st.markdown(f"""
            <div style="display:flex; padding:0.5rem 0;
                        border-bottom:1px solid #F0F4FA;">
              <div style="font-size:0.78rem; color:#9CA3AF;
                          width:110px; flex-shrink:0; font-weight:500;">
                {key}
              </div>
              <div style="font-size:0.85rem; color:#374151; font-weight:500;">
                {val}
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
          ⚠️ <strong>Disclaimer:</strong> BurnoutSense is a
          non-diagnostic research prototype. It is intended for
          awareness and academic purposes only, not clinical use.
          Results should not replace professional mental health
          assessment.
        </div>""", unsafe_allow_html=True)

    # References
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">Key References</p>',
                unsafe_allow_html=True)
    refs = [
        "Maslach, C., Jackson, S. E., & Leiter, M. P. (1996). Maslach Burnout Inventory Manual (3rd ed.).",
        "Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP.",
        "World Health Organization. (2019). Burn-out an occupational phenomenon. ICD-11.",
        "Shamsuddin, A. M., et al. (2024). Semi-automatic sentiment identification for Malay-English code-switched data.",
        "Wei, J., & Zou, K. (2019). EDA: Easy Data Augmentation Techniques for Text Classification. EMNLP.",
    ]
    for ref in refs:
        st.markdown(f"""
        <div style="font-size:0.82rem; color:#4B5563; padding:0.4rem 0
                    0.4rem 1rem; border-left:3px solid #C7D7F0;
                    margin-bottom:0.5rem; line-height:1.5;">
          {ref}
        </div>""", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
  BurnoutSense · UiTM MSc Data Science Research 2026 ·
  Nur Fathihah Binti Salmizi ·
  Non-diagnostic research prototype only
</div>""", unsafe_allow_html=True)
