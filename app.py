"""
Student Performance Predictor
Main Streamlit Application Entry Point
"""

import streamlit as st
import pandas as pd
import os
import sys

# ── Page Config (MUST be first Streamlit call) ────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
  }

  /* Dark gradient background */
  .stApp {
    background: linear-gradient(135deg, #0d0d1a 0%, #111130 40%, #0a1628 100%) !important;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12122a 0%, #0d1b2e 100%) !important;
    border-right: 1px solid rgba(108,99,255,0.2) !important;
  }

  /* Cards & containers */
  div[data-testid="metric-container"] {
    background: rgba(108,99,255,0.1) !important;
    border: 1px solid rgba(108,99,255,0.25) !important;
    border-radius: 12px !important;
    padding: 12px !important;
  }

  /* Tabs */
  button[data-baseweb="tab"] {
    background: transparent !important;
    color: #aaa !important;
    border-radius: 8px 8px 0 0 !important;
    font-weight: 500 !important;
  }
  button[data-baseweb="tab"][aria-selected="true"] {
    background: rgba(108,99,255,0.2) !important;
    color: #b8b0ff !important;
    border-bottom: 2px solid #6C63FF !important;
  }

  /* Inputs */
  .stSelectbox > div > div, .stNumberInput > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
  }

  /* Slider */
  .stSlider > div[data-baseweb="slider"] {
    padding: 4px 0 !important;
  }

  /* Submit button */
  button[kind="primaryFormSubmit"], button[kind="primary"] {
    background: linear-gradient(135deg, #6C63FF, #00D4AA) !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    padding: 14px !important;
    transition: all 0.3s ease !important;
  }
  button[kind="primaryFormSubmit"]:hover, button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(108,99,255,0.4) !important;
  }

  /* Expander */
  details {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    padding: 4px 12px !important;
  }

  /* DataFrames */
  .stDataFrame {
    border: 1px solid rgba(108,99,255,0.2) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
  }

  /* Divider */
  hr { border-color: rgba(255,255,255,0.1) !important; }

  /* Remove Streamlit branding */
  #MainMenu, footer, header { visibility: hidden; }

  /* Plotly chart container */
  .js-plotly-plot { border-radius: 12px !important; }

  /* Info / warning / success boxes */
  .stAlert { border-radius: 10px !important; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
  ::-webkit-scrollbar-thumb { background: rgba(108,99,255,0.4); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# ── Helper: Load Data ──────────────────────────────────────────
@st.cache_data
def load_dataset():
    path = 'data/student_data.csv'
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# ── Helper: Load Predictor ─────────────────────────────────────
@st.cache_resource
def load_predictor():
    try:
        from models.predictor import StudentPredictor
        return StudentPredictor()
    except Exception as e:
        return None


# ── Setup Check ────────────────────────────────────────────────
def run_setup():
    """Auto-generate data and train models if not present."""
    import subprocess

    with st.spinner("⚙️ Generating dataset..."):
        subprocess.run([sys.executable, 'data/generate_dataset.py'], check=True)

    with st.spinner("🤖 Training ML models (this may take ~30 seconds)..."):
        subprocess.run([sys.executable, 'models/trainer.py'], check=True)

    st.success("✅ Setup complete! Reloading...")
    st.rerun()


# ── Sidebar ────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        # Logo / Brand
        st.markdown("""
        <div style="text-align:center;padding:20px 0 10px">
          <div style="font-size:42px">🎓</div>
          <h2 style="color:#b8b0ff;font-size:18px;font-weight:700;margin:6px 0 2px">
            Student Performance
          </h2>
          <p style="color:#6C63FF;font-size:13px;font-weight:600;margin:0;letter-spacing:1px">
            PREDICTOR
          </p>
        </div>
        <hr style="border-color:rgba(108,99,255,0.3);margin:10px 0">
        """, unsafe_allow_html=True)

        # Navigation
        st.markdown('<p style="color:#777;font-size:11px;font-weight:600;letter-spacing:1.5px;margin:8px 0 4px">NAVIGATION</p>', unsafe_allow_html=True)

        pages = {
            "📊 Dashboard":           "dashboard",
            "🔮 Predict Performance": "prediction",
            "📈 Analytics":           "analytics",
            "🎯 Feature Importance":  "feature_importance",
            "💡 Recommendations":     "recommendations",
        }

        if 'current_page' not in st.session_state:
            st.session_state['current_page'] = 'dashboard'

        for label, page_key in pages.items():
            is_active = st.session_state['current_page'] == page_key
            btn_style = "primary" if is_active else "secondary"
            if st.button(label, key=f"nav_{page_key}", use_container_width=True, type=btn_style):
                st.session_state['current_page'] = page_key
                st.rerun()

        # Dataset Info
        df = load_dataset()
        if df is not None:
            st.markdown('<hr style="border-color:rgba(108,99,255,0.2);margin:16px 0">', unsafe_allow_html=True)
            st.markdown('<p style="color:#777;font-size:11px;font-weight:600;letter-spacing:1.5px;margin:0 0 8px">DATASET INFO</p>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:12px">
              <p style="color:#aaa;font-size:12px;margin:2px 0">
                🎓 <strong style="color:#ddd">{len(df):,}</strong> Students
              </p>
              <p style="color:#aaa;font-size:12px;margin:2px 0">
                📋 <strong style="color:#ddd">{len(df.columns)}</strong> Features
              </p>
              <p style="color:#aaa;font-size:12px;margin:2px 0">
                📈 Avg CGPA: <strong style="color:#00D4AA">{df['final_cgpa'].mean():.2f}</strong>
              </p>
              <p style="color:#aaa;font-size:12px;margin:2px 0">
                ✅ Pass Rate: <strong style="color:#00C853">
                {(df['performance']=='Pass').sum()/len(df)*100:.1f}%</strong>
              </p>
            </div>
            """, unsafe_allow_html=True)

        # Prediction history badge
        if 'last_prediction' in st.session_state:
            res = st.session_state['last_prediction']['result']
            st.markdown(f"""
            <div style="background:rgba(108,99,255,0.15);border:1px solid rgba(108,99,255,0.3);
                        border-radius:10px;padding:10px;margin-top:12px;text-align:center">
              <p style="color:#777;font-size:10px;font-weight:600;letter-spacing:1px;margin:0 0 4px">LAST PREDICTION</p>
              <p style="color:#b8b0ff;font-size:18px;font-weight:700;margin:0">Grade {res['grade']}</p>
              <p style="color:#aaa;font-size:11px;margin:2px 0">CGPA ~{res['cgpa_estimate']:.1f}/10</p>
              <p style="color:#666;font-size:10px;margin:0">Confidence: {res['confidence']:.0f}%</p>
            </div>""", unsafe_allow_html=True)

        st.markdown('<hr style="border-color:rgba(108,99,255,0.2);margin:16px 0 8px">', unsafe_allow_html=True)
        st.markdown('<p style="color:#444;font-size:10px;text-align:center">Built with ❤️ using Python & Streamlit</p>', unsafe_allow_html=True)

    return st.session_state['current_page']


# ── Main ───────────────────────────────────────────────────────
def main():
    models_ready = os.path.exists('models/saved/metadata.json')
    data_ready   = os.path.exists('data/student_data.csv')

    # ── First-run setup ───────────────────────────────────────
    if not data_ready or not models_ready:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px">
          <div style="font-size:64px;margin-bottom:16px">🎓</div>
          <h1 style="color:#b8b0ff;font-size:32px;margin:0 0 8px">Student Performance Predictor</h1>
          <p style="color:#aaa;font-size:16px;margin:0 0 30px">
            First-time setup: generating dataset and training ML models.
          </p>
        </div>
        """, unsafe_allow_html=True)

        missing = []
        if not data_ready:   missing.append("📊 Student dataset")
        if not models_ready: missing.append("🤖 Trained ML models")

        for m in missing:
            st.warning(f"Missing: {m}")

        if st.button("🚀 Run Setup Now", type="primary", use_container_width=True):
            run_setup()
        return

    # ── Load resources ─────────────────────────────────────────
    df        = load_dataset()
    predictor = load_predictor()

    if df is None:
        st.error("❌ Could not load dataset. Please run setup again.")
        return
    if predictor is None:
        st.error("❌ Could not load models. Please run `python models/trainer.py`.")
        return

    # ── Render navigation & page ───────────────────────────────
    current_page = render_sidebar()

    if current_page == 'dashboard':
        from pages.dashboard import show
        show(df)

    elif current_page == 'prediction':
        from pages.prediction import show
        show(predictor, df)

    elif current_page == 'analytics':
        from pages.analytics import show
        show(df)

    elif current_page == 'feature_importance':
        from pages.feature_importance import show
        show(predictor, df)

    elif current_page == 'recommendations':
        from pages.recommendations import show
        show(predictor)


if __name__ == '__main__':
    main()
