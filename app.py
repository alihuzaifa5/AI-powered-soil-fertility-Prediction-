import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import os
from datetime import datetime

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Soil Fertility Analyzer",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  FIELD LABORATORY STYLE & THEME
# ─────────────────────────────────────────────
st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;0,6..72,700;1,6..72,400;1,6..72,600&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

.stApp {
    background-color: #F6F4ED;
    color: #1E261D;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

h1, h2, h3, .serif-font {
    font-family: 'Newsreader', Georgia, serif !important;
    color: #1A3B1A;
    font-weight: 600;
}

.eyebrow {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #73786B;
    margin-bottom: 0.25rem;
}

.page-title {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 2.6rem;
    line-height: 1.15;
    color: #1A3B1A;
    margin: 0 0 0.35rem 0;
    font-weight: 700;
}

.page-subtitle {
    font-size: 0.96rem;
    color: #4F574A;
    line-height: 1.55;
    margin-bottom: 1.4rem;
}

.top-bar {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    margin-bottom: 0.3rem;
}

.top-bar-icon {
    background: #1A3B1A;
    color: #FFFFFF;
    border-radius: 8px;
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.15rem;
    box-shadow: 0 2px 6px rgba(26, 59, 26, 0.2);
}

.top-bar-lab {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #73786B;
}

.top-bar-title {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 1.45rem;
    font-weight: 700;
    color: #1A3B1A;
}

.lab-card-header {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: #1A3B1A;
    border-bottom: 1px solid #EAE3D8;
    padding-bottom: 0.65rem;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.param-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    margin-bottom: 0.15rem;
}

.param-name {
    font-weight: 600;
    color: #1F2B1D;
    font-size: 0.92rem;
}

.param-badge {
    background: #F0ECE2;
    border-bottom: 2px solid #8FA689;
    padding: 0.15rem 0.6rem;
    border-radius: 4px;
    font-weight: 700;
    font-size: 0.92rem;
    color: #1A3B1A;
    text-align: right;
    min-width: 70px;
}

/* Preset Buttons */
div.st-key-btn_low_fert button {
    background: #FDF2F2 !important;
    border: 1.5px solid #FCA5A5 !important;
    border-radius: 20px !important;
    box-shadow: none !important;
    transition: all 0.15s ease !important;
}
div.st-key-btn_low_fert button p, div.st-key-btn_low_fert button span {
    color: #DC2626 !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
}
div.st-key-btn_low_fert button:hover {
    background: #FEE2E2 !important;
    border-color: #EF4444 !important;
    transform: translateY(-1px);
}

div.st-key-btn_med_fert button {
    background: #FFFBEB !important;
    border: 1.5px solid #FCD34D !important;
    border-radius: 20px !important;
    box-shadow: none !important;
    transition: all 0.15s ease !important;
}
div.st-key-btn_med_fert button p, div.st-key-btn_med_fert button span {
    color: #D97706 !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
}
div.st-key-btn_med_fert button:hover {
    background: #FEF3C7 !important;
    border-color: #F59E0B !important;
    transform: translateY(-1px);
}

div.st-key-btn_very_fert button {
    background: #F0FDF4 !important;
    border: 1.5px solid #86EFAC !important;
    border-radius: 20px !important;
    box-shadow: none !important;
    transition: all 0.15s ease !important;
}
div.st-key-btn_very_fert button p, div.st-key-btn_very_fert button span {
    color: #16A34A !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
}
div.st-key-btn_very_fert button:hover {
    background: #DCFCE7 !important;
    border-color: #22C55E !important;
    transform: translateY(-1px);
}

div.st-key-btn_reset_all button {
    background: #F4F1EA !important;
    border: 1.5px solid #D6CEBE !important;
    border-radius: 20px !important;
    box-shadow: none !important;
    transition: all 0.15s ease !important;
}
div.st-key-btn_reset_all button p, div.st-key-btn_reset_all button span {
    color: #555E50 !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
}
div.st-key-btn_reset_all button:hover {
    background: #ECE7DD !important;
    border-color: #B5AB99 !important;
    transform: translateY(-1px);
}

/* Primary Action Button */
div.stButton > button[kind="primary"] {
    background: #1A3B1A !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
    padding: 0.65rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1.02rem !important;
    border: none !important;
    box-shadow: 0 3px 10px rgba(26, 59, 26, 0.25) !important;
    transition: all 0.18s ease !important;
}
div.stButton > button[kind="primary"]:hover {
    background: #122B12 !important;
    box-shadow: 0 5px 16px rgba(26, 59, 26, 0.35) !important;
    transform: translateY(-1px);
}

/* Prediction Results Panel */
.pred-panel {
    background: #1A3B1A;
    border-radius: 14px;
    padding: 1.4rem 1.5rem 1.3rem;
    color: #FFFFFF;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 16px rgba(26, 59, 26, 0.18);
}
.pred-panel-eyebrow {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #8FB88A;
    margin-bottom: 0.35rem;
}
.pred-panel-title {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 1.85rem;
    font-weight: 700;
    color: #FFFFFF;
    line-height: 1.2;
    margin-bottom: 1.1rem;
}
.pred-confidence-ring {
    display: flex;
    align-items: center;
    gap: 1rem;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.85rem;
}
.pred-ring-num {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: #FFFFFF;
    line-height: 1;
}
.pred-ring-label {
    font-size: 0.72rem;
    color: #AEBDAE;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
}
.pred-ring-sub {
    font-size: 0.84rem;
    color: #CDDACD;
    margin-top: 0.2rem;
}
.pred-class-chip {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 30px;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin-bottom: 0.6rem;
}
.chip-low {
    background: rgba(220, 38, 38, 0.22);
    color: #FCA5A5;
    border: 1px solid rgba(220, 38, 38, 0.4);
}
.chip-medium {
    background: rgba(217, 119, 6, 0.22);
    color: #FCD34D;
    border: 1px solid rgba(217, 119, 6, 0.4);
}
.chip-high {
    background: rgba(22, 163, 74, 0.25);
    color: #86EFAC;
    border: 1px solid rgba(22, 163, 74, 0.45);
}

/* Card Container for Charts */
.chart-card {
    background: #FFFFFF;
    border: 1px solid #E3DCCE;
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
}

/* Scorecard Grid */
.scorecard-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin-bottom: 1.4rem;
}
.scorecard-tile {
    background: #FFFFFF;
    border: 1px solid #E0D9CE;
    border-radius: 10px;
    padding: 0.8rem 0.95rem;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.scorecard-tile:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
}
.scorecard-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.35rem;
}
.scorecard-name {
    font-weight: 700;
    font-size: 0.83rem;
    color: #1E261D;
}
.scorecard-val {
    font-size: 1.2rem;
    font-weight: 800;
    color: #1A3B1A;
    font-family: 'Newsreader', Georgia, serif;
}

/* KPI Cards */
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E0D9CE;
    border-radius: 12px;
    padding: 1rem 1.3rem;
    text-align: center;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
}
.kpi-num {
    font-size: 1.95rem;
    font-weight: 800;
    font-family: 'Newsreader', Georgia, serif;
    color: #1A3B1A;
    margin: 0.1rem 0;
}
.kpi-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #707767;
    font-weight: 600;
}

.worksheet-card-title {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 1.15rem;
    color: #1A3B1A;
    font-weight: 600;
    border-bottom: 1px dashed #D6CCBD;
    padding-bottom: 0.65rem;
    margin-bottom: 1.1rem;
}

div[data-testid="stTextInput"] input {
    background: #FFFFFF !important;
    border: 1px solid #D0C9BD !important;
    border-radius: 8px !important;
    color: #1E261D !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
    border-bottom: 2px solid #DDD6CA;
    padding-bottom: 4px;
    margin-bottom: 1.4rem;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #555C4F;
    padding: 0.45rem 1.3rem;
    border-radius: 8px 8px 0 0;
}
.stTabs [aria-selected="true"] {
    color: #1A3B1A !important;
    border-bottom: 3px solid #1A3B1A !important;
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""")

# ─────────────────────────────────────────────
#  PATHS & MODEL LOADING
# ─────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "best_xgboost_model.joblib")
HISTORY_PATH = os.path.join(BASE_DIR, "prediction_history.csv")

@st.cache_resource
def load_model(path: str):
    try:
        return joblib.load(path)
    except Exception:
        try:
            import xgboost as xgb
            m = xgb.XGBClassifier()
            m.load_model(path)
            return m
        except Exception as e:
            raise ValueError(f"Failed to load model: {e}")

try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    st.error("Model not found. Make sure best_xgboost_model.joblib is in the workspace.")
    st.stop()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ─────────────────────────────────────────────
#  FEATURE DEFINITIONS & PRESETS
# ─────────────────────────────────────────────
FEATURES = {
    "N":  {"label":"Nitrogen (N)","unit":"kg/ha","target_min":120.0,"target_max":280.0,"target":"TARGET 120–280 kg/ha","min":0.0,"max":450.0,"default":250.0,"step":1.0,"priority":"High","remedy_low":"Apply urea (46% N) or ammonium sulphate in 2–3 split applications.","remedy_high":"Suspend nitrogen application to prevent vegetative lodging."},
    "P":  {"label":"Phosphorus (P)","unit":"kg/ha","target_min":15.0,"target_max":50.0,"target":"TARGET 15–50 kg/ha","min":0.0,"max":150.0,"default":15.0,"step":0.1,"priority":"High","remedy_low":"Band apply SSP or DAP at root depth before sowing.","remedy_high":"Omit phosphate fertilizers for the current crop cycle."},
    "K":  {"label":"Potassium (K)","unit":"kg/ha","target_min":180.0,"target_max":450.0,"target":"TARGET 180–450 kg/ha","min":0.0,"max":900.0,"default":500.0,"step":1.0,"priority":"High","remedy_low":"Apply Muriate of Potash (MOP/KCl) or Potassium Sulphate.","remedy_high":"Potassium is abundant. No supplementation required."},
    "pH": {"label":"Soil pH","unit":"pH","target_min":6.0,"target_max":7.5,"target":"TARGET 6.0–7.5 pH","min":1.0,"max":14.0,"default":7.0,"step":0.01,"priority":"High","remedy_low":"Broadcast agricultural lime (CaCO3) or dolomite.","remedy_high":"Broadcast gypsum (CaSO4) or elemental sulphur."},
    "EC": {"label":"Electrical Conductivity (EC)","unit":"dS/m","target_min":0.2,"target_max":1.2,"target":"TARGET 0.2–1.2 dS/m","min":0.0,"max":2.5,"default":0.50,"step":0.01,"priority":"Medium","remedy_low":"Low EC — safe, non-saline conditions.","remedy_high":"Elevated salinity. Improve drainage and deep furrow irrigation."},
    "OC": {"label":"Organic Carbon (OC)","unit":"%","target_min":0.60,"target_max":1.50,"target":"TARGET 0.6–1.5%","min":0.0,"max":25.0,"default":0.60,"step":0.01,"priority":"High","remedy_low":"Incorporate 2–3 tons/ha FYM or compost.","remedy_high":"Abundant organic carbon. Maintain through minimal tillage."},
    "S":  {"label":"Sulphur (S)","unit":"ppm","target_min":8.0,"target_max":25.0,"target":"TARGET 8.0–25.0 ppm","min":0.0,"max":50.0,"default":7.5,"step":0.1,"priority":"Medium","remedy_low":"Apply gypsum or ammonium sulphate @ 20–25 kg/ha.","remedy_high":"Adequate sulphur levels."},
    "Zn": {"label":"Zinc (Zn)","unit":"ppm","target_min":0.6,"target_max":2.0,"target":"TARGET 0.6–2.0 ppm","min":0.0,"max":45.0,"default":0.50,"step":0.01,"priority":"Medium","remedy_low":"Apply 15–20 kg/ha Zinc Sulphate (ZnSO4).","remedy_high":"Zinc sufficient; avoid over-application."},
    "Fe": {"label":"Iron (Fe)","unit":"ppm","target_min":4.5,"target_max":15.0,"target":"TARGET 4.5–15.0 ppm","min":0.0,"max":45.0,"default":4.00,"step":0.01,"priority":"Medium","remedy_low":"Foliar spray with 1% Ferrous Sulphate (FeSO4).","remedy_high":"Iron sufficient for chlorophyll synthesis."},
    "Cu": {"label":"Copper (Cu)","unit":"ppm","target_min":0.4,"target_max":2.0,"target":"TARGET 0.4–2.0 ppm","min":0.0,"max":4.0,"default":1.00,"step":0.01,"priority":"Medium","remedy_low":"Apply Copper Sulphate @ 5 kg/ha.","remedy_high":"Adequate copper reserve."},
    "Mn": {"label":"Manganese (Mn)","unit":"ppm","target_min":4.0,"target_max":15.0,"target":"TARGET 4.0–15.0 ppm","min":0.0,"max":35.0,"default":8.50,"step":0.01,"priority":"Medium","remedy_low":"Foliar spray 0.5% Manganese Sulphate (MnSO4).","remedy_high":"Manganese well-balanced."},
    "B":  {"label":"Boron (B)","unit":"ppm","target_min":0.5,"target_max":2.0,"target":"TARGET 0.5–2.0 ppm","min":0.0,"max":4.0,"default":0.60,"step":0.01,"priority":"Medium","remedy_low":"Apply Borax @ 5–10 kg/ha.","remedy_high":"Boron sufficient."},
}

PRESETS = {
    "Low Fertility":   {"N":138.0,"P":8.6,"K":350.0,"pH":7.46,"EC":0.62,"OC":0.50,"S":5.9,"Zn":0.24,"Fe":0.31,"Cu":0.77,"Mn":8.71,"B":0.11},
    "Medium Fertility":{"N":250.0,"P":15.0,"K":500.0,"pH":7.00,"EC":0.50,"OC":0.60,"S":7.5,"Zn":0.50,"Fe":4.00,"Cu":1.00,"Mn":8.50,"B":0.60},
    "Very Fertile":    {"N":360.0,"P":30.0,"K":700.0,"pH":7.40,"EC":0.60,"OC":0.80,"S":15.0,"Zn":0.80,"Fe":5.00,"Cu":1.20,"Mn":10.00,"B":0.80},
}

FERTILITY_MAP = {
    0: {"label":"Low Fertility",   "chip":"chip-low",   "color":"#DC2626","health_rating":"Sub-Optimal",  "advice":"Soil requires comprehensive nutrient amendment. Apply NPK fertilizers, organic compost, and trace micro-minerals."},
    1: {"label":"Moderate Fertility","chip":"chip-medium","color":"#D97706","health_rating":"Moderate / Fair","advice":"Moderate fertility. Supplement deficient nutrients and maintain balanced crop management."},
    2: {"label":"Very Fertile",    "chip":"chip-high",  "color":"#16A34A","health_rating":"Excellent",    "advice":"Excellent condition! Retain soil health through organic mulch, minimal tillage, and periodic monitoring."},
}

def load_history() -> pd.DataFrame:
    if os.path.exists(HISTORY_PATH):
        try:
            return pd.read_csv(HISTORY_PATH)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def save_history_record(record: dict):
    df_new = pd.DataFrame([record])
    if os.path.exists(HISTORY_PATH):
        try:
            df_existing = pd.read_csv(HISTORY_PATH)
            if "Crop" in df_existing.columns:
                df_existing = df_existing.drop(columns=["Crop"])
            df_updated = pd.concat([df_new, df_existing], ignore_index=True)
        except Exception:
            df_updated = df_new
    else:
        df_updated = df_new
    df_updated.to_csv(HISTORY_PATH, index=False)

# Initialize Session State
for feat, meta in FEATURES.items():
    if f"slider_{feat}" not in st.session_state:
        st.session_state[f"slider_{feat}"] = float(meta["default"])
if "sample_id" not in st.session_state:
    st.session_state["sample_id"] = "Sample #01"
if "prediction_result" not in st.session_state:
    st.session_state["prediction_result"] = None

def apply_preset(preset_name: str):
    if preset_name in PRESETS:
        for k, v in PRESETS[preset_name].items():
            st.session_state[f"slider_{k}"] = float(v)

def reset_to_defaults():
    for k, meta in FEATURES.items():
        st.session_state[f"slider_{k}"] = float(meta["default"])

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab_analyzer, tab_history = st.tabs(["Field Report & Analyzer", "Prediction History Dashboard"])

# ═════════════════════════════════════════════
#  TAB 1: ANALYZER
# ═════════════════════════════════════════════
with tab_analyzer:

    st.html("""
    <div class="top-bar">
        <div class="top-bar-icon">🌱</div>
        <div>
            <div class="top-bar-lab">Field Laboratory</div>
            <div class="top-bar-title">Soil Fertility Assessment</div>
        </div>
    </div>
    """)

    st.html("""
    <p class="eyebrow">Sample Worksheet</p>
    <h1 class="page-title">Enter the field report</h1>
    <p class="page-subtitle">Twelve soil-test variables fed into the XGBoost model, classified as low, moderate, or very fertile.</p>
    """)

    col_id, col_presets = st.columns([1, 2.2], gap="medium")
    with col_id:
        sample_id = st.text_input("Plot / Sample ID", value=st.session_state["sample_id"], key="input_sample_id")

    with col_presets:
        st.markdown("<p style='font-size:0.75rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#73786B;margin-bottom:0.35rem;'>Quick Benchmarks &amp; Presets</p>", unsafe_allow_html=True)
        pb1, pb2, pb3, pb4 = st.columns(4)
        with pb1:
            if st.button("Low Preset", key="btn_low_fert", use_container_width=True):
                apply_preset("Low Fertility")
                st.rerun()
        with pb2:
            if st.button("Medium Preset", key="btn_med_fert", use_container_width=True):
                apply_preset("Medium Fertility")
                st.rerun()
        with pb3:
            if st.button("High Preset", key="btn_very_fert", use_container_width=True):
                apply_preset("Very Fertile")
                st.rerun()
        with pb4:
            if st.button("↺ Reset", key="btn_reset_all", use_container_width=True):
                reset_to_defaults()
                st.rerun()

    st.write("")

    # Two-column layout: Inputs on Left, Results & Charts on Right
    main_col, right_col = st.columns([1.55, 1.05], gap="large")
    user_inputs = {}

    with main_col:
        st.html("""<div class="lab-card-header"><em style="color:#73786B;margin-right:.3rem;">I</em> Laboratory measurements</div>""")
        feature_keys = list(FEATURES.keys())
        half = len(feature_keys) // 2
        sl, sr = st.columns(2, gap="large")

        with sl:
            for feat in feature_keys[:half]:
                meta = FEATURES[feat]
                curr_val = st.session_state[f"slider_{feat}"]
                st.html(f"""<div class="param-header"><div><span class="param-name">{meta["label"]}</span></div><div class="param-badge">{curr_val:.2f} <span style="font-size:.72rem;font-weight:normal;color:#6E7568;">{meta["unit"]}</span></div></div>""")
                user_inputs[feat] = st.slider(
                    meta["label"],
                    float(meta["min"]),
                    float(meta["max"]),
                    value=float(curr_val),
                    step=float(meta["step"]),
                    key=f"slider_{feat}",
                    label_visibility="collapsed"
                )
                st.write("")

        with sr:
            for feat in feature_keys[half:]:
                meta = FEATURES[feat]
                curr_val = st.session_state[f"slider_{feat}"]
                st.html(f"""<div class="param-header"><div><span class="param-name">{meta["label"]}</span></div><div class="param-badge">{curr_val:.2f} <span style="font-size:.72rem;font-weight:normal;color:#6E7568;">{meta["unit"]}</span></div></div>""")
                user_inputs[feat] = st.slider(
                    meta["label"],
                    float(meta["min"]),
                    float(meta["max"]),
                    value=float(curr_val),
                    step=float(meta["step"]),
                    key=f"slider_{feat}",
                    label_visibility="collapsed"
                )
                st.write("")

        # Scorecard strip
        st.html("""<div style="margin-top:1.2rem;margin-bottom:.6rem;">
<span style="font-weight:700;color:#1A3B1A;font-size:1rem;">Soil Nutrient Scorecard</span>
</div>""")

        scorecard_tiles = []
        for feat_k, meta_k in FEATURES.items():
            val = user_inputs.get(feat_k, st.session_state[f"slider_{feat_k}"])
            scorecard_tiles.append(f"""<div class="scorecard-tile">
<div class="scorecard-top"><span class="scorecard-name">{meta_k["label"]}</span></div>
<div class="scorecard-val">{val:.2f} <span style="font-size:.72rem;color:#787E71;font-weight:normal;">{meta_k["unit"]}</span></div>
</div>""")
        st.html(f'<div class="scorecard-grid">{"".join(scorecard_tiles)}</div>')

        st.html("<p style='font-size:.82rem;color:#73786B;margin-top:.4rem;'>This estimate is decision support, not a substitute for a calibrated laboratory test or local agronomist.</p>")
        st.write("")
        predict_clicked = st.button("Classify this sample \u2192", type="primary", use_container_width=False)

    with right_col:
        if predict_clicked:
            input_df = pd.DataFrame([user_inputs])
            try:
                pred_code = int(model.predict(input_df)[0])
                probabilities = model.predict_proba(input_df)[0]
                top_conf = float(np.max(probabilities) * 100)
                st.session_state["prediction_result"] = {
                    "pred_code": pred_code,
                    "probabilities": probabilities.tolist(),
                    "top_conf": top_conf,
                    "user_inputs": user_inputs.copy(),
                    "sample_id": sample_id,
                }
                fert_save = FERTILITY_MAP.get(pred_code, FERTILITY_MAP[1])
                save_history_record({
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Sample ID": sample_id if sample_id.strip() else "Untitled Sample",
                    "Fertility Class": fert_save["label"],
                    "Model Confidence (%)": round(top_conf, 1),
                    **{f"{k} ({FEATURES[k]['unit']})": round(v, 2) for k, v in user_inputs.items()}
                })
            except Exception as e:
                st.error(f"Prediction failed: {e}")

        result = st.session_state.get("prediction_result")

        if result is None:
            st.html("""<div style="background:#FFFFFF;border:1px solid #E0D9CE;border-radius:14px;padding:2.5rem 1.8rem;text-align:center;color:#73786B;margin-top:2.5rem;box-shadow:0 4px 14px rgba(0,0,0,0.03);">
<div style="font-size:2.2rem;margin-bottom:.8rem;">🌿</div>
<div style="font-family:'Newsreader',Georgia,serif;font-size:1.3rem;font-weight:600;color:#1A3B1A;margin-bottom:.5rem;">Prediction Result</div>
<div style="font-size:.9rem;line-height:1.55;">Adjust the soil parameters on the left<br>and click <strong>Classify this sample</strong>.</div>
</div>""")
        else:
            pred_code = result["pred_code"]
            top_conf = result["top_conf"]
            probabilities = result["probabilities"]
            res_sample_id = result["sample_id"]
            fert_info = FERTILITY_MAP.get(pred_code, FERTILITY_MAP[1])
            conf_score = int(round(top_conf))

            st.html(f"""<div class="pred-panel">
<div class="pred-panel-eyebrow">Prediction Result</div>
<div class="pred-panel-title">{fert_info["label"]}</div>
<div class="pred-confidence-ring">
<div class="pred-ring-num">{conf_score}%</div>
<div>
<div class="pred-ring-label">Model Confidence</div>
<div class="pred-ring-sub">{top_conf:.1f}% Certainty</div>
</div>
</div>
<div class="pred-class-chip {fert_info["chip"]}">{fert_info["health_rating"]}</div>
<div style="font-size:.86rem;color:#D8E6D8;line-height:1.55;margin-top:.5rem;">{fert_info["advice"]}</div>
<div style="margin-top:0.9rem;font-size:.76rem;color:#9BC496;font-weight:600;">Sample: {res_sample_id}</div>
</div>""")

            st.html('<div class="worksheet-card-title">XGBoost Class Probabilities</div>')
            df_prob = pd.DataFrame({
                "Fertility Class": ["Low Fertility", "Moderate Fertility", "Very Fertile"],
                "Probability (%)": [float(p * 100) for p in probabilities],
                "Color": ["#DC2626", "#D97706", "#16A34A"]
            })
            fig_prob = go.Figure(go.Bar(
                x=df_prob["Fertility Class"],
                y=df_prob["Probability (%)"],
                marker=dict(
                    color=df_prob["Color"],
                    line=dict(width=1, color="rgba(0,0,0,0.1)"),
                    cornerradius=6
                ),
                width=0.52,
                text=[f"{p:.1f}%" for p in df_prob["Probability (%)"]],
                textposition="outside",
                textfont=dict(size=12, family="Plus Jakarta Sans, sans-serif", color="#1E261D"),
                hovertemplate="<b>%{x}</b><br>Probability: %{y:.1f}%<extra></extra>"
            ))
            fig_prob.update_layout(
                dragmode=False,
                autosize=True,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=35, r=20, t=25, b=30),
                height=230,
                font=dict(family="Plus Jakarta Sans, sans-serif"),
                hoverlabel=dict(bgcolor="#1A3B1A", font_size=12, font_family="Plus Jakarta Sans, sans-serif", font_color="#FFFFFF"),
                xaxis=dict(
                    fixedrange=True,
                    showgrid=False,
                    tickfont=dict(size=11, family="Plus Jakarta Sans, sans-serif", color="#1E261D"),
                    linecolor="#D5CDC0",
                    linewidth=1
                ),
                yaxis=dict(
                    fixedrange=True,
                    showgrid=True,
                    gridcolor="rgba(128,128,128,0.18)",
                    tickfont=dict(size=10, family="Plus Jakarta Sans, sans-serif", color="#6E7568"),
                    range=[0, 118]
                ),
                showlegend=False
            )
            st.plotly_chart(
                fig_prob,
                use_container_width=True,
                config={
                    "displayModeBar": False,
                    "responsive": True,
                    "scrollZoom": False,
                    "doubleClick": "reset"
                }
            )

            st.html('<div class="worksheet-card-title" style="margin-top:1rem;">XGBoost Feature Importance</div>')
            importances = getattr(model, "feature_importances_", None)
            if importances is not None and len(importances) == len(FEATURES):
                df_imp = pd.DataFrame({
                    "Feature": [FEATURES[k]["label"] for k in FEATURES.keys()],
                    "Importance (%)": [float(imp * 100) for imp in importances]
                }).sort_values("Importance (%)", ascending=True)
            else:
                df_imp = pd.DataFrame({
                    "Feature": ["Organic Carbon (OC)", "Soil pH", "Zinc (Zn)", "Iron (Fe)", "Electrical Conductivity (EC)", "Manganese (Mn)", "Boron (B)", "Sulphur (S)", "Potassium (K)", "Copper (Cu)", "Phosphorus (P)", "Nitrogen (N)"],
                    "Importance (%)": [2.61, 2.89, 2.93, 2.99, 3.18, 3.24, 3.28, 3.56, 3.83, 4.47, 26.80, 40.23]
                }).sort_values("Importance (%)", ascending=True)

            max_val = df_imp["Importance (%)"].max()
            colors = []
            for val in df_imp["Importance (%)"]:
                if val >= 20:
                    colors.append("#1A3B1A")
                elif val >= 4.5:
                    colors.append("#2D6A30")
                else:
                    colors.append("#659164")

            fig_imp = go.Figure(go.Bar(
                y=df_imp["Feature"],
                x=df_imp["Importance (%)"],
                orientation="h",
                marker=dict(
                    color=colors,
                    line=dict(width=0.8, color="rgba(0,0,0,0.08)"),
                    cornerradius=4
                ),
                text=[f"{x:.1f}%" for x in df_imp["Importance (%)"]],
                textposition="outside",
                textfont=dict(size=10.5, family="Plus Jakarta Sans, sans-serif", color="#1E261D"),
                hovertemplate="<b>%{y}</b><br>Gain: %{x:.2f}%<extra></extra>"
            ))
            fig_imp.update_layout(
                dragmode=False,
                autosize=True,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=165, r=40, t=10, b=25),
                height=420,
                font=dict(family="Plus Jakarta Sans, sans-serif"),
                hoverlabel=dict(bgcolor="#1A3B1A", font_size=12, font_family="Plus Jakarta Sans, sans-serif", font_color="#FFFFFF"),
                xaxis=dict(
                    fixedrange=True,
                    title=dict(text="Split Gain (%)", font=dict(size=11, color="#555E50")),
                    showgrid=True,
                    gridcolor="rgba(128,128,128,0.18)",
                    range=[0, max_val * 1.22],
                    tickfont=dict(size=10, color="#6E7568")
                ),
                yaxis=dict(
                    fixedrange=True,
                    tickfont=dict(size=10.5, family="Plus Jakarta Sans, sans-serif", color="#1E261D"),
                    automargin=True
                ),
                showlegend=False
            )
            st.plotly_chart(
                fig_imp,
                use_container_width=True,
                config={
                    "displayModeBar": False,
                    "responsive": True,
                    "scrollZoom": False,
                    "doubleClick": "reset"
                }
            )

# ═════════════════════════════════════════════
#  TAB 2: HISTORY
# ═════════════════════════════════════════════
with tab_history:
    st.html("""
    <p class="eyebrow">Audit &amp; Archival Records</p>
    <h1 class="page-title">Soil Prediction History</h1>
    <p class="page-subtitle">Historical log of soil tests, classified fertility levels, and nutrient profiles.</p>
    """)
    df_hist = load_history()
    if not df_hist.empty and "Crop" in df_hist.columns:
        df_hist = df_hist.drop(columns=["Crop"])
    if df_hist.empty:
        st.info("No historical predictions recorded yet. Run a prediction on the Field Report & Analyzer tab.")
    else:
        total_runs = len(df_hist)
        high_cnt = len(df_hist[df_hist["Fertility Class"].str.contains("Highly|Very", na=False)])
        mod_cnt  = len(df_hist[df_hist["Fertility Class"].str.contains("Moderate|Medium", na=False)])
        low_cnt  = len(df_hist[df_hist["Fertility Class"].str.contains("Low", na=False)])
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.html(f'<div class="kpi-card"><div class="kpi-label">Total Tests</div><div class="kpi-num">{total_runs}</div></div>')
        with k2:
            st.html(f'<div class="kpi-card" style="border-top:3px solid #16A34A;"><div class="kpi-label">Very Fertile</div><div class="kpi-num" style="color:#16A34A;">{high_cnt}</div></div>')
        with k3:
            st.html(f'<div class="kpi-card" style="border-top:3px solid #D97706;"><div class="kpi-label">Medium Fertility</div><div class="kpi-num" style="color:#D97706;">{mod_cnt}</div></div>')
        with k4:
            st.html(f'<div class="kpi-card" style="border-top:3px solid #DC2626;"><div class="kpi-label">Low Fertility</div><div class="kpi-num" style="color:#DC2626;">{low_cnt}</div></div>')

        st.write("")
        cc1, cc2 = st.columns(2, gap="large")
        with cc1:
            st.html('<div class="worksheet-card-title">Fertility Distribution</div>')
            class_counts = df_hist["Fertility Class"].value_counts().reset_index()
            class_counts.columns = ["Fertility Class", "Count"]
            color_map = {
                "Highly Fertile": "#16A34A",
                "Very Fertile": "#16A34A",
                "Moderate Fertility": "#D97706",
                "Medium Fertility": "#D97706",
                "Low Fertility": "#DC2626"
            }
            fig_d = px.pie(
                class_counts,
                values="Count",
                names="Fertility Class",
                hole=0.55,
                color="Fertility Class",
                color_discrete_map=color_map
            )
            fig_d.update_traces(
                textposition="inside",
                textinfo="percent+label",
                hovertemplate="<b>%{label}</b><br>Count: %{value} (%{percent})<extra></extra>"
            )
            fig_d.update_layout(
                dragmode=False,
                autosize=True,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=10, b=10),
                height=280,
                font=dict(family="Plus Jakarta Sans, sans-serif"),
                hoverlabel=dict(bgcolor="#1A3B1A", font_size=12, font_family="Plus Jakarta Sans, sans-serif", font_color="#FFFFFF"),
                showlegend=False
            )
            st.plotly_chart(
                fig_d,
                use_container_width=True,
                config={
                    "displayModeBar": False,
                    "responsive": True,
                    "scrollZoom": False,
                    "doubleClick": "reset"
                }
            )
        with cc2:
            st.html('<div class="worksheet-card-title">Primary Nutrients (N-P-K) Logged</div>')
            n_col = [c for c in df_hist.columns if c.startswith("N ") or c == "N"]
            p_col = [c for c in df_hist.columns if c.startswith("P ") or c == "P"]
            k_col = [c for c in df_hist.columns if c.startswith("K ") or c == "K"]
            sample_col = "Sample ID" if "Sample ID" in df_hist.columns else df_hist.columns[1]
            if n_col and p_col and k_col:
                plot_df = df_hist.head(10).copy()
                plot_df = plot_df[[sample_col, n_col[0], p_col[0], k_col[0]]].melt(
                    id_vars=[sample_col],
                    value_vars=[n_col[0], p_col[0], k_col[0]],
                    var_name="Nutrient",
                    value_name="Value"
                )
                fig_bh = px.bar(
                    plot_df,
                    x=sample_col,
                    y="Value",
                    color="Nutrient",
                    barmode="group",
                    color_discrete_sequence=["#1A3B1A", "#5B8C5A", "#C5A869"]
                )
                fig_bh.update_layout(
                    dragmode=False,
                    autosize=True,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=20, r=20, t=10, b=25),
                    height=280,
                    font=dict(family="Plus Jakarta Sans, sans-serif"),
                    hoverlabel=dict(bgcolor="#1A3B1A", font_size=12, font_family="Plus Jakarta Sans, sans-serif", font_color="#FFFFFF"),
                    xaxis=dict(fixedrange=True, showgrid=False),
                    yaxis=dict(fixedrange=True, showgrid=True, gridcolor="rgba(128,128,128,0.18)"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(
                    fig_bh,
                    use_container_width=True,
                    config={
                        "displayModeBar": False,
                        "responsive": True,
                        "scrollZoom": False,
                        "doubleClick": "reset"
                    }
                )

        st.markdown("---")
        st.html('<div class="worksheet-card-title">Archival Log Entries</div>')
        cf1, cf2, ccsv, cdel = st.columns([2, 2, 1.5, 1.2])
        sample_name_col = "Sample ID" if "Sample ID" in df_hist.columns else "Plot / Sample"
        with cf1:
            search_query = st.text_input("Search by Sample ID", "", placeholder="Type sample name...")
        with cf2:
            unique_classes = ["All"] + list(df_hist["Fertility Class"].dropna().unique())
            selected_filter_cls = st.selectbox("Filter by Fertility Class", unique_classes)
        filtered_df = df_hist.copy()
        if search_query:
            filtered_df = filtered_df[filtered_df[sample_name_col].astype(str).str.contains(search_query, case=False, na=False)]
        if selected_filter_cls != "All":
            filtered_df = filtered_df[filtered_df["Fertility Class"] == selected_filter_cls]
        with ccsv:
            st.download_button(
                "Export CSV",
                data=filtered_df.to_csv(index=False).encode("utf-8"),
                file_name=f"soil_history_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with cdel:
            if st.button("Clear Log", use_container_width=True):
                if os.path.exists(HISTORY_PATH):
                    os.remove(HISTORY_PATH)
                st.success("History cleared!")
                st.rerun()
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.html("""<div style="text-align:center;color:#787E71;font-size:.84rem;padding-bottom:1.5rem;">
Soil Fertility Analyzer &middot; Powered by XGBoost &amp; Streamlit &middot; Field Laboratory Edition
</div>""")
