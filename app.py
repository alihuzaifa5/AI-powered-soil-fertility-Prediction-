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
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  FIELD LABORATORY STYLE
# ─────────────────────────────────────────────
st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;0,6..72,700;1,6..72,400;1,6..72,600&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
.stApp{background:#F5F3EC;color:#1E261D;font-family:'Plus Jakarta Sans',-apple-system,sans-serif}
h1,h2,h3{font-family:'Newsreader',Georgia,serif!important;color:#1A3B1A;font-weight:600}
.eyebrow{font-size:.7rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:#73786B;margin-bottom:.2rem}
.page-title{font-family:'Newsreader',Georgia,serif;font-size:2.55rem;line-height:1.1;color:#1A3B1A;margin:0 0 .3rem;font-weight:700}
.page-subtitle{font-size:.95rem;color:#4F574A;line-height:1.55;margin-bottom:1.4rem}
.top-bar{display:flex;align-items:center;gap:.55rem;margin-bottom:.2rem}
.top-bar-icon{background:#1A3B1A;color:#fff;border-radius:8px;width:34px;height:34px;display:flex;align-items:center;justify-content:center;font-size:1.1rem}
.top-bar-lab{font-size:.7rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:#73786B}
.top-bar-title{font-family:'Newsreader',Georgia,serif;font-size:1.45rem;font-weight:700;color:#1A3B1A}
.lab-card-header{font-family:'Newsreader',Georgia,serif;font-size:1.1rem;font-weight:600;color:#1A3B1A;border-bottom:1px solid #EAE3D8;padding-bottom:.65rem;margin-bottom:1.3rem;display:flex;align-items:center;gap:.5rem}
.param-header{display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:.12rem}
.param-name{font-weight:600;color:#1F2B1D;font-size:.92rem}
.param-target{font-size:.67rem;letter-spacing:.07em;text-transform:uppercase;color:#878D80;font-weight:600;margin-top:.08rem}
.param-badge{background:#F0ECE2;border-bottom:2px solid #8FA689;padding:.15rem .6rem;border-radius:4px;font-weight:700;font-size:.92rem;color:#1A3B1A;text-align:right;min-width:70px}
div.st-key-btn_low_fert button{background:#F5F3EC!important;border:1.5px solid #C0B8AC!important;border-radius:30px!important;box-shadow:none!important}
div.st-key-btn_low_fert button p,div.st-key-btn_low_fert button span{color:#C62828!important;font-weight:600!important;font-size:.88rem!important}
div.st-key-btn_low_fert button:hover{background:#FDF0F0!important;border-color:#C62828!important}
div.st-key-btn_med_fert button{background:#F5F3EC!important;border:1.5px solid #C0B8AC!important;border-radius:30px!important;box-shadow:none!important}
div.st-key-btn_med_fert button p,div.st-key-btn_med_fert button span{color:#B45309!important;font-weight:600!important;font-size:.88rem!important}
div.st-key-btn_med_fert button:hover{background:#FEF5E7!important;border-color:#B45309!important}
div.st-key-btn_very_fert button{background:#F5F3EC!important;border:1.5px solid #C0B8AC!important;border-radius:30px!important;box-shadow:none!important}
div.st-key-btn_very_fert button p,div.st-key-btn_very_fert button span{color:#1A6B35!important;font-weight:600!important;font-size:.88rem!important}
div.st-key-btn_very_fert button:hover{background:#EDF7F0!important;border-color:#1A6B35!important}
div.st-key-btn_reset_all button{background:#F5F3EC!important;border:1.5px solid #C0B8AC!important;border-radius:30px!important;box-shadow:none!important}
div.st-key-btn_reset_all button p,div.st-key-btn_reset_all button span{color:#555E50!important;font-weight:600!important;font-size:.88rem!important}
div.stButton>button[kind="primary"]{background:#1A3B1A!important;color:#fff!important;border-radius:8px!important;padding:.6rem 1.8rem!important;font-weight:600!important;font-size:1rem!important;border:none!important;box-shadow:0 2px 8px rgba(26,59,26,.22)!important;transition:all .18s ease!important}
div.stButton>button[kind="primary"]:hover{background:#122B12!important;box-shadow:0 4px 14px rgba(26,59,26,.32)!important}
.pred-panel{background:#1A3B1A;border-radius:14px;padding:1.4rem 1.5rem 1.2rem;color:#fff;margin-bottom:1rem}
.pred-panel-eyebrow{font-size:.65rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:#8FB88A;margin-bottom:.4rem}
.pred-panel-title{font-family:'Newsreader',Georgia,serif;font-size:1.8rem;font-weight:700;color:#fff;line-height:1.2;margin-bottom:1.1rem}
.pred-confidence-ring{display:flex;align-items:center;gap:1rem;background:rgba(255,255,255,.07);border-radius:10px;padding:.75rem 1rem;margin-bottom:.85rem}
.pred-ring-num{font-family:'Newsreader',Georgia,serif;font-size:2.6rem;font-weight:700;color:#fff;line-height:1}
.pred-ring-label{font-size:.72rem;color:#AEBDAE;text-transform:uppercase;letter-spacing:.1em}
.pred-ring-sub{font-size:.82rem;color:#CDDACD;margin-top:.2rem}
.pred-class-chip{display:inline-block;padding:.3rem .9rem;border-radius:30px;font-size:.8rem;font-weight:700;letter-spacing:.06em;margin-bottom:.6rem}
.chip-low{background:rgba(198,40,40,.18);color:#FF8A8A;border:1px solid rgba(198,40,40,.3)}
.chip-medium{background:rgba(217,119,6,.18);color:#FFBE6E;border:1px solid rgba(217,119,6,.3)}
.chip-high{background:rgba(43,102,40,.35);color:#86EFAC;border:1px solid rgba(43,102,40,.4)}
.nutr-panel{background:#fff;border:1px solid #E0D9CE;border-radius:14px;padding:1.2rem 1.5rem .8rem;box-shadow:0 2px 10px rgba(26,59,26,.04);margin-bottom:1rem}
.nutr-panel-title{font-family:'Newsreader',Georgia,serif;font-size:1rem;font-weight:600;color:#1A3B1A;border-bottom:1px solid #EAE3D8;padding-bottom:.55rem;margin-bottom:.9rem}
.scorecard-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:.7rem;margin-bottom:1.4rem}
.scorecard-tile{background:#fff;border:1px solid #E0D9CE;border-radius:10px;padding:.8rem .95rem;display:flex;flex-direction:column;justify-content:space-between;transition:transform .15s ease}
.scorecard-tile:hover{transform:translateY(-2px);box-shadow:0 4px 10px rgba(0,0,0,.05)}
.scorecard-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:.35rem}
.scorecard-name{font-weight:700;font-size:.83rem;color:#1E261D}
.scorecard-status-dot{width:9px;height:9px;border-radius:50%;display:inline-block}
.dot-green{background:#2E7D32;box-shadow:0 0 5px rgba(46,125,50,.4)}
.dot-amber{background:#E68A00;box-shadow:0 0 5px rgba(230,138,0,.4)}
.dot-red{background:#C62828;box-shadow:0 0 5px rgba(198,40,40,.4)}
.scorecard-val{font-size:1.18rem;font-weight:800;color:#1A3B1A;font-family:'Newsreader',Georgia,serif}
.scorecard-target{font-size:.65rem;color:#7A8072;letter-spacing:.04em;margin-top:.15rem}
.kpi-card{background:#fff;border:1px solid #E0D9CE;border-radius:12px;padding:1rem 1.3rem;text-align:center;box-shadow:0 2px 6px rgba(0,0,0,.02)}
.kpi-num{font-size:1.9rem;font-weight:800;font-family:'Newsreader',Georgia,serif;color:#1A3B1A;margin:.1rem 0}
.kpi-label{font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;color:#707767;font-weight:600}
.worksheet-card-title{font-family:'Newsreader',Georgia,serif;font-size:1.15rem;color:#1A3B1A;font-weight:600;border-bottom:1px dashed #D6CCBD;padding-bottom:.65rem;margin-bottom:1.2rem}
div[data-testid="stTextInput"] input{background:#fff!important;border:1px solid #D0C9BD!important;border-radius:8px!important;color:#1E261D!important}
.stTabs [data-baseweb="tab-list"]{gap:8px;background:transparent;border-bottom:2px solid #DDD6CA;padding-bottom:4px;margin-bottom:1.4rem}
.stTabs [data-baseweb="tab"]{font-family:'Newsreader',Georgia,serif;font-size:1.1rem;font-weight:600;color:#555C4F;padding:.45rem 1.3rem;border-radius:8px 8px 0 0}
.stTabs [aria-selected="true"]{color:#1A3B1A!important;border-bottom:3px solid #1A3B1A!important}
#MainMenu{visibility:hidden}footer{visibility:hidden}
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
#  FEATURE DEFINITIONS
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
    0: {"label":"Low Fertility",   "chip":"chip-low",   "color":"#FF8A8A","health_rating":"Sub-Optimal",  "advice":"Soil requires comprehensive nutrient amendment. Apply NPK fertilizers, compost, and trace micro-minerals."},
    1: {"label":"Moderate Fertility","chip":"chip-medium","color":"#FFBE6E","health_rating":"Moderate / Fair","advice":"Moderate fertility. Supplement deficient nutrients and maintain balanced crop management."},
    2: {"label":"Very Fertile",    "chip":"chip-high",  "color":"#86EFAC","health_rating":"Excellent",    "advice":"Excellent condition! Retain soil health through organic mulch, minimal tillage, and periodic monitoring."},
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

for feat, meta in FEATURES.items():
    if f"slider_{feat}" not in st.session_state:
        st.session_state[f"slider_{feat}"] = meta["default"]
if "sample_id" not in st.session_state:
    st.session_state["sample_id"] = "Sample #01"
if "prediction_result" not in st.session_state:
    st.session_state["prediction_result"] = None

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab_analyzer, tab_history = st.tabs(["Field Report & Analyzer", "Prediction History Dashboard"])

# ═════════════════════════════════════════════
#  TAB 1
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

    sample_id = st.text_input("Plot / Sample", value=st.session_state["sample_id"], key="input_sample_id")

    st.write("")

    # Two-column layout
    main_col, right_col = st.columns([1.65, 1], gap="large")
    user_inputs = {}

    with main_col:
        st.html("""<div class="lab-card-header"><em style="color:#73786B;margin-right:.3rem;">I</em> Laboratory measurements</div>""")
        feature_keys = list(FEATURES.keys())
        half = len(feature_keys) // 2
        sl, sr = st.columns(2, gap="large")

        with sl:
            for feat in feature_keys[:half]:
                meta = FEATURES[feat]; curr_val = st.session_state[f"slider_{feat}"]
                st.html(f"""<div class="param-header"><div><span class="param-name">{meta["label"]}</span></div><div class="param-badge">{curr_val:.2f} <span style="font-size:.72rem;font-weight:normal;color:#6E7568;">{meta["unit"]}</span></div></div>""")
                user_inputs[feat] = st.slider(meta["label"], float(meta["min"]), float(meta["max"]), step=float(meta["step"]), key=f"slider_{feat}", label_visibility="collapsed")
                st.write("")

        with sr:
            for feat in feature_keys[half:]:
                meta = FEATURES[feat]; curr_val = st.session_state[f"slider_{feat}"]
                st.html(f"""<div class="param-header"><div><span class="param-name">{meta["label"]}</span></div><div class="param-badge">{curr_val:.2f} <span style="font-size:.72rem;font-weight:normal;color:#6E7568;">{meta["unit"]}</span></div></div>""")
                user_inputs[feat] = st.slider(meta["label"], float(meta["min"]), float(meta["max"]), step=float(meta["step"]), key=f"slider_{feat}", label_visibility="collapsed")
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

        st.html("<p style='font-size:.8rem;color:#73786B;margin-top:.5rem;'>This estimate is decision support, not a substitute for a calibrated laboratory test or local agronomist.</p>")
        st.write("")
        predict_clicked = st.button("Classify this sample \u2192", type="primary", use_container_width=False)

    with right_col:
        if predict_clicked:
            input_df = pd.DataFrame([user_inputs])
            try:
                pred_code    = int(model.predict(input_df)[0])
                probabilities = model.predict_proba(input_df)[0]
                top_conf     = float(np.max(probabilities) * 100)
                st.session_state["prediction_result"] = {
                    "pred_code": pred_code, "probabilities": probabilities.tolist(),
                    "top_conf": top_conf,
                    "user_inputs": user_inputs.copy(), "sample_id": sample_id,
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
            st.html("""<div style="background:#fff;border:1px solid #E0D9CE;border-radius:14px;padding:2.2rem 1.5rem;text-align:center;color:#73786B;margin-top:2.5rem;">
<div style="font-size:2rem;margin-bottom:.8rem;">🌿</div>
<div style="font-family:'Newsreader',Georgia,serif;font-size:1.2rem;font-weight:600;color:#1A3B1A;margin-bottom:.5rem;">Prediction Result</div>
<div style="font-size:.88rem;line-height:1.5;">Adjust the soil parameters on the left<br>and click <strong>Classify this sample</strong>.</div>
</div>""")
        else:
            pred_code  = result["pred_code"]
            top_conf   = result["top_conf"]
            probabilities = result["probabilities"]
            res_inputs = result["user_inputs"]
            res_sample_id = result["sample_id"]
            fert_info  = FERTILITY_MAP.get(pred_code, FERTILITY_MAP[1])
            conf_score = int(round(top_conf))

            st.html(f"""<div class="pred-panel">
<div class="pred-panel-eyebrow">Prediction Result</div>
<div class="pred-panel-title">{fert_info["label"]}</div>
<div class="pred-confidence-ring">
<div class="pred-ring-num">{conf_score}</div>
<div>
<div class="pred-ring-label">Model Confidence</div>
<div class="pred-ring-sub">{top_conf:.1f}%</div>
</div>
</div>
<div class="pred-class-chip {fert_info["chip"]}">{fert_info["health_rating"]}</div>
<div style="font-size:.84rem;color:#CDDACD;line-height:1.55;margin-top:.6rem;">{fert_info["advice"]}</div>
<div style="margin-top:1rem;font-size:.75rem;color:#8FB88A;">{res_sample_id}</div>
</div>""")

            st.html('<p class="worksheet-card-title">XGBoost Class Probabilities</p>')
            df_prob = pd.DataFrame({
                "Fertility Class": ["Low Fertility", "Moderate Fertility", "Very Fertile"],
                "Probability (%)": [p * 100 for p in probabilities],
                "Color": ["#C62828", "#D97706", "#2D5A27"]
            })
            fig_prob = go.Figure(go.Bar(
                x=df_prob["Fertility Class"],
                y=df_prob["Probability (%)"],
                marker_color=df_prob["Color"],
                width=0.45,
                text=[f"{p:.1f}%" for p in df_prob["Probability (%)"]],
                textposition="auto",
                hovertemplate="<b>%{x}</b>: %{y:.1f}%<extra></extra>"
            ))
            fig_prob.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF", margin=dict(l=10,r=10,t=10,b=10), height=200, xaxis=dict(showgrid=False,tickfont=dict(size=11,color="#1A3B1A")), yaxis=dict(showgrid=True,gridcolor="#EAE3D8",tickfont=dict(size=10,color="#73786B"),range=[0,105]), showlegend=False)
            st.plotly_chart(fig_prob, use_container_width=True, config={"displayModeBar": False})

            st.html('<p class="worksheet-card-title">XGBoost Feature Importance</p>')
            importances = getattr(model, "feature_importances_", None)
            if importances is not None and len(importances) == len(FEATURES):
                df_imp = pd.DataFrame({
                    "Feature": [FEATURES[k]["label"] for k in FEATURES.keys()],
                    "Importance (%)": [imp * 100 for imp in importances]
                }).sort_values("Importance (%)", ascending=True)
            else:
                df_imp = pd.DataFrame({
                    "Feature": ["Organic Carbon", "Soil pH", "Zinc", "Iron", "Elec. Cond.", "Manganese", "Boron", "Sulphur", "Potassium", "Copper", "Phosphorus", "Nitrogen"],
                    "Importance (%)": [2.61, 2.89, 2.93, 2.99, 3.18, 3.24, 3.28, 3.56, 3.83, 4.47, 26.80, 40.23]
                })
            fig_imp = go.Figure(go.Bar(
                y=df_imp["Feature"],
                x=df_imp["Importance (%)"],
                orientation="h",
                marker_color="#2A5329",
                hovertemplate="<b>%{y}</b>: %{x:.2f}% gain<extra></extra>"
            ))
            fig_imp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF", margin=dict(l=10,r=10,t=10,b=10), height=290, xaxis=dict(title="Decision Split Gain (%)", showgrid=True, gridcolor="#EAE3D8"), yaxis=dict(tickfont=dict(size=10)))
            st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar": False})



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
        k1,k2,k3,k4 = st.columns(4)
        with k1: st.html(f'<div class="kpi-card"><div class="kpi-label">Total Tests</div><div class="kpi-num">{total_runs}</div></div>')
        with k2: st.html(f'<div class="kpi-card" style="border-top:3px solid #1A3B1A;"><div class="kpi-label">Very Fertile</div><div class="kpi-num" style="color:#1A3B1A;">{high_cnt}</div></div>')
        with k3: st.html(f'<div class="kpi-card" style="border-top:3px solid #D97706;"><div class="kpi-label">Medium Fertility</div><div class="kpi-num" style="color:#D97706;">{mod_cnt}</div></div>')
        with k4: st.html(f'<div class="kpi-card" style="border-top:3px solid #C62828;"><div class="kpi-label">Low Fertility</div><div class="kpi-num" style="color:#C62828;">{low_cnt}</div></div>')

        st.write("")
        cc1, cc2 = st.columns(2, gap="large")
        with cc1:
            st.html('<p class="worksheet-card-title">Fertility Distribution</p>')
            class_counts = df_hist["Fertility Class"].value_counts().reset_index()
            class_counts.columns = ["Fertility Class","Count"]
            color_map = {"Highly Fertile":"#1A3B1A","Very Fertile":"#1A3B1A","Moderate Fertility":"#D97706","Medium Fertility":"#D97706","Low Fertility":"#C62828"}
            fig_d = px.pie(class_counts, values="Count", names="Fertility Class", hole=0.55, color="Fertility Class", color_discrete_map=color_map)
            fig_d.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=20,r=20,t=10,b=10), height=280, legend=dict(orientation="h",yanchor="bottom",y=-0.2,xanchor="center",x=0.5))
            st.plotly_chart(fig_d, use_container_width=True)
        with cc2:
            st.html('<p class="worksheet-card-title">Primary Nutrients (N-P-K) Logged</p>')
            n_col = [c for c in df_hist.columns if c.startswith("N ") or c=="N"]
            p_col = [c for c in df_hist.columns if c.startswith("P ") or c=="P"]
            k_col = [c for c in df_hist.columns if c.startswith("K ") or c=="K"]
            sample_col = "Sample ID" if "Sample ID" in df_hist.columns else df_hist.columns[1]
            if n_col and p_col and k_col:
                plot_df = df_hist.head(10).copy()
                plot_df = plot_df[[sample_col,n_col[0],p_col[0],k_col[0]]].melt(id_vars=[sample_col],value_vars=[n_col[0],p_col[0],k_col[0]],var_name="Nutrient",value_name="Value")
                fig_bh = px.bar(plot_df, x=sample_col, y="Value", color="Nutrient", barmode="group", color_discrete_sequence=["#1A3B1A","#8FA689","#C5B99F"])
                fig_bh.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF", margin=dict(l=20,r=20,t=10,b=10), height=280, xaxis=dict(showgrid=False), yaxis=dict(showgrid=True,gridcolor="#EAE3D8"), legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
                st.plotly_chart(fig_bh, use_container_width=True)

        st.markdown("---")
        st.html('<p class="worksheet-card-title">Archival Log Entries</p>')
        cf1,cf2,ccsv,cdel = st.columns([2,2,1.5,1.2])
        sample_name_col = "Sample ID" if "Sample ID" in df_hist.columns else "Plot / Sample"
        with cf1: search_query = st.text_input("Search by Sample ID","",placeholder="Type sample name...")
        with cf2:
            unique_classes = ["All"] + list(df_hist["Fertility Class"].dropna().unique())
            selected_filter_cls = st.selectbox("Filter by Fertility Class", unique_classes)
        filtered_df = df_hist.copy()
        if search_query: filtered_df = filtered_df[filtered_df[sample_name_col].astype(str).str.contains(search_query, case=False, na=False)]
        if selected_filter_cls != "All": filtered_df = filtered_df[filtered_df["Fertility Class"]==selected_filter_cls]
        with ccsv:
            st.download_button("Export CSV", data=filtered_df.to_csv(index=False).encode("utf-8"), file_name=f"soil_history_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv", use_container_width=True)
        with cdel:
            if st.button("Clear Log", use_container_width=True):
                if os.path.exists(HISTORY_PATH): os.remove(HISTORY_PATH)
                st.success("History cleared!"); st.rerun()
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.html("""<div style="text-align:center;color:#787E71;font-size:.82rem;padding-bottom:1.5rem;">
Soil Fertility Analyzer &middot; Powered by XGBoost &amp; Streamlit &middot; Field Laboratory Edition
</div>""")
