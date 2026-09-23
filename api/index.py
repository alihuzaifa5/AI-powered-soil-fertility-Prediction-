import json
import os
import joblib
import pandas as pd
import numpy as np
from http.server import BaseHTTPRequestHandler

# ─────────────────────────────────────────────
#  GLOBAL MODEL CACHING (persists across warm lambdas)
# ─────────────────────────────────────────────
_MODEL = None

def get_model():
    global _MODEL
    if _MODEL is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "best_xgboost_model.joblib")
        if not os.path.exists(model_path):
            model_path = os.path.join(os.getcwd(), "best_xgboost_model.joblib")
        _MODEL = joblib.load(model_path)
    return _MODEL

FERTILITY_MAP = {
    0: {
        "label": "Low Fertility",
        "health_rating": "Sub-optimal / Deficient",
        "chip": "chip-low",
        "advice": "Soil profile shows critical nutrient depletion or severe chemical imbalance. Immediate soil amendment and balanced fertilization required.",
    },
    1: {
        "label": "Moderate Fertility",
        "health_rating": "Moderately Balanced",
        "chip": "chip-medium",
        "advice": "Productive soil with minor localized nutrient limitations. Targeted secondary/micronutrient supplementation recommended for peak yields.",
    },
    2: {
        "label": "Very Fertile",
        "health_rating": "Optimum Soil Health",
        "chip": "chip-high",
        "advice": "Excellent macro and micronutrient reserve. Highly supportive for multi-crop production. Maintain current organic stewardship.",
    },
}

FEATURE_KEYS = ['N', 'P', 'K', 'pH', 'EC', 'OC', 'S', 'Zn', 'Fe', 'Cu', 'Mn', 'B']

FEATURE_LABELS = {
    "N": "Nitrogen (N)",
    "P": "Phosphorus (P)",
    "K": "Potassium (K)",
    "pH": "Soil pH",
    "EC": "Electrical Conductivity",
    "OC": "Organic Carbon",
    "S": "Sulphur (S)",
    "Zn": "Zinc (Zn)",
    "Fe": "Iron (Fe)",
    "Cu": "Copper (Cu)",
    "Mn": "Manganese (Mn)",
    "B": "Boron (B)",
}

class handler(BaseHTTPRequestHandler):
    def _send_json(self, status_code, payload):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        self._send_json(200, {
            "status": "healthy",
            "service": "AI Soil Fertility Prediction API",
            "features": FEATURE_KEYS
        })

    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(raw_body) if raw_body else {}

            sample_id = data.get("sample_id", "Plot-001")
            user_inputs = {k: float(data.get(k, 0.0)) for k in FEATURE_KEYS}
            df = pd.DataFrame([user_inputs])

            model = get_model()
            pred_code = int(model.predict(df)[0])
            probabilities = model.predict_proba(df)[0].tolist()
            top_conf = float(max(probabilities) * 100)
            fert_info = FERTILITY_MAP.get(pred_code, FERTILITY_MAP[1])

            # Feature importances
            importances = getattr(model, "feature_importances_", None)
            if importances is not None and len(importances) == len(FEATURE_KEYS):
                imp_list = [
                    {"feature": FEATURE_LABELS[k], "importance": round(float(importances[i]) * 100, 2)}
                    for i, k in enumerate(FEATURE_KEYS)
                ]
            else:
                default_imp = [
                    ("Organic Carbon", 2.61), ("Soil pH", 2.89), ("Zinc", 2.93),
                    ("Iron", 2.99), ("Elec. Cond.", 3.18), ("Manganese", 3.24),
                    ("Boron", 3.28), ("Sulphur", 3.56), ("Potassium", 3.83),
                    ("Copper", 4.47), ("Phosphorus", 26.80), ("Nitrogen", 40.23)
                ]
                imp_list = [{"feature": f, "importance": v} for f, v in default_imp]

            # Sort importances ascending for horizontal bar chart
            imp_list.sort(key=lambda x: x["importance"])

            response_data = {
                "success": True,
                "sample_id": sample_id,
                "pred_code": pred_code,
                "label": fert_info["label"],
                "health_rating": fert_info["health_rating"],
                "chip": fert_info["chip"],
                "advice": fert_info["advice"],
                "confidence": round(top_conf, 1),
                "probabilities": [
                    {"label": "Low Fertility", "probability": round(probabilities[0] * 100, 1), "color": "#C62828"},
                    {"label": "Moderate Fertility", "probability": round(probabilities[1] * 100, 1), "color": "#D97706"},
                    {"label": "Very Fertile", "probability": round(probabilities[2] * 100, 1), "color": "#2D5A27"},
                ],
                "feature_importances": imp_list,
                "inputs": user_inputs,
            }
            self._send_json(200, response_data)

        except Exception as e:
            self._send_json(500, {
                "success": False,
                "error": str(e)
            })
