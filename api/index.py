import json
import os
import xgboost as xgb
import numpy as np
from http.server import BaseHTTPRequestHandler

# ─────────────────────────────────────────────
#  GLOBAL BOOSTER CACHING (fast, lightweight memory footprint)
# ─────────────────────────────────────────────
_BOOSTER = None

def get_booster():
    global _BOOSTER
    if _BOOSTER is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "model.json")
        if not os.path.exists(model_path):
            model_path = os.path.join(os.getcwd(), "model.json")
        _BOOSTER = xgb.Booster()
        _BOOSTER.load_model(model_path)
    return _BOOSTER

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
            "model_format": "Native XGBoost JSON (Lightweight Serverless)",
            "features": FEATURE_KEYS
        })

    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(raw_body) if raw_body else {}

            sample_id = data.get("sample_id", "Plot-001")
            user_inputs = {k: float(data.get(k, 0.0)) for k in FEATURE_KEYS}
            
            # Format input array for XGBoost DMatrix
            row_vals = [user_inputs[k] for k in FEATURE_KEYS]
            np_arr = np.array([row_vals], dtype=np.float32)
            dmatrix = xgb.DMatrix(np_arr, feature_names=FEATURE_KEYS)

            booster = get_booster()
            probs = booster.predict(dmatrix)[0]
            pred_code = int(np.argmax(probs))
            top_conf = float(probs[pred_code] * 100)
            fert_info = FERTILITY_MAP.get(pred_code, FERTILITY_MAP[1])

            # Calculate feature gain percentages
            raw_scores = booster.get_score(importance_type="gain")
            total_gain = sum(raw_scores.values()) if raw_scores else 1.0
            
            imp_list = []
            for k in FEATURE_KEYS:
                gain_val = raw_scores.get(k, 0.0)
                pct = round((gain_val / total_gain) * 100, 2) if total_gain > 0 else 0.0
                imp_list.append({"feature": FEATURE_LABELS[k], "importance": pct})
            
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
                    {"label": "Low Fertility", "probability": round(float(probs[0]) * 100, 1), "color": "#C62828"},
                    {"label": "Moderate Fertility", "probability": round(float(probs[1]) * 100, 1), "color": "#D97706"},
                    {"label": "Very Fertile", "probability": round(float(probs[2]) * 100, 1), "color": "#2D5A27"},
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
