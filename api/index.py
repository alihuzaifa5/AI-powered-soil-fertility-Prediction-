import json
import math
import os
from http.server import BaseHTTPRequestHandler

# ─────────────────────────────────────────────
#  ZERO-DEPENDENCY PURE PYTHON XGBOOST INFERENCE ENGINE
#  Bundle size: ~1.5 MB (0 MB external pip packages, 100% Vercel Serverless compliant)
# ─────────────────────────────────────────────

_MODEL_DATA = None
_TREES = None
_TREE_INFO = None
_FEATURE_NAMES = None
_FEATURE_IMPORTANCES = None

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

def load_model():
    global _MODEL_DATA, _TREES, _TREE_INFO, _FEATURE_NAMES, _FEATURE_IMPORTANCES
    if _MODEL_DATA is not None:
        return

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(curr_dir)
    candidates = [
        os.path.join(curr_dir, "model.json"),
        os.path.join(base_dir, "model.json"),
        os.path.join(os.getcwd(), "model.json"),
        os.path.join(os.getcwd(), "api", "model.json"),
    ]
    model_path = None
    for c in candidates:
        if os.path.exists(c):
            model_path = c
            break

    if not model_path:
        raise FileNotFoundError("Could not find model.json in any expected location")

    with open(model_path, "r", encoding="utf-8") as f:
        _MODEL_DATA = json.load(f)

    gb = _MODEL_DATA["learner"]["gradient_booster"]["model"]
    _TREES = gb["trees"]
    _TREE_INFO = gb["tree_info"]
    _FEATURE_NAMES = _MODEL_DATA["learner"]["feature_names"]

    # Pre-calculate feature importances from gain loss changes
    f_gains = {k: 0.0 for k in _FEATURE_NAMES}
    for t in _TREES:
        for s_idx, loss in zip(t["split_indices"], t["loss_changes"]):
            if s_idx < len(_FEATURE_NAMES):
                f_gains[_FEATURE_NAMES[s_idx]] += float(loss)
    tot = sum(f_gains.values()) or 1.0
    imp = [
        {"feature": FEATURE_LABELS.get(k, k), "importance": round(v / tot * 100, 2)}
        for k, v in f_gains.items()
    ]
    imp.sort(key=lambda x: x["importance"])
    _FEATURE_IMPORTANCES = imp


def predict_fertility(user_inputs):
    load_model()
    margins = [0.0, 0.0, 0.0]
    vals = [float(user_inputs.get(k, 0.0)) for k in _FEATURE_NAMES]

    for t, cls in zip(_TREES, _TREE_INFO):
        curr = 0
        lefts = t["left_children"]
        rights = t["right_children"]
        splits = t["split_indices"]
        conds = t["split_conditions"]
        weights = t["base_weights"]

        while lefts[curr] != -1:
            f_idx = splits[curr]
            val = vals[f_idx]
            if val < conds[curr]:
                curr = lefts[curr]
            else:
                curr = rights[curr]
        margins[cls] += weights[curr]

    max_m = max(margins)
    exp_m = [math.exp(v - max_m) for v in margins]
    sum_exp = sum(exp_m)
    probs = [v / sum_exp for v in exp_m]
    return probs


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
            "engine": "Zero-Dependency Native Tree Evaluator",
            "features": list(FEATURE_LABELS.keys())
        })

    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(raw_body) if raw_body else {}

            sample_id = data.get("sample_id", "Plot-001")
            user_inputs = {k: float(data.get(k, 0.0)) for k in FEATURE_LABELS.keys()}

            probs = predict_fertility(user_inputs)
            pred_code = int(probs.index(max(probs)))
            top_conf = float(probs[pred_code] * 100)
            fert_info = FERTILITY_MAP.get(pred_code, FERTILITY_MAP[1])

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
                "feature_importances": _FEATURE_IMPORTANCES,
                "inputs": user_inputs,
            }
            self._send_json(200, response_data)

        except Exception as e:
            self._send_json(500, {
                "success": False,
                "error": str(e)
            })
