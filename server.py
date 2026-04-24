from fastapi import FastAPI, Request, HTTPException
import os

from embedder import get_embedding
from preprocessing.preprocess_image import ImagePreprocessingError
from preprocessing.check_image_quality import check_image_quality
from preprocessing.skin_check import is_skin
import joblib
import numpy as np
import uvicorn

app = FastAPI()

BASE = os.path.join(os.path.dirname(__file__), "trained_data_two_stage_new")
CONFIDENCE_THRESHOLD = 0.3
DEMOTION_THRESHOLDS = {
    "eczema": 0.75,
}

print("Loading Stage 1 condition model...")
clf_stage1, le_stage1 = joblib.load(os.path.join(BASE, "stage1_condition.pkl"))
print(f"  Classes: {list(le_stage1.classes_)}")

SEVERITY_CONDITIONS = {
    "acne-blackheads": "stage2_acne-blackheads.pkl",
    "acne-fungal":     "stage2_acne-fungal.pkl",
    "acne-papules":    "stage2_acne-papules.pkl",
    "acne-pustules":   "stage2_acne-pustules.pkl",
    "acne-whiteheads": "stage2_acne-whiteheads.pkl",
    "eczema":          "stage2_eczema.pkl",
    "enlarged-pores":  "stage2_enlarged-pores.pkl",
    "melasma":         "stage2_melasma.pkl",
    "milia":           "stage2_milia.pkl",
    "post-inflammatory-erythema":         "stage2_post-inflammatory-erythema.pkl",
    "post-inflammatory-hyperpigmentation": "stage2_post-inflammatory-hyperpigmentation.pkl",
}

print("Loading severity classifiers...")
severity_classifiers = {
    condition: joblib.load(os.path.join(BASE, filename))
    for condition, filename in SEVERITY_CONDITIONS.items()
}
print("All models loaded. Server ready.\n")

@app.post("/analyze")
async def analyze(request: Request):
    try:
        # ------------------ FIRST PHASE --------------------------------------
        image_data = await request.body()
        print("\n>>> New request received")

        if not is_skin(image_data):
            print(f"  >>> REJECTED: Skin check failed")
            return {
                "primary_prediction": "out-of-scope",
                "severity": None,
                "confidence": 0.0,
                "candidates": [],
            }

        print("  Extracting embedding...")
        emb = get_embedding(image_data).reshape(1, -1)
        print(f"  Embedding shape: {emb.shape}")

        print("  Running Stage 1...")

        proba1 = clf_stage1.predict_proba(emb)[0]
        top_idx = np.argsort(proba1)[::-1]
        candidates  = [
            {"label": le_stage1.classes_[i], "score": float(proba1[i])}
            for i in top_idx
        ]

        top_label = candidates[0]["label"]
        top_score = candidates[0]["score"]
        print(f"  Stage 1 results (all {len(candidates)}):")
        for c in candidates:
            print(f"    {c['label']:<45} {c['score']:.2%}")

        # ------------------ CONFIDENCE GATE (STAGE 1 ONLY) -------------------
        if top_label in DEMOTION_THRESHOLDS and top_score < DEMOTION_THRESHOLDS[top_label]:
            candidates_filtered = [c for c in candidates if c["label"] != top_label]
            top_label = candidates_filtered[0]["label"]
            top_score = candidates_filtered[0]["score"]
            print(f"  Demoted eczema, new top: {top_label} ({top_score:.2%})")
            
        if top_score < CONFIDENCE_THRESHOLD:
            print(f"  >>> REJECTED: Low confidence ({top_score:.2%} < {CONFIDENCE_THRESHOLD:.0%})")
            return {
                "primary_prediction": "out-of-scope",
                "severity": None,
                "confidence": top_score,
                "candidates": candidates,
            }

        # ------------------ SECOND PHASE --------------------------------------
        severity = None
        if top_label in severity_classifiers:
            print(f"  Running Stage 2 for [{top_label}]...")
            lr_sev, le_sev = severity_classifiers[top_label]
            sev_proba = lr_sev.predict_proba(emb)[0]
            sev_candidates = sorted(
                [{"label": le_sev.classes_[i], "score": float(sev_proba[i])} for i in range(len(sev_proba))],
                key=lambda x: x["score"], reverse=True
            )
            print(f"  Stage 2 severity breakdown:")
            for s in sev_candidates:
                print(f"    {s['label']:<20} {s['score']:.2%}")
            severity = sev_candidates[0]["label"]
        else:
            print(f"  No severity model for [{top_label}]")

        # CONCATENATE LABELS WITH SEVERITY
        primary = f"{top_label}-{severity}" if severity else top_label
        print(f"  Final: {primary} | Confidence: {top_score:.2%}")

        return {
            "primary_prediction": primary,
            "severity": severity,
            "confidence": top_score,
            "candidates": candidates,
        }

    except ImagePreprocessingError as e:
        print(f"  Image error: {e}")
        raise HTTPException(status_code=400, detail=f"Image validation failed: {str(e)}")
    except Exception as e:
        print(f"  Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/check-quality")
async def check_quality(request: Request):
    image_data = await request.body()
    return check_image_quality(image_data)

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=7860, workers=1)