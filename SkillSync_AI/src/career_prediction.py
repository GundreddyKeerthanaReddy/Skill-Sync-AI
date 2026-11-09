"""Classifier-based career prediction."""
from .data_preprocessing import split_skills

def predict_careers(user_skills, bundle, top_n=5):
    skills = split_skills(user_skills)
    if not skills:
        raise ValueError("Please provide at least one technical skill.")
    model = bundle["model"]
    text = " ".join(skills)
    pred = model.predict([text])[0]
    ranked = []
    clf = model.named_steps["classifier"]
    if hasattr(clf, "predict_proba"):
        probs = model.predict_proba([text])[0]
        ranked = sorted(zip(model.classes_, probs), key=lambda x: x[1], reverse=True)
    elif hasattr(clf, "decision_function"):
        scores = model.decision_function([text])
        scores = scores[0] if getattr(scores, "ndim", 1) > 1 else scores
        ranked = sorted(zip(model.classes_, scores), key=lambda x: x[1], reverse=True)
        # Convert decision scores to a display-only relative confidence distribution.
        import numpy as np
        vals = np.array([v for _, v in ranked], dtype=float)
        exp = np.exp(vals - vals.max())
        ranked = [(r, float(p)) for (r, _), p in zip(ranked, exp / exp.sum())]
    else:
        ranked = [(pred, 1.0)]
    return {"predicted_career": str(pred), "top_predictions": [{"career": str(r), "confidence": round(float(p) * 100, 1)} for r, p in ranked[:top_n]]}
