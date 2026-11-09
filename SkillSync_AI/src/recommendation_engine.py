"""Combine classifier ranking with skill-gap fit."""
from .skill_gap_analysis import analyze_skill_gap
from .career_prediction import predict_careers

def recommend_careers(user_skills, bundle, preference="Any", top_n=5):
    gaps = analyze_skill_gap(user_skills, bundle["career_profiles"])
    pred = predict_careers(user_skills, bundle, top_n=max(top_n, 5))
    confidence = {x["career"]: x["confidence"] for x in pred["top_predictions"]}
    results = []
    for item in gaps:
        if preference != "Any" and preference.lower() not in item["career"].lower():
            continue
        # Blend measured profile similarity with classifier confidence where available.
        model_signal = confidence.get(item["career"], 0.0)
        final = 0.8 * item["match_percentage"] + 0.2 * model_signal
        results.append({**item, "model_confidence": model_signal, "recommendation_score": round(final, 1)})
    return sorted(results, key=lambda x: x["recommendation_score"], reverse=True)[:top_n], pred

def learning_roadmap(recommendation, limit=8):
    return recommendation.get("missing_skills", [])[:limit]
