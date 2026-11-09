"""Compare user skills with role profiles using TF-IDF cosine similarity."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .data_preprocessing import split_skills

def analyze_skill_gap(user_skills, career_profiles):
    skills = split_skills(user_skills)
    if not skills:
        raise ValueError("Please enter or select at least one skill.")
    # Aggregate role skill evidence from the available dataset.
    role_map = {}
    for row in career_profiles:
        role_map.setdefault(row["career"], set()).update(split_skills(row["skills_text"]))
    roles = []
    for role, required in role_map.items():
        required = sorted(required)
        docs = [" ".join(skills), " ".join(required)]
        vec = TfidfVectorizer(token_pattern=r"(?u)\b[\w+#.-]+\b")
        matrix = vec.fit_transform(docs)
        similarity = float(cosine_similarity(matrix[0:1], matrix[1:2])[0, 0])
        matched = sorted(set(skills) & set(required))
        missing = sorted(set(required) - set(skills))
        coverage = len(matched) / max(len(required), 1)
        # Combine lexical cosine similarity and explicit requirement coverage.
        score = 100 * (0.65 * similarity + 0.35 * coverage)
        roles.append({"career": role, "match_percentage": round(min(score, 100), 1),
                      "cosine_similarity": round(similarity, 4), "matched_skills": matched,
                      "missing_skills": missing, "required_skills": required})
    return sorted(roles, key=lambda x: x["match_percentage"], reverse=True)
