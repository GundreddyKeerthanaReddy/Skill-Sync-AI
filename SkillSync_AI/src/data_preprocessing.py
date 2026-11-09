"""Dataset loading, cleaning and skill text normalization."""
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "dataset.csv"

SKILL_ALIASES = {
    "scikit learn": "scikit-learn", "sklearn": "scikit-learn", "js": "javascript",
    "powerbi": "power bi", "nodejs": "node.js", "rest apis": "rest api",
    "ml": "machine learning", "ai": "artificial intelligence", "c++": "c++",
}

def normalize_text(value):
    text = str(value or "").lower().strip()
    text = re.sub(r"[^a-z0-9+#.\s-]", " ", text)
    return re.sub(r"\s+", " ", text)

def normalize_skill(value):
    skill = normalize_text(value)
    return SKILL_ALIASES.get(skill, skill)

def split_skills(value):
    if isinstance(value, (list, tuple, set)):
        raw = list(value)
    else:
        raw = re.split(r"[,;|/]", str(value or ""))
    return sorted({normalize_skill(x) for x in raw if normalize_skill(x)})

def _load_and_clean(path):
    df = pd.read_csv(path)
    lower = {str(c).strip().lower(): c for c in df.columns}
    role_col = next((lower[k] for k in ("career", "role", "career path", "suggested career path", "job title") if k in lower), None)
    skill_col = next((lower[k] for k in ("skills", "skill", "technical skills", "skills required") if k in lower), None)
    if role_col is None or skill_col is None:
        raise ValueError("Dataset must contain a career/role column and a skills column.")
    out = pd.DataFrame({"career": df[role_col], "skills": df[skill_col]})
    out["career"] = out["career"].fillna("").astype(str).str.strip()
    out["skills_list"] = out["skills"].fillna("").apply(split_skills)
    out["skills_text"] = out["skills_list"].apply(lambda x: " ".join(x))
    out = out[(out.career != "") & (out.skills_text != "")].drop_duplicates(subset=["career", "skills_text"])
    if len(out) < 6 or out.career.nunique() < 2:
        raise ValueError("Dataset needs at least six usable rows across two career classes.")
    return out.reset_index(drop=True)

def load_dataset(path=None):
    if path is not None:
        return _load_and_clean(path)
    public_path = ROOT / "data" / "career_path_public.csv"
    if public_path.exists():
        try:
            return _load_and_clean(public_path)
        except Exception:
            # A stale, partial or schema-incompatible external download must not
            # prevent the bundled offline fallback from launching.
            pass
    return _load_and_clean(DATA_PATH)
