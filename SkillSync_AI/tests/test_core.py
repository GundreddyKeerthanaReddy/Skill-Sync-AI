import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_preprocessing import load_dataset, split_skills
from src.model_training import train_models
from src.recommendation_engine import recommend_careers

def test_pipeline():
    df = load_dataset()
    assert len(df) >= 10 and df.career.nunique() >= 2
    assert split_skills("Python, SKLearn, SQL") == ["python", "scikit-learn", "sql"]
    bundle = train_models(save=False)
    recs, pred = recommend_careers(["python", "pandas", "numpy", "machine learning"], bundle)
    assert recs and pred["predicted_career"]
    assert 0 <= recs[0]["match_percentage"] <= 100
    assert isinstance(recs[0]["missing_skills"], list)
    print("CORE_PIPELINE_OK", len(df), bundle["model_name"], recs[0]["career"])

if __name__ == "__main__": test_pipeline()
