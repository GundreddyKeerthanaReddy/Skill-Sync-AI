from pathlib import Path
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.model_training import load_or_train, train_models
from src.data_preprocessing import load_dataset, split_skills
from src.recommendation_engine import recommend_careers, learning_roadmap
from src.visualization import career_bar_chart, skill_donut, confusion_heatmap

st.set_page_config(page_title="SkillSync AI", page_icon="🎯", layout="wide")
st.markdown("""<style>
/* SkillSync AI visual polish: explicit contrast works in both Streamlit themes. */
.block-container {padding-top: 1.5rem; padding-bottom: 2.5rem; max-width: 1450px;}
[data-testid="stMetric"] {
  background: #ffffff !important; padding: 18px 18px !important;
  border-radius: 14px !important; border: 1px solid #dbe3ef !important;
  box-shadow: 0 3px 12px rgba(15,23,42,.07);
}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {
  color: #475569 !important; font-weight: 600 !important;
}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div {
  color: #14213d !important; font-weight: 750 !important;
}
[data-testid="stMetricDelta"] {color: #475569 !important;}
.hero {padding: 28px 30px; border-radius: 16px; background: linear-gradient(120deg,#14213d,#315c9b); color:#ffffff; margin-bottom: 8px;}
.hero h1, .hero p {color:#ffffff !important;}
.stButton > button[kind="primary"] {background: #ef4f5f; border: 0; color: white; font-weight: 650; border-radius: 10px; min-height: 44px;}
.stButton > button[kind="primary"]:hover {background: #d93f50; border: 0; color: white;}
section[data-testid="stSidebar"] {border-right: 1px solid rgba(148,163,184,.18);}
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {border-radius: 9px;}
[data-testid="stDataFrame"] {border: 1px solid #e2e8f0; border-radius: 10px;}
</style>""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Training and evaluating career models...")
def get_bundle():
    return load_or_train()

bundle = get_bundle()
df = load_dataset()
all_skills = sorted({s for row in bundle["career_profiles"] for s in row["skills_text"].split()})
roles = sorted(df.career.unique())

with st.sidebar:
    st.title("🎯 SkillSync AI")
    st.caption("Intelligent skill-gap analysis & career guidance")
    page = st.radio("Navigate", ["Home Dashboard", "User Profile", "Skill Assessment", "Career Prediction", "Skill Gap Analysis", "Career Recommendations", "Skills to Learn", "Model Performance", "About Project"])
    st.divider()
    st.caption(f"{bundle['career_count']} career paths · {bundle['dataset_rows']} profile records")

if "skills" not in st.session_state: st.session_state.skills = []
if "recommendations" not in st.session_state: st.session_state.recommendations = []
if "prediction" not in st.session_state: st.session_state.prediction = None

def skill_input():
    selected = st.multiselect("Select your technical skills", all_skills, default=[s for s in st.session_state.skills if s in all_skills], help="Choose skills you can currently use.")
    custom = st.text_input("Add other skills (comma-separated)", placeholder="e.g. Java, Git, Power BI")
    combined = sorted(set(selected + split_skills(custom)))
    st.session_state.skills = combined
    st.caption(f"{len(combined)} skills selected")
    return combined

def run_analysis(skills, preference="Any"):
    if not skills:
        st.warning("Select at least one skill to run the analysis."); return
    with st.spinner("Analyzing your profile..."):
        recs, prediction = recommend_careers(skills, bundle, preference=preference, top_n=6)
    st.session_state.recommendations = recs
    st.session_state.prediction = prediction
    st.success("Your career analysis is ready.")

if page == "Home Dashboard":
    st.markdown('<div class="hero"><h1>Build your future with SkillSync AI</h1><p>Discover career paths that match your skills. Identify gaps and get a practical learning direction.</p></div>', unsafe_allow_html=True)
    st.write("")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Career paths", bundle["career_count"]); c2.metric("Dataset profiles", bundle["dataset_rows"])
    c3.metric("ML algorithms", 3); c4.metric("Your skills", len(st.session_state.skills))
    st.subheader("Start your assessment")
    skills = skill_input()
    preference = st.selectbox("Preferred career area", ["Any"] + roles)
    if st.button("Analyze my career fit", type="primary", use_container_width=True): run_analysis(skills, preference)
    if st.session_state.recommendations:
        st.subheader("Your current top matches")
        st.pyplot(career_bar_chart(st.session_state.recommendations))
    st.info("SkillSync combines supervised ML career classification with TF-IDF cosine similarity and explicit skill coverage.")
elif page == "User Profile":
    st.title("👤 User Profile")
    st.write("Create a local skill profile. No account or personal data is uploaded.")
    name = st.text_input("Display name", value=st.session_state.get("display_name", "Learner"))
    education = st.selectbox("Education level", ["Undergraduate", "Graduate", "Postgraduate", "Self-taught", "Other"])
    interest = st.selectbox("Primary interest", ["AI & Data", "Software Development", "Cloud & DevOps", "Cybersecurity", "Design", "Business & Analytics", "Exploring"])
    skills = skill_input()
    if st.button("Save profile"):
        st.session_state.display_name = name
        st.session_state.profile = {"education": education, "interest": interest}
        st.success("Profile saved for this session.")
    st.write({"Name": st.session_state.get("display_name", name), **st.session_state.get("profile", {})})
elif page == "Skill Assessment":
    st.title("🧩 Skill Assessment")
    st.write("Select the skills you currently possess. Add skills not listed in the dataset in the custom field.")
    skills = skill_input()
    st.metric("Skills in your profile", len(skills))
    if skills:
        st.write(", ".join(skills))
    if st.button("Submit assessment", type="primary"): run_analysis(skills)
elif page == "Career Prediction":
    st.title("🔮 Career Prediction")
    skills = skill_input()
    if st.button("Predict suitable careers", type="primary"): run_analysis(skills)
    if st.session_state.prediction:
        pred = st.session_state.prediction
        st.success(f"Predicted career: {pred['predicted_career']}")
        st.dataframe(pd.DataFrame(pred["top_predictions"]), use_container_width=True, hide_index=True)
elif page == "Skill Gap Analysis":
    st.title("📊 Skill Gap Analysis")
    skills = skill_input()
    target = st.selectbox("Career to analyze", roles)
    if st.button("Analyze skill gap", type="primary"):
        if not skills: st.warning("Please select at least one skill.")
        else:
            recs,_ = recommend_careers(skills,bundle,top_n=len(roles))
            item = next((x for x in recs if x["career"] == target), None)
            if item:
                st.metric("Career compatibility", f"{item['match_percentage']}%")
                a,b = st.columns(2); a.success("Matched skills: " + (", ".join(item["matched_skills"]) or "None")); b.warning("Missing skills: " + (", ".join(item["missing_skills"]) or "None"))
                st.pyplot(skill_donut(item["matched_skills"], item["missing_skills"]))
            else: st.error("No matching career profile was found.")
elif page == "Career Recommendations":
    st.title("🌟 Career Recommendations")
    skills = skill_input(); preference = st.selectbox("Career preference", ["Any"] + roles)
    if st.button("Generate recommendations", type="primary"): run_analysis(skills, preference)
    if st.session_state.recommendations:
        st.pyplot(career_bar_chart(st.session_state.recommendations))
        for i,item in enumerate(st.session_state.recommendations,1):
            with st.expander(f"{i}. {item['career']} — {item['recommendation_score']}% compatibility", expanded=(i==1)):
                st.write(f"Matched: {', '.join(item['matched_skills']) or 'None'}")
                st.write(f"Skills to develop: {', '.join(item['missing_skills']) or 'No major gaps in this profile'}")
                st.caption(f"Cosine similarity: {item['cosine_similarity']:.3f} · Classifier signal: {item['model_confidence']:.1f}%")
elif page == "Skills to Learn":
    st.title("📚 Personalized Skills to Learn")
    skills = skill_input()
    if st.button("Build my learning priorities", type="primary"): run_analysis(skills)
    if st.session_state.recommendations:
        for item in st.session_state.recommendations[:3]:
            st.subheader(item["career"])
            todo = learning_roadmap(item)
            if todo:
                for skill in todo: st.checkbox(skill, key=f"learn_{item['career']}_{skill}")
            else: st.success("Your listed skills cover this role profile well.")
elif page == "Model Performance":
    st.title("⚙️ Model Performance Dashboard")
    metrics = bundle["metrics"]
    rows = [{"Model": k, "Accuracy": v["accuracy"], "Precision": v["precision"], "Recall": v["recall"], "F1-score": v["f1"]} for k,v in metrics.items()]
    st.caption("Metrics are computed at runtime using a held-out test split of the loaded dataset.")
    st.dataframe(pd.DataFrame(rows).style.format({c:"{:.3f}" for c in ["Accuracy","Precision","Recall","F1-score"]}), use_container_width=True, hide_index=True)
    selected_model = st.selectbox("Confusion matrix", list(metrics))
    st.pyplot(confusion_heatmap(metrics, selected_model))
    st.write(f"Selected model: **{bundle['model_name']}** (highest weighted F1 on this split).")
    if st.button("Retrain models"):
        get_bundle.clear(); st.cache_resource.clear(); st.rerun()
elif page == "About Project":
    st.title("ℹ️ About SkillSync AI")
    st.write("SkillSync AI is an educational career guidance prototype that analyzes a learner's technical skills and compares them with career profiles.")
    st.subheader("Methodology")
    st.markdown("1. Clean and normalize career-skill records.\n2. Convert skill text into TF-IDF features.\n3. Train Logistic Regression, Random Forest and Linear SVM classifiers.\n4. Evaluate models using a held-out test set.\n5. Rank career profiles using TF-IDF cosine similarity and explicit skill coverage.\n6. Present missing skills and learning priorities.")
    st.warning("Career compatibility is a dataset-relative guidance score, not a hiring probability or guarantee of employability.")
    st.write("Stack: Python, Pandas, NumPy, Scikit-learn, Streamlit, Matplotlib, Seaborn, Jupyter Notebook, Git/GitHub.")
