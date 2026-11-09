"""Matplotlib/Seaborn visualizations used by Streamlit."""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def career_bar_chart(recommendations):
    df = pd.DataFrame(recommendations)
    fig, ax = plt.subplots(figsize=(8, 4))
    if not df.empty:
        sns.barplot(data=df, x="recommendation_score", y="career", ax=ax)
        ax.set_xlabel("Recommendation score (%)"); ax.set_ylabel(""); ax.set_xlim(0, 100)
    fig.tight_layout(); return fig

def skill_donut(matched, missing):
    fig, ax = plt.subplots(figsize=(4, 4))
    values = [len(matched), len(missing)]
    if sum(values) == 0: values = [1, 0]
    ax.pie(values, labels=["Matched", "To learn"], autopct="%1.0f%%", startangle=90,
           wedgeprops={"width": 0.42})
    ax.set_title("Skill coverage"); fig.tight_layout(); return fig

def confusion_heatmap(metrics, model_name):
    item = metrics[model_name]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(item["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                xticklabels=item["labels"], yticklabels=item["labels"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(model_name)
    fig.tight_layout(); return fig
