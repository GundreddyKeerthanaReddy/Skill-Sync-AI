# SkillSync AI
## Intelligent ML-Based Skill Gap Analysis & Career Recommendation System

### 1. Project Overview

SkillSync AI is an intelligent, machine learning-based career guidance and skill gap analysis platform designed to help students, graduates, and job seekers identify suitable career opportunities based on their existing technical skills.

In today's competitive job market, students often struggle to identify which career path aligns with their current skill sets. They may possess programming, database, data science, or web development knowledge but lack clarity about which professional roles best match their capabilities. Additionally, they may not know which additional skills they need to acquire to become suitable candidates for their desired roles.

SkillSync AI addresses these challenges by combining supervised machine learning, Natural Language Processing (NLP), TF-IDF vectorization, and cosine similarity techniques.

The system analyzes a learner's technical skills, predicts suitable career categories, ranks potential job roles according to skill compatibility, identifies missing skills, and generates personalized learning priorities.

The application is developed using Python, Scikit-learn, Pandas, NumPy, and Streamlit. It follows a local-first architecture, allowing users to execute the entire application on their own systems without depending on paid APIs, cloud databases, or external application servers.

### 2. Problem Statement

Students and job seekers frequently face difficulties in selecting appropriate career paths because of the following challenges:

- Lack of awareness regarding career opportunities associated with their technical skills.
- Difficulty identifying the difference between their current skills and industry role requirements.
- Limited access to personalized career guidance.
- Uncertainty regarding which skills to prioritize when preparing for specific job roles.
- Difficulty evaluating their technical readiness for different career domains.

Traditional career guidance approaches often depend on generic recommendations and manual assessments. These approaches may not provide detailed, personalized skill comparisons.

Therefore, there is a need for an intelligent system that can analyze a user's skill profile, predict suitable career categories, calculate role compatibility, and provide actionable recommendations for skill development.

### 3. Project Objectives

The primary objective of SkillSync AI is to develop an ML-powered career recommendation platform that provides personalized career guidance through automated skill analysis.

The specific objectives are:

1. Develop a structured career-skill dataset containing different career categories and their associated technical skills.
2. Apply data preprocessing techniques to clean, normalize, and validate the dataset.
3. Extract meaningful textual features using TF-IDF vectorization.
4. Train multiple supervised machine learning algorithms to classify career categories.
5. Compare Logistic Regression, Random Forest, and Linear Support Vector Machine models.
6. Automatically identify the best-performing classifier using evaluation metrics.
7. Implement cosine similarity to identify career roles that closely match a user's skill profile.
8. Calculate skill compatibility scores using skill coverage and similarity.
9. Identify matched and missing skills for selected career roles.
10. Generate personalized learning priorities to help users improve their technical profiles.
11. Develop an interactive Streamlit dashboard for visualizing career recommendations and model performance.

### 4. System Features

#### 4.1 User Profile and Skill Selection

The application allows users to create a technical skill profile by selecting their existing skills from a predefined collection.

Users can select skills belonging to different domains, including:

- Programming: Python, Java, C++, JavaScript.
- Data Science: Pandas, NumPy, Matplotlib, Seaborn.
- Machine Learning: Scikit-learn, TensorFlow, PyTorch.
- Databases: SQL, MySQL, MongoDB.
- Web Development: HTML, CSS, React, Next.js.
- Cloud and DevOps: AWS, Azure, Git, GitHub.
- Data Analytics: Excel, Power BI, Tableau.

The system also supports custom skills, allowing users to enter technical abilities that are not available in the predefined skill collection.

The selected skills are normalized and converted into a structured representation for further processing.

#### 4.2 Career Classification

Career classification is one of the core machine learning functionalities of SkillSync AI.

The system uses supervised learning algorithms to predict the career category associated with a user's technical skill profile.

Three classification algorithms are implemented:

**A. Logistic Regression**

Logistic Regression is a supervised classification algorithm used to predict the probability of a sample belonging to a particular class.

In SkillSync AI, it analyzes the input skill features and estimates the likelihood of different career categories.

It is useful as a baseline classification algorithm because of its simplicity, computational efficiency, and interpretability.

**B. Random Forest Classifier**

Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve classification performance.

Each decision tree generates a prediction, and the final output is determined through majority voting.

In this project, Random Forest identifies relationships between different skill combinations and career categories.

Its ensemble structure helps capture nonlinear patterns in the training data.

**C. Linear Support Vector Machine (Linear SVM)**

Linear SVM is a supervised learning algorithm that identifies an optimal decision boundary between different classes.

It attempts to maximize the margin separating career categories in the feature space.

Since TF-IDF produces high-dimensional sparse feature representations, Linear SVM is particularly suitable for this text-based classification task.

The system trains and evaluates all three algorithms and compares their performance to select the best-performing model.

#### 4.3 Automatic Model Comparison

SkillSync AI evaluates the three classification algorithms using multiple performance metrics.

The evaluation process includes:

- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1-score

The model comparison allows users to understand how effectively different algorithms classify career categories.

The best-performing model is selected based on the configured evaluation criterion and saved as a reusable model bundle.

#### 4.4 TF-IDF Feature Extraction

Term Frequency-Inverse Document Frequency (TF-IDF) is a text feature extraction technique used to convert textual information into numerical representations.

In SkillSync AI, technical skills are represented as text documents.

For example:

Python Pandas NumPy Machine Learning SQL

TF-IDF converts these skill descriptions into numerical vectors.

The technique considers two important factors:

- Term Frequency (TF): Measures how frequently a particular skill appears in a document.
- Inverse Document Frequency (IDF): Measures how informative a term is across the collection of documents.

The TF-IDF representation helps the system identify important skills and compare different career profiles mathematically.

#### 4.5 Career Recommendation Engine

The recommendation engine identifies suitable career roles based on the learner's selected skills.

Instead of depending entirely on the classifier's predicted category, the system combines two complementary approaches:

1. Machine learning-based career prediction.
2. Similarity-based skill matching.

The recommendation engine ranks career profiles according to their relevance to the user's existing skills.

This hybrid approach provides a more detailed career recommendation mechanism than relying on a single classification output.

#### 4.6 Cosine Similarity

Cosine similarity measures the similarity between two numerical vectors by calculating the cosine of the angle between them.

It is used to compare the user's TF-IDF skill representation with the TF-IDF representation of available career profiles.

The formula is:

Cosine Similarity = (A · B) / (||A|| × ||B||)

Where:

- A represents the user's skill vector.
- B represents the career profile vector.
- A · B represents their dot product.
- ||A|| and ||B|| represent their vector magnitudes.

A higher cosine similarity indicates greater textual similarity between the two profiles.

#### 4.7 Skill Coverage and Compatibility Score

Apart from cosine similarity, the system calculates how many required skills of a particular role are already possessed by the learner.

Skill Coverage is calculated as:

Skill Coverage (%) = (Number of Matched Skills / Total Required Skills) × 100

For example:

Consider a Data Scientist role requiring:

Python, SQL, Pandas, NumPy, Machine Learning, Statistics.

Suppose a learner possesses:

Python, SQL, Pandas, NumPy.

The learner has four matched skills out of six required skills.

Skill Coverage = (4 / 6) × 100 = 66.67%

The system combines role similarity and skill coverage to produce a compatibility score.

The final compatibility score is a relative ranking measure based on the configured recommendation-engine weighting. It should not be interpreted as a hiring probability.

#### 4.8 Skill Gap Analysis

Skill Gap Analysis identifies the difference between the skills currently possessed by a learner and the skills required for a selected career role.

The analysis divides skills into two categories:

**Matched Skills:** Technical skills that the learner already possesses and that are also required for the selected role.

**Missing Skills:** Technical skills required by the selected role that are not present in the learner's current profile.

For example:

Target Role: Data Scientist

Existing Skills:
Python, SQL, Pandas, NumPy.

Required Skills:
Python, SQL, Pandas, NumPy, Machine Learning, Statistics, Scikit-learn.

Matched Skills:
Python, SQL, Pandas, NumPy.

Missing Skills:
Machine Learning, Statistics, Scikit-learn.

This functionality helps users understand their current technical strengths and identify areas requiring improvement.

#### 4.9 Personalized Learning Priorities

After identifying missing skills, SkillSync AI generates a list of learning priorities.

These recommendations help users focus on the technical competencies required for their selected career path.

For example, a learner interested in becoming a Machine Learning Engineer may receive the following priorities:

1. Machine Learning fundamentals.
2. Statistics and probability.
3. Scikit-learn.
4. Model evaluation techniques.
5. Deep Learning fundamentals.

The recommendations are derived from the missing skills in the selected career profile. They represent suggested learning priorities rather than a guaranteed training roadmap.

### 5. Technology Stack

The project uses the following technologies:

**Programming Language**
- Python: Main programming language used to implement data preprocessing, machine learning, recommendation logic, and application functionality.

**Data Processing**
- Pandas: Dataset loading, cleaning, manipulation, and analysis.
- NumPy: Numerical operations and array-based computations.

**Machine Learning**
- Scikit-learn: Implementation of classification algorithms, TF-IDF vectorization, cosine similarity, evaluation metrics, and model persistence.

**Data Visualization**
- Matplotlib: Generating charts and graphical representations.
- Seaborn: Statistical visualization and confusion matrix heatmaps.

**Application Development**
- Streamlit: Developing the interactive web-based dashboard.

**Development Environment**
- Jupyter Notebook: Exploratory data analysis and experimentation.
- Visual Studio Code: Project development and debugging.
- Git and GitHub: Version control and source code management.

### 6. Dataset Description

SkillSync AI supports two dataset sources.

**A. Bundled Fallback Dataset**

The project includes a compact, curated dataset stored at:

data/dataset.csv

It contains predefined career categories and their associated technical skills.

This dataset enables the application to run without requiring an internet connection or an external dataset download.

It is primarily intended for demonstrating the complete functionality of the system.

Since the dataset is small and curated, its classification performance should not be treated as evidence of real-world career prediction accuracy.

**B. Optional Public Dataset**

The application also supports the public Career Path Prediction dataset.

Dataset name: Career Path Prediction  
Publisher: Mendeley Data  
Records: Approximately 15,000 entries  
DOI: 10.17632/cns7h4b9g7.1  
License: CC BY 4.0

Dataset URL:
https://data.mendeley.com/datasets/cns7h4b9g7/1

Users can attempt to download the dataset using:

python -m src.dataset_download

The application checks whether the public dataset exists and whether its columns can be mapped to the expected schema.

If the public dataset is available and compatible, the application uses it. Otherwise, it automatically uses the bundled fallback dataset.

### 7. System Architecture

The application follows a modular architecture in which each Python module is responsible for a specific functionality.

**Module 1: data_preprocessing.py**

Responsible for:
- Loading career-skill datasets.
- Validating dataset structure.
- Handling missing values.
- Cleaning and normalizing skill names.
- Removing duplicate records.
- Preparing structured data for model training.

**Module 2: feature_extraction.py**

Responsible for:
- Converting skill descriptions into numerical representations.
- Implementing TF-IDF vectorization.
- Generating feature matrices.
- Preparing input features for classification and similarity analysis.

**Module 3: model_training.py**

Responsible for:
- Splitting data into training and testing sets.
- Training Logistic Regression.
- Training Random Forest.
- Training Linear SVM.
- Evaluating model performance.
- Selecting the best-performing classifier.
- Saving the trained model bundle.

**Module 4: career_prediction.py**

Responsible for:
- Receiving user skill information.
- Transforming input using the trained vectorizer.
- Predicting the appropriate career category.
- Generating classification-related outputs.

**Module 5: skill_gap_analysis.py**

Responsible for:
- Comparing user skills with career requirements.
- Calculating cosine similarity.
- Identifying matched skills.
- Identifying missing skills.
- Calculating skill coverage.

**Module 6: recommendation_engine.py**

Responsible for:
- Combining career similarity and model prediction signals.
- Ranking potential career roles.
- Calculating compatibility scores.
- Generating personalized skill recommendations.

**Module 7: visualization.py**

Responsible for:
- Creating career ranking charts.
- Visualizing skill coverage.
- Generating confusion matrices.
- Presenting model comparison results.

**Module 8: app.py**

Responsible for:
- Launching the Streamlit application.
- Collecting user information and technical skills.
- Displaying career recommendations.
- Presenting skill gap analysis.
- Displaying learning priorities.
- Showing model performance and visualizations.

### 8. Working Methodology

The complete workflow of SkillSync AI consists of the following stages:

**Stage 1: Data Collection**

The system loads the bundled career-skill dataset or the optional public dataset.

**Stage 2: Data Preprocessing**

The dataset is validated, cleaned, normalized, and prepared for feature extraction.

**Stage 3: Feature Extraction**

TF-IDF converts textual skill descriptions into numerical feature vectors.

**Stage 4: Model Training**

The prepared dataset is used to train Logistic Regression, Random Forest, and Linear SVM classifiers.

**Stage 5: Model Evaluation**

The three algorithms are evaluated using accuracy, precision, recall, and F1-score.

**Stage 6: Best Model Selection**

The system compares the evaluation results and selects the best-performing classifier.

**Stage 7: User Skill Input**

The learner selects existing technical skills and optionally adds custom skills.

**Stage 8: Career Prediction**

The trained classifier predicts the most suitable career category based on the input profile.

**Stage 9: Similarity-Based Role Ranking**

The recommendation engine compares the learner's skills with available career profiles using TF-IDF and cosine similarity.

**Stage 10: Skill Gap Identification**

The system identifies matched and missing skills for recommended or selected roles.

**Stage 11: Learning Recommendations**

The application generates learning priorities based on the identified skill gaps.

**Stage 12: Dashboard Visualization**

The final results are presented through interactive Streamlit components and charts.

### 9. Model Evaluation Metrics

The system evaluates the classification algorithms using four major metrics.

**Accuracy**

Measures the proportion of correctly classified predictions.

Accuracy = Correct Predictions / Total Predictions

**Precision**

Measures how many predictions made for a particular class were actually correct.

Precision = TP / (TP + FP)

**Recall**

Measures how many actual instances of a class were correctly identified.

Recall = TP / (TP + FN)

**F1-Score**

Represents the harmonic mean of precision and recall.

F1 = 2 × (Precision × Recall) / (Precision + Recall)

Since the project supports multiple career categories, weighted precision, recall, and F1-score are used to account for differences in class frequencies.

**Confusion Matrix**

A confusion matrix provides a visual representation of actual versus predicted career categories.

It helps identify:
- Correct classifications.
- Incorrect classifications.
- Career categories that are frequently confused with one another.

### 10. Streamlit Application Interface

The project provides an interactive dashboard with multiple sections.

**Home Dashboard**

Displays an overview of the application and allows learners to select their technical skills and initiate career analysis.

**Career Prediction**

Displays the career category predicted by the selected machine learning classifier.

**Career Recommendations**

Displays ranked career roles and their compatibility scores.

**Skill Gap Analysis**

Allows users to select a target role and examine matched and missing technical skills.

**Skills to Learn**

Displays personalized learning priorities based on the selected career path.

**Model Performance**

Displays the comparative performance of Logistic Regression, Random Forest, and Linear SVM using evaluation metrics and confusion matrices.

### 11. Installation and Execution

**System Requirements**

- Windows 10 or Windows 11.
- Python 3.10 or Python 3.11 (64-bit).
- Visual Studio Code.
- Python extension for VS Code.
- Internet connection only for optional dataset download and package installation.

**Step 1: Extract the Project**

Extract the SkillSync AI ZIP file and open the project folder in VS Code.

**Step 2: Create a Virtual Environment**

```powershell
py -3.11 -m venv .venv
```

**Step 3: Activate the Environment**

```powershell
.venv\Scripts\Activate.ps1
```

If activation is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

**Step 4: Install Dependencies**

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Step 5: Optional Dataset Download**

```powershell
python -m src.dataset_download
```

**Step 6: Train and Evaluate Models**

```powershell
python -c "from src.model_training import train_models; b=train_models(); print('Selected:', b['model_name']); print(b['metrics'])"
```

**Step 7: Launch the Application**

```powershell
streamlit run app.py
```

The application will normally be accessible at:

http://localhost:8501

### 12. Demonstration Workflow

The following workflow can be followed while demonstrating the application:

1. Open the SkillSync AI Home Dashboard.
2. Select technical skills such as Python, Pandas, NumPy, SQL, and Machine Learning.
3. Add any additional skills using the custom skill option.
4. Click the Analyze My Career Fit button.
5. Display the ranked career recommendations.
6. Explain the compatibility scores and predicted career category.
7. Navigate to Career Prediction and explain the classifier output.
8. Open Skill Gap Analysis and select a target career role.
9. Explain the matched and missing skills.
10. Navigate to Skills to Learn and explain the personalized learning priorities.
11. Open Model Performance and compare the three machine learning algorithms.
12. Display the confusion matrix and explain the classification results.

### 13. Jupyter Notebook and Exploratory Data Analysis

The project includes a Jupyter Notebook:

notebooks/SkillSync_AI_EDA.ipynb

The notebook documents the exploratory analysis and experimentation performed during development.

It includes:
- Dataset inspection.
- Dataset structure and statistical summaries.
- Missing-value analysis.
- Data cleaning.
- Career category distribution.
- Technical skill frequency analysis.
- Feature exploration.
- Model evaluation.
- Career profile comparisons.

The notebook can be executed through VS Code with the Jupyter extension or uploaded to Google Colab.

### 14. Limitations and Considerations

Although SkillSync AI provides an integrated career recommendation workflow, certain limitations must be considered.

1. **Dataset Quality:** The bundled fallback dataset is small and curated. Model performance obtained from it is primarily intended for functional demonstration.

2. **External Dataset Dependency:** The optional public dataset depends on an external download endpoint, which may change or become unavailable.

3. **Career Recommendation Scope:** Recommendations are limited to the career categories and skill profiles represented in the loaded dataset.

4. **Compatibility Score Interpretation:** The compatibility score represents relative similarity and skill coverage, not the probability of securing employment.

5. **Custom Skills:** Custom skills are accepted as user input, but the accuracy of recommendations depends on whether those skills can be meaningfully matched with the available career profiles.

6. **No Real-Time Job Market Integration:** The system does not retrieve live job openings, salary information, or real-time industry demand.

7. **Educational Purpose:** The platform is designed for educational career guidance and skill development. It should not be used as an automated hiring or recruitment decision system.

### 15. Future Enhancements

The following enhancements could improve the capabilities of SkillSync AI:

- Integration of larger and more diverse career datasets.
- Incorporation of real-time job market information.
- Addition of salary and career growth insights.
- Personalized learning roadmaps with estimated learning durations.
- Integration of online learning resources and certifications.
- Resume parsing and automatic skill extraction using NLP.
- Support for experience-based and education-based recommendations.
- Advanced semantic matching using transformer-based embeddings.
- Interactive career progression analysis.
- Personalized progress tracking and skill assessment.

### 16. Conclusion

SkillSync AI demonstrates how machine learning and text similarity techniques can be combined to develop an intelligent career guidance platform.

By integrating supervised classification algorithms, TF-IDF feature extraction, cosine similarity, and skill coverage analysis, the system provides a structured approach to identifying suitable career opportunities and understanding individual skill gaps.

The application goes beyond simple career prediction by explaining which skills are already available in a learner's profile, which competencies are missing, and what areas should be prioritized for further learning.

Its modular architecture, local-first implementation, and interactive Streamlit dashboard make it a practical educational prototype for demonstrating machine learning applications in career guidance.

Ultimately, SkillSync AI aims to help learners make more informed career decisions by connecting their existing technical capabilities with potential professional opportunities.