## ❤️ heart-disease-prediction-ml

An end-to-end machine learning system that predicts heart disease risk from patient clinical data (age, cholesterol, ECG, and more). Covers data cleaning, EDA, and comparison of 6 classification models, followed by hyperparameter tuning, cross-validation, and deployment as an interactive Streamlit app with risk scoring and health recommendations.
Final intership Project — Phase 1 (Data Preprocessing & EDA) + Phase 2 (Optimization, Deployment & Clinical Insights)
An end-to-end ML system that predicts heart disease risk from patient clinical records, with an interactive Streamlit prediction tool.

## 📁 Project Structure
```
heart_disease_project/
├── data/
│   ├── heart.csv                     # Original raw dataset (1,025 rows)
│   └── heart_cleaned.csv             # Preprocessed & encoded dataset (302 rows)
├── scripts/
│   ├── phase1_preprocessing_eda.ipynb        # Data cleaning, outlier handling, EDA
│   ├── phase1_model_training.ipynb           # Baseline model training & comparison
│   ├── Phase 2 - hyperparameter_tuning.ipynb # GridSearchCV + cross-validation + best model
│   └── Phase2 Deployment.ipynb               # Deployment prep / pipeline export
├── app/
│   └── app.py                        # Streamlit prediction app
├── output/
│   ├── heart_models/
│   │   ├── heart_disease_best_pipeline.pkl   # Deployable pipeline (preprocessing + model)
│   │   ├── best_model_info.json
│   │   ├── Logistic_Regression.pkl
│   │   ├── Decision_Tree.pkl
│   │   ├── Random_Forest.pkl
│   │   ├── SVM_RBF.pkl
│   │   ├── KNN.pkl
│   │   ├── Naive_Bayes.pkl
│   │   └── model_comparison.csv
│   ├── heart_plots/                  # All EDA & model evaluation charts
│   ├── model_comparison.csv          # Phase 1 baseline metrics
│   ├── tuned_model_comparison.csv    # Phase 2 tuned metrics
│   └── best_model_feature_importance.csv
└── README.md

```
## ⚙️ Phase 1 Summary

•	Removed 723 exact duplicate rows (1,025 → 302 unique records); no missing values

•	Outliers in trestbps, chol, thalach, oldpeak capped via IQR method

•	One-hot encoded nominal features (cp, restecg, slope, thal); scaled continuous features

•	Compared 6 baseline models — Logistic Regression led at 85.25% accuracy / 0.898 ROC-AUC

## 🚀 Phase 2 Summary
•	Built a single deployable sklearn.Pipeline (ColumnTransformer + classifier)

•	Tuned 4 candidate models with GridSearchCV + 5-fold Stratified Cross-Validation

•	Final model: Logistic Regression — 85.25% accuracy, 0.894 ROC-AUC. Selected over SVM (RBF, 0.909 ROC-AUC) because it wins on Accuracy/Precision/F1 and is fully interpretable, which the clinical-insights requirement demands.

•	Deployed as a Streamlit app with prediction probability, Low/Medium/High risk category, and tailored health recommendations

•	Clinical insights generated from Logistic Regression coefficients (see Final Report, Section 6)

## 📈Tuned Model Comparison
```
| Model | Test Accuracy	| Test ROC-AUC|

| SVM (RBF)	| 83.61% | 0.9091|
|Naive Bayes	| 81.97%	| 0.8983 |
| KNN	| 83.61% | 0.8961 |
| Logistic Regression (deployed) | ✅	85.25% |	0.8939 |
| Random Forest	| 78.69%	| 0.8799 |
| Decision Tree	| 75.41%	| 0.8274 |

```
## ▶️ How to Run

1️⃣ Clone the repo
```
git clone: https://github.com/ManojgowdaBY/heart-disease-prediction-ml
cd heart-disease-prediction-ml
```
2️⃣ Install dependencies
```
pip install pandas numpy scikit-learn matplotlib seaborn joblib streamlit

```

3️⃣ Run the notebooks (Phase 1 & 2 — preprocessing, EDA, training, tuning)
```
Open Jupyter Notebook and run in order:
jupyter notebook
Then open and run, top to bottom:
1.scripts/phase1_preprocessing_eda.ipynb 2.scripts/phase1_model_training.ipynb 3.scripts/Phase 2 - hyperparameter_tuning.ipynb
```
4️⃣ Launch the prediction app
⚠️The Streamlit app must be run from a terminal (Anaconda Prompt / Command Prompt) — not from inside a Jupyter cell.
```
cd app
streamlit run app.py
```
This opens the app automatically in your browser at http://localhost:8501. Enter patient details (age, sex, chest pain type, blood pressure, cholesterol, etc.) to get:

1.Predicted probability of heart disease 
2.Risk category (Low / Medium / High) 
3.Tailored health recommendations

## 🩺Clinical Insights

Based on the Logistic Regression coefficients, the strongest predictors of heart disease risk in this dataset are:

💔Chest pain type (```cp```) — atypical/asymptomatic pain types raise predicted risk the most

🧬Thalassemia (```thal```) and sex — both shift predicted risk substantially

❤️‍🔥Max heart rate (```thalach```) — higher achieved heart rate associates with higher predicted risk

## ✅ Deliverables Checklist

•	 💻Source Code (scripts/, app/)

•	 📓jupyter Notebook

•	 🔗GitHub Repository Link — https://github.com/ManojgowdaBY/heart-disease-prediction-ml

•	 Final Report (outputs/Heart_Disease_Prediction_Final_Report.docx)

## ⚠️Disclaimer
This project is for educational purposes. Predictions are statistical associations learned from a small (302-record) public dataset and must not be used as an actual medical diagnosis.

## 👤 Author
ManojGowda B Y 🎓B.E. Computer Science Engineering (AI/ML) — DBIT Bengaluru 🔗 GitHub: @ManojgowdaBY

