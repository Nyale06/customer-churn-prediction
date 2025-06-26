# 📊 Customer Churn Prediction in the Telecom Sector

This project is a personal Data Science initiative aimed at predicting customer churn in the telecom industry using supervised machine learning techniques. It includes a complete ML pipeline from data preprocessing to model evaluation and an interactive web app built with Streamlit.

---

## 🚀 Project Overview

Customer churn is a critical metric for telecom companies — this project helps identify customers likely to leave so the company can take proactive measures.

### 🔍 Key Features
- Exploratory Data Analysis (EDA)
- Handling class imbalance using:
  - `class_weight` (Random Forest, SVM)
  - `SMOTE` (optional)
- Model training & tuning using:
  - Random Forest
  - Support Vector Machine (SVM)
  - XGBoost
- Hyperparameter tuning with `GridSearchCV`
- Model evaluation using:
  - F1-score
  - Accuracy
  - ROC-AUC
- Interactive dashboard with Streamlit

---

## 🧰 Tech Stack

- **Languages**: Python
- **Libraries**: 
  - `pandas`, `numpy`, `matplotlib`, `seaborn`
  - `scikit-learn`, `xgboost`, `imblearn`
  - `streamlit` (for the dashboard)
- **Tools**: Jupyter Notebook, VS Code, Git, GitHub

---

## 📂 Project Structure

```bash
customer_churn_prediction/
├── data/                   # (Optional) Raw & processed datasets
├── models/                 # Saved models (e.g., .pkl or .joblib)
├── notebooks/              # Jupyter notebooks for EDA & experiments
├── app/
│   └── streamlit_app.py    # Streamlit dashboard
├── churn_pipeline.py       # Main training & evaluation script
├── requirements.txt
├── README.md
└── .gitignore
