# Credit Card Fraud Detection (Capstone)

## 1. Project Title and Description
This project builds an end-to-end machine learning pipeline for **credit card fraud detection** using the Kaggle ULB dataset.

The workflow includes:
- data cleaning and preprocessing
- exploratory data analysis (EDA)
- feature engineering and feature selection
- model training and comparison
- hyperparameter tuning
- model explainability with SHAP

Main objective: produce a model that identifies fraudulent transactions while balancing fraud catch rate and false alerts.

## 2. Problem Statement
Financial fraud detection is a highly imbalanced binary classification problem where fraudulent transactions are rare.

Given historical labeled transactions, predict whether each transaction is:
- `0` = legitimate
- `1` = fraud

Because fraud is rare, the project prioritizes **PR-AUC, Precision, Recall, and F1** (not accuracy alone).

## 3. Dataset Source and Description
- **Dataset**: Credit Card Fraud Detection
- **Source**: Kaggle (ULB ML Group)
- **Link**: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

### Dataset overview
- Original size: 284,807 transactions
- Target: `Class` (`0` legit, `1` fraud)
- Fraud ratio: ~0.17%
- Features:
  - `Time`, `Amount`
  - `V1` to `V28` (PCA-transformed anonymized features)

> Note: The dataset is highly imbalanced and from a limited historical period, so careful validation and threshold tuning are essential.

## 4. Installation Instructions
### Prerequisites
- Python 3.10+
- `pip`

### Setup
From the project root:
```bash
cd Capstone
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate     # Windows

pip install --upgrade pip
pip install numpy pandas matplotlib seaborn scikit-learn xgboost category-encoders shap umap-learn lime jupyter
```

## 5. How to Run the Code
### Option A: Jupyter Notebook (recommended)
```bash
cd Capstone
jupyter notebook capstone.ipynb
```
Run cells in order from top to bottom.

### Option B: Dataset download helper
If you use the helper script:
```bash
python download_dataset.py
```
Ensure your Kaggle credentials/config are set up if required by the script.

## 6. Results Summary
### Modeling highlights
- Baselines tested: Logistic Regression, Decision Tree
- Ensembles tested: Random Forest, XGBoost
- Best untuned performance came from **XGBoost** and **Random Forest**
- Hyperparameter tuning was applied to top models

### Best observed tuned model
From the notebook leaderboard, **XGBoost (tuned)** achieved the strongest overall fraud-class performance, including:
- ROC-AUC: **0.9847**
- PR-AUC: **0.9012**
- F1: **0.8889**
- Precision: **0.8936**
- Recall: **0.8842**

### Explainability
Section 5a adds SHAP-based explainability to answer:
> “Why did the model make this prediction?”

It includes:
- global feature impact (beeswarm)
- local single-prediction explanation (waterfall + top positive/negative contributors)

## 7. Author Information
- **Author**: Paolo Cabrera
- **Program/Track**: AIM - AI/ML Capstone

---
If you use this project for deployment, add leakage-safe preprocessing pipelines, temporal validation, and drift monitoring before production rollout.
