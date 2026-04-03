# Credit Card Fraud Detection (Capstone)

## 1. Project Title and Description
This project builds an end-to-end machine learning pipeline for **credit card fraud detection** using the Kaggle ULB dataset.

The workflow includes:
- data cleaning and preprocessing
- exploratory data analysis (EDA)
- feature engineering and feature selection
- model training and comparison
- hyperparameter tuning
- model explainability with SHAP and LIME
- model limitations analysis (leakage, imbalance, generalization)

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

## 4. Project Files
| File | Description |
|------|-------------|
| `Paolo_Cabrera_Capstone.ipynb` | Main notebook — run this end-to-end |
| `requirements.txt` | All Python dependencies with pinned versions |
| `data/creditcard.csv` | Dataset (downloaded separately — see Section 5) |

## 5. How to Run the Code

### Step 1 — Set up a virtual environment
```bash
cd Capstone
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate     # Windows
```

### Step 2 — Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3 — Download the Kaggle dataset

You will need a Kaggle account and API token.

**Get your API token:**
1. Go to https://www.kaggle.com/settings → Account → API → Create New Token
2. This downloads a `kaggle.json` file with your credentials

**Option A — Kaggle CLI (recommended):**
```bash
pip install kaggle

# Place kaggle.json in the default location
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Download the dataset into the data/ folder
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip
```

**Option B — Store token in `.env` file:**

Create a `.env` file in the `Capstone/` directory:
```
KAGGLE_API_TOKEN=<your_token_here>
```
Then run the download helper:
```bash
python download_dataset.py
```

The dataset will be saved to `data/creditcard.csv`.

### Step 4 — Launch the notebook
```bash
jupyter notebook Paolo_Cabrera_Capstone.ipynb
```
Run all cells in order from top to bottom.

## 6. Results Summary
### Modeling highlights
- Baselines tested: Logistic Regression, Decision Tree
- Ensembles tested: Random Forest, XGBoost
- Best untuned performance came from **XGBoost** and **Random Forest**
- Hyperparameter tuning applied to top models via GridSearchCV and RandomizedSearchCV

### Best observed tuned model
From the notebook leaderboard, **XGBoost (tuned)** achieved the strongest overall fraud-class performance:
- ROC-AUC: **0.9847**
- PR-AUC: **0.9012**
- F1: **0.8889**
- Precision: **0.8936**
- Recall: **0.8842**

### Explainability
Sections 6a and 6b add SHAP and LIME-based explainability to answer:
> "Why did the model make this prediction?"

- **SHAP**: global feature impact (beeswarm, bar chart) + local prediction explanation (waterfall)
- **LIME**: local linear surrogate explanation for individual transactions

### Limitations
Section 5c documents known limitations:
- Class imbalance handled via reweighting only (no SMOTE)
- Data leakage in scaling, feature selection, and target encoding (fitted on full dataset)
- Generalization limited to one bank, two-day window, 2013 data

## 7. Author Information
- **Author**: Paolo Cabrera
- **Program/Track**: AIM - AI/ML Capstone

---
If you use this project for deployment, add leakage-safe preprocessing pipelines, temporal validation, and drift monitoring before production rollout.
