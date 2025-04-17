# Fraud Detection Model Deployment

This project contains trained machine learning pipelines for fraud/default prediction using both Random Forest and LightGBM classifiers. The models support different levels of preprocessing integration.

## Feature Engineering Overview

The features used were engineered from raw data using the `feature_engineering()` pipeline (custom parsing, encoding, and transformations).

### Numeric Features (with median imputation)

- `int_rate_num`, `dti`, `dti_joint`, `bc_util`, `all_util`, `il_util`
- `total_acc`, `mo_sin_old_il_acct`, `mo_sin_old_rev_tl_op`
- `mths_since_last_delinq`, `mths_since_last_record`, `mths_since_last_major_derog`
- `mths_since_recent_bc_dlq`, `mths_since_recent_revol_delinq`, `percent_bc_gt_75`
- `last_fico_range_low`, `last_fico_range_high`
- `sec_app_fico_range_low`, `sec_app_fico_range_high`
- `emp_length_num`, `term_months`, `sec_app_revol_util`, `sec_app_open_acc`

### Categorical Encoded Features

Encoded via `LabelEncoder` or `OrdinalEncoder`:
- `emp_title_clean_encoded`, `home_ownership_encoded`, `purpose_encoded`, `grade_encoded`, `verification_status_encoded`, `application_type_encoded`

Missing values are filled with default codes or constants.

### Binary Flag Features

Binarized from 'y'/'n':
- `hardship_flag_bin`, `debt_settlement_flag_bin`, `pymnt_plan_bin`

---

## Files Included

| File                   | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `rf_pipeline.pkl`      | Random Forest model trained on preprocessed features (requires preprocessing) |
| `rf_pipeline_full.pkl` | Random Forest pipeline including preprocessing and model                     |
| `lgb_model.pkl`        | LightGBM model only (requires manual preprocessing and thresholding)         |
| `lgb_threshold.pkl`    | Tuned classification threshold (e.g., 0.68) for use with `lgb_model.pkl`     |
| `lgb_pipeline_full.pkl`| LightGBM pipeline with preprocessing and threshold logic built in            |

---

## How to Use the Models

### Option 1: `rf_pipeline.pkl` — Requires Preprocessed Features

```python
import joblib

rf_model = joblib.load('rf_pipeline.pkl')
y_pred = rf_model.predict(X_preprocessed)
```
This requires manual application of feature_engineering() and handling missing values.

### Option 2: `lgb_model.pkl` + `lgb_threshold.pkl` — Manual Preprocessing and Thresholding

```python
import joblib

model = joblib.load('lgb_model.pkl')
threshold = joblib.load('lgb_threshold.pkl')

X_proc = feature_engineering(raw_df)
y_probs = model.predict(X_proc)
y_pred = (y_probs >= threshold).astype(int)
```
Use only if you want to experiment with thresholds or apply custom input handling.


### Option 3: `rf_pipeline_full.pkl` — Full Pipeline (Raw Input Accepted)

```python
import joblib

pipeline = joblib.load('rf_pipeline_full.pkl')
y_pred = pipeline.predict(raw_df)
```
This includes all encoding, imputation, and model steps inside the pipeline. Ready for deployment or API use.


### Option 4 (Recommended): `lgb_pipeline_full.pkl` — LightGBM with Preprocessing and Threshold

```python
import joblib

pipeline = joblib.load('lgb_pipeline_full.pkl')
y_pred = pipeline.predict(raw_df)

```
This pipeline handles:

- Feature engineering

- Imputation

- Encoding

- LightGBM classification

- Tuned threshold application

## Comparison Table

| Step                     | rf_pipeline.pkl | lgb_model.pkl | rf_pipeline_full.pkl | lgb_pipeline_full.pkl |
|--------------------------|------------------|----------------|------------------------|--------------------------|
| Raw Data Accepted        | No               | No             | Yes                    | Yes                      |
| Feature Engineering      | Manual           | Manual         | Included               | Included                 |
| Missing Value Handling   | Manual           | Manual         | Included               | Included                 |
| Threshold Application    | Default (0.5)    | Uses .pkl      | Default (0.5)          | Tuned (e.g., 0.68)       |

## Setup requirement

```python
pip install scikit-learn lightgbm pandas joblib
```

## Usage Recommendation
For production, deployment, or consistent prediction results:

- Use `rf_pipeline_full.pkl` for a robust, interpretable Random Forest model with full preprocessing included.

- Use `lgb_pipeline_full.pkl` for the best-performing LightGBM model with preprocessing and optimized threshold logic built-in.

Both models accept raw data input and encapsulate all preprocessing and feature handling logic inside the pipeline.

### Example: Loading and Predicting from Full Pipeline

```python
import joblib
import pandas as pd

# Load raw input data
raw_df = pd.read_csv("your_raw_input.csv")

# Load the full pipeline (choose one)
model = joblib.load("lgb_pipeline_full.pkl")   # or "rf_pipeline_full.pkl"

# Predict fraud/default classification
y_pred = model.predict(raw_df)
```
If your data matches the structure used during training, the pipeline will automatically apply all transformations and return predictions.
