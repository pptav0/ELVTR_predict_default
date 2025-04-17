from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib

# --- Load Pre-trained Pipeline (LightGBM or RF Full Pipeline) ---
MODEL_PATH = '../../models/lgb_pipeline_full.pkl'   # or 'rf_pipeline_full.pkl'
pipeline = joblib.load(MODEL_PATH)

# --- FastAPI App Init ---
app = FastAPI(title="Fraud Detection API")

# --- Pydantic Input Model ---
class InputRecord(BaseModel):
    term: str
    int_rate: str
    grade: str
    sub_grade: str
    emp_title: str
    emp_length: str
    home_ownership: str
    verification_status: str
    issue_d: str
    loan_status: str
    pymnt_plan: str
    purpose: str
    title: str
    zip_code: str
    addr_state: str
    earliest_cr_line: str
    revol_util: str
    initial_list_status: str
    last_pymnt_d: str
    next_pymnt_d: str
    last_credit_pull_d: str
    application_type: str
    hardship_flag: str
    debt_settlement_flag: str
    dti: float
    dti_joint: float = None
    il_util: float = None
    all_util: float = None
    bc_util: float = None
    total_acc: int
    mo_sin_old_il_acct: float
    mo_sin_old_rev_tl_op: float
    mths_since_last_delinq: float
    mths_since_last_record: float
    mths_since_last_major_derog: float
    mths_since_recent_bc_dlq: float
    mths_since_recent_revol_delinq: float
    percent_bc_gt_75: float
    last_fico_range_high: float
    last_fico_range_low: float
    sec_app_fico_range_low: float
    sec_app_fico_range_high: float
    sec_app_open_acc: float
    sec_app_revol_util: float

# --- Helper Function to Convert Input to DataFrame ---
def preprocess_raw_data(input_data: List[InputRecord]) -> pd.DataFrame:
    df = pd.DataFrame([r.dict() for r in input_data])
    return df

# --- Endpoint ---
@app.post("/predict")
def predict(input_data: List[InputRecord]):
    try:
        df_raw = preprocess_raw_data(input_data)
        y_pred = pipeline.predict(df_raw)
        return {"predictions": y_pred.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
