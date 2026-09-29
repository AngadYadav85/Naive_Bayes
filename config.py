"""Shared settings used by training, prediction and the web app."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "loan_data.csv"
MODEL_PATH = ROOT / "models" / "loan_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

TARGET = "loan_status"
ID_COLUMN = "applicant_id"          # dropped before training if present
POSITIVE_LABEL = "Rejected"         # class 1 (same as LabelEncoder in the notebook)
NEGATIVE_LABEL = "Approved"         # class 0

NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "loan_amount",
    "monthly_emi",
    "credit_score",
    "existing_loans",
    "has_property",
]
CATEGORICAL_FEATURES = ["city", "education", "employment_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE = 42
TEST_SIZE = 0.2
