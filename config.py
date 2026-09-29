"""Shared settings used by training, prediction and the web app."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _find(name, folder):
    """Use folder/name if it exists, else name in the main folder (flat upload)."""
    in_folder = ROOT / folder / name
    flat = ROOT / name
    if in_folder.exists():
        return in_folder
    if flat.exists():
        return flat
    return in_folder  # default location (used when saving a new file)


def _find_csv():
    path = _find("loan_data.csv", "data")
    if path.exists():
        return path
    others = sorted(ROOT.glob("*.csv")) + sorted((ROOT / "data").glob("*.csv"))
    return others[0] if others else path


DATA_PATH = _find_csv()
MODEL_PATH = _find("loan_model.joblib", "models")
METRICS_PATH = _find("metrics.json", "models")

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
