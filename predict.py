"""Load the trained model and score a single applicant.

Usage:
    python predict.py            # runs a built-in example
Or from Python:
    from predict import predict
    predict({...})
"""
import joblib
import pandas as pd

import config as cfg

_model = None


def get_model():
    global _model
    if _model is None:
        if not cfg.MODEL_PATH.exists():
            raise FileNotFoundError(
                f"{cfg.MODEL_PATH} not found. Run `python train.py` first."
            )
        _model = joblib.load(cfg.MODEL_PATH)
    return _model


def get_categories():
    """Category values the model learned, e.g. {'city': ['Bhopal', ...], ...}."""
    encoder = get_model().named_steps["preprocess"].named_transformers_["cat"]
    return {
        col: [str(v) for v in cats]
        for col, cats in zip(cfg.CATEGORICAL_FEATURES, encoder.categories_)
    }


def predict(applicant: dict) -> dict:
    row = pd.DataFrame([applicant])[cfg.FEATURES]
    model = get_model()
    proba_reject = float(model.predict_proba(row)[0][1])
    label = cfg.POSITIVE_LABEL if proba_reject >= 0.5 else cfg.NEGATIVE_LABEL
    return {
        "prediction": label,
        "probability_rejected": proba_reject,
        "probability_approved": 1 - proba_reject,
    }


if __name__ == "__main__":
    example = {
        "age": 25,
        "annual_income": 2580000,
        "loan_amount": 436000,
        "monthly_emi": 5400,
        "credit_score": 972.0,
        "existing_loans": 1,
        "has_property": 1,
        "city": "Bhopal",
        "education": "Graduate",
        "employment_type": "Self-Employed",
    }
    print(predict(example))
