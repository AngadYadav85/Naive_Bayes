import pytest

import config as cfg

pytestmark = pytest.mark.skipif(
    not cfg.MODEL_PATH.exists(), reason="Train the model first: python train.py"
)


def test_predict_returns_valid_output():
    from predict import get_categories, predict

    cats = get_categories()
    applicant = {
        "age": 30, "annual_income": 600000, "loan_amount": 300000,
        "monthly_emi": 10000, "credit_score": 700, "existing_loans": 1,
        "has_property": 1,
        "city": cats["city"][0],
        "education": cats["education"][0],
        "employment_type": cats["employment_type"][0],
    }
    out = predict(applicant)
    assert out["prediction"] in {cfg.POSITIVE_LABEL, cfg.NEGATIVE_LABEL}
    assert abs(out["probability_rejected"] + out["probability_approved"] - 1) < 1e-9
