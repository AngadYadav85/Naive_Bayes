"""Streamlit web app for the loan approval model.  Run: streamlit run app.py"""
import json

import pandas as pd
import streamlit as st

import config as cfg
from predict import get_categories, predict

st.set_page_config(page_title="Loan Default Prediction", page_icon="🏦")
st.title("🏦 Loan Default Prediction")
st.caption("Gaussian Naïve Bayes model · enter applicant details to get a prediction")

if not cfg.MODEL_PATH.exists():
    st.error("Model file not found. Run `python train.py` and commit `models/loan_model.joblib`.")
    st.stop()

cats = get_categories()

with st.form("applicant"):
    c1, c2 = st.columns(2)
    age = c1.number_input("Age", 18, 100, 30)
    annual_income = c2.number_input("Annual income", 0, 100_000_000, 600_000, step=10_000)
    loan_amount = c1.number_input("Loan amount", 0, 100_000_000, 300_000, step=10_000)
    monthly_emi = c2.number_input("Monthly EMI", 0, 10_000_000, 10_000, step=500)
    credit_score = c1.number_input("Credit score", 0, 1000, 700)
    existing_loans = c2.number_input("Existing loans", 0, 50, 0)
    has_property = c1.selectbox("Owns property?", [1, 0], format_func=lambda v: "Yes" if v else "No")
    city = c2.selectbox("City", cats["city"])
    education = c1.selectbox("Education", cats["education"])
    employment_type = c2.selectbox("Employment type", cats["employment_type"])
    submitted = st.form_submit_button("Predict")

if submitted:
    result = predict(
        {
            "age": age,
            "annual_income": annual_income,
            "loan_amount": loan_amount,
            "monthly_emi": monthly_emi,
            "credit_score": credit_score,
            "existing_loans": existing_loans,
            "has_property": has_property,
            "city": city,
            "education": education,
            "employment_type": employment_type,
        }
    )
    if result["prediction"] == cfg.POSITIVE_LABEL:
        st.error(f"Prediction: **{result['prediction']}**")
    else:
        st.success(f"Prediction: **{result['prediction']}**")
    st.progress(result["probability_rejected"], text=f"Probability of rejection: {result['probability_rejected']:.1%}")

if cfg.METRICS_PATH.exists():
    with st.expander("Model performance (held-out test set)"):
        m = json.loads(cfg.METRICS_PATH.read_text())
        st.dataframe(
            pd.DataFrame(
                {k: [round(m[k], 3)] for k in ["accuracy", "precision", "recall", "f1"]}
            ),
            hide_index=True,
        )
