# 🏦 Loan Default Prediction (Gaussian Naïve Bayes)

Predicts whether a loan application will be **Approved** or **Rejected** from
applicant details. Includes training code, a prediction module and a Streamlit web app.

```
Load data → clean → one-hot encode → train/test split → SMOTE (train only)
          → GaussianNB → evaluate → save → serve with Streamlit
```

## Project structure

```
├── app.py                 # Streamlit web app
├── train.py               # trains the model, saves models/loan_model.joblib
├── predict.py             # predict(applicant_dict) helper
├── config.py              # column names and settings
├── requirements.txt
├── data/                  # put loan_data.csv here
├── models/                # trained model + metrics.json are saved here
├── notebooks/             # original exploration notebook
└── tests/test_predict.py
```

## 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Add your data

Copy your dataset to `data/loan_data.csv`. It needs these columns
(`applicant_id` is optional and ignored):

`age, annual_income, loan_amount, monthly_emi, credit_score, existing_loans,
has_property, city, education, employment_type, loan_status`

`loan_status` must contain `Approved` / `Rejected`.

## 3. Train

```bash
python train.py
```

This writes `models/loan_model.joblib` and `models/metrics.json`.

## 4. Run the app locally

```bash
streamlit run app.py
```

## 5. Deploy free on Streamlit Community Cloud

1. Push this repo to GitHub **including `models/loan_model.joblib`**
   (train first, then commit).
2. Go to <https://share.streamlit.io> → **New app** → pick your repo,
   branch `main`, main file `app.py` → **Deploy**.

> Train with the same library versions you deploy with (`pip install -r requirements.txt`
> first), otherwise scikit-learn may warn when loading the model file.

## Push to GitHub

```bash
git init
git add .
git commit -m "Loan default prediction with Naive Bayes"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## Use from Python

```python
from predict import predict
predict({
    "age": 25, "annual_income": 2580000, "loan_amount": 436000,
    "monthly_emi": 5400, "credit_score": 972, "existing_loans": 1,
    "has_property": 1, "city": "Bhopal", "education": "Graduate",
    "employment_type": "Self-Employed",
})
# {'prediction': 'Approved', 'probability_rejected': ..., 'probability_approved': ...}
```

## Notes on model quality

In the original notebook the test results were weak (accuracy ≈ 44%,
precision ≈ 20% and recall ≈ 59% for the *Rejected* class), so treat this as a
learning project, not a real credit-decision tool. Ideas to improve it:
try Logistic Regression / Random Forest / Gradient Boosting, scale features,
tune the decision threshold, and add more informative features.

## Tests

```bash
pytest
```
