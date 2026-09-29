"""Train the Gaussian Naive Bayes loan model and save it to models/.

Usage:
    python train.py                       # uses data/loan_data.csv
    python train.py --data path/to.csv
"""
import argparse
import json

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import config as cfg


def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna()
    if cfg.ID_COLUMN in df.columns:
        df = df.drop(columns=cfg.ID_COLUMN)
    missing = [c for c in cfg.FEATURES + [cfg.TARGET] if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV is missing required columns: {missing}")
    return df


def train(data_path=None):
    """Train on the CSV, save model + metrics, and return the metrics."""
    df = load_data(data_path or cfg.DATA_PATH)
    X = df[cfg.FEATURES]
    y = (df[cfg.TARGET] == cfg.POSITIVE_LABEL).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE
    )

    preprocess = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cfg.CATEGORICAL_FEATURES,
            )
        ],
        remainder="passthrough",
    )

    # Fit encoder on training data only, then balance classes with SMOTE
    # (training data only, exactly like the notebook).
    X_train_enc = preprocess.fit_transform(X_train)
    X_res, y_res = SMOTE(random_state=cfg.RANDOM_STATE).fit_resample(X_train_enc, y_train)

    nb = GaussianNB().fit(X_res, y_res)

    # Bundle encoder + model so predictions take the raw columns directly.
    pipeline = Pipeline([("preprocess", preprocess), ("model", nb)])

    y_pred = pipeline.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_class": cfg.POSITIVE_LABEL,
    }

    cfg.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, cfg.MODEL_PATH)
    cfg.METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(cfg.DATA_PATH))
    args = parser.parse_args()
    metrics = train(args.data)
    print(json.dumps(metrics, indent=2))
    print(f"\nModel saved to {cfg.MODEL_PATH}")


if __name__ == "__main__":
    main()
