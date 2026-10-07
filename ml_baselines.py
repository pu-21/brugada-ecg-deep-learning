from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from data_utils import load_feature_csv, make_split
from metrics import build_pr_curve, compute_classification_metrics, find_best_f1_threshold, save_json


def get_models(random_state: int = 42):
    return {
        "lr": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "xgboost": XGBClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        ),
        "rf": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="../model_ready_ecg_features.csv")
    parser.add_argument("--target", default="brugada")
    parser.add_argument("--outdir", default="outputs/ml")
    args = parser.parse_args()

    x, y = load_feature_csv(args.csv, target_col=args.target)
    split = make_split(x, y)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, model in get_models().items():
        model.fit(split.x_train, split.y_train)
        val_prob = model.predict_proba(split.x_val)[:, 1]
        threshold = find_best_f1_threshold(split.y_val, val_prob)
        test_prob = model.predict_proba(split.x_test)[:, 1]
        metrics = compute_classification_metrics(split.y_test, test_prob, threshold=threshold)
        metrics["model"] = name
        metrics["val_auprc"] = float(average_precision_score(split.y_val, val_prob))
        rows.append(metrics)
        save_json(metrics, outdir / f"{name}_metrics.json")
        save_json(build_pr_curve(split.y_test, test_prob), outdir / f"{name}_pr_curve.json")
        joblib.dump(model, outdir / f"{name}.joblib")
        print(name, metrics)

    pd.DataFrame(rows).sort_values("auprc", ascending=False).to_csv(outdir / "summary.csv", index=False)


if __name__ == "__main__":
    main()
