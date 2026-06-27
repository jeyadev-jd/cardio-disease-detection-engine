"""Cardio risk modeling project with a CLI interface.

Usage examples:
    python cardio.py train --data cardio_train.csv --sep ";"
    python cardio.py train --data cardio_train.csv --output-dir outputs
    python cardio.py predict --model-path outputs/best_model.pkl --age-years 54 --height 168 \
        --weight 79 --ap-hi 145 --ap-lo 92 --cholesterol 2 --gluc 1 --smoke 0 --alco 0 --active 1

If no dataset path is provided or the file is missing, a synthetic dataset is generated so the
pipeline can always be executed end-to-end.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TARGET_COL = "cardio"


@dataclass
class TrainArtifacts:
    best_model_name: str
    best_auc: float
    metrics: pd.DataFrame
    feature_columns: list[str]
    model_path: Path
    metrics_path: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cardiovascular disease risk modeling interface"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and evaluate multiple models")
    train_parser.add_argument("--data", type=str, default="", help="Path to dataset CSV")
    train_parser.add_argument("--sep", type=str, default=";", help="CSV separator")
    train_parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    train_parser.add_argument("--test-size", type=float, default=0.2, help="Test split size")

    predict_parser = subparsers.add_parser("predict", help="Predict CVD risk for one person")
    predict_parser.add_argument("--model-path", type=str, required=True, help="Path to saved model")
    predict_parser.add_argument("--age-years", type=float, required=True)
    predict_parser.add_argument("--height", type=float, required=True)
    predict_parser.add_argument("--weight", type=float, required=True)
    predict_parser.add_argument("--ap-hi", type=float, required=True)
    predict_parser.add_argument("--ap-lo", type=float, required=True)
    predict_parser.add_argument("--cholesterol", type=int, choices=[1, 2, 3], required=True)
    predict_parser.add_argument("--gluc", type=int, choices=[1, 2, 3], required=True)
    predict_parser.add_argument("--smoke", type=int, choices=[0, 1], required=True)
    predict_parser.add_argument("--alco", type=int, choices=[0, 1], required=True)
    predict_parser.add_argument("--active", type=int, choices=[0, 1], required=True)

    return parser


def _generate_synthetic_cardio_data(n_samples: int = 5000) -> pd.DataFrame:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=7,
        n_redundant=1,
        n_classes=2,
        class_sep=1.1,
        random_state=RANDOM_STATE,
    )

    df = pd.DataFrame(
        {
            "age": (X[:, 0] * 2500 + 18500).clip(10950, 27375).astype(int),
            "height": (X[:, 1] * 12 + 168).clip(140, 210),
            "weight": (X[:, 2] * 15 + 73).clip(40, 140),
            "ap_hi": (X[:, 3] * 18 + 128).clip(85, 210),
            "ap_lo": (X[:, 4] * 12 + 82).clip(55, 140),
            "cholesterol": np.digitize(X[:, 5], bins=[-0.6, 0.6]) + 1,
            "gluc": np.digitize(X[:, 6], bins=[-0.7, 0.7]) + 1,
            "smoke": (X[:, 7] > 0.4).astype(int),
            "alco": (X[:, 8] > 0.5).astype(int),
            "active": (X[:, 9] > -0.1).astype(int),
            TARGET_COL: y.astype(int),
        }
    )
    return df


def load_dataset(data_path: str, sep: str = ";") -> Tuple[pd.DataFrame, str]:
    if data_path and Path(data_path).exists():
        df = pd.read_csv(data_path, sep=sep)
        source = f"file:{data_path}"
        return df, source

    df = _generate_synthetic_cardio_data()
    source = "synthetic"
    return df, source


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "age" in out.columns and "age_years" not in out.columns:
        out["age_years"] = out["age"] / 365.0
    elif "age_years" not in out.columns:
        raise ValueError("Dataset must include either 'age' (days) or 'age_years'.")

    out["BMI"] = out["weight"] / (out["height"] / 100.0) ** 2
    out["is_hypertensive"] = ((out["ap_hi"] >= 140) | (out["ap_lo"] >= 90)).astype(int)
    out["pulse_pressure"] = out["ap_hi"] - out["ap_lo"]
    out["MAP"] = (out["ap_hi"] + 2 * out["ap_lo"]) / 3.0
    out["high_cholesterol"] = (out["cholesterol"] >= 2).astype(int)
    out["high_glucose"] = (out["gluc"] >= 2).astype(int)

    out["risk_factors"] = (
        out["is_hypertensive"]
        + out["high_cholesterol"]
        + out["high_glucose"]
        + out["smoke"].astype(int)
        + (out["BMI"] >= 25).astype(int)
    )

    out["risk_score"] = (
        (out["age_years"] / out["age_years"].max()) * 25
        + ((out["ap_hi"] - 100) / 100).clip(0, 1) * 30
        + (out["cholesterol"] / 3) * 20
        + (out["gluc"] / 3) * 15
        + (out["smoke"].astype(float) * 10)
    ).clip(0, 100)

    return out


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    feature_cols = [
        "age_years",
        "height",
        "weight",
        "BMI",
        "ap_hi",
        "ap_lo",
        "cholesterol",
        "gluc",
        "smoke",
        "alco",
        "active",
        "is_hypertensive",
        "pulse_pressure",
        "MAP",
        "risk_factors",
        "risk_score",
    ]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns after engineering: {missing}")
    return feature_cols


def train_models(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[Dict[str, Pipeline], pd.DataFrame, list[str]]:
    df = add_engineered_features(df)

    required_cols = ["height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active", TARGET_COL]
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(f"Dataset missing required columns: {missing_required}")

    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )

    models: Dict[str, Pipeline] = {
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
        "RandomForest": Pipeline(
            [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=12,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                )
            ]
        ),
        "GradientBoosting": Pipeline(
            [
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=200,
                        learning_rate=0.08,
                        max_depth=3,
                        random_state=RANDOM_STATE,
                    ),
                )
            ]
        ),
    }

    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1": f1_score(y_test, y_pred),
                "ROC_AUC": roc_auc_score(y_test, y_proba),
            }
        )

    metrics_df = pd.DataFrame(rows).sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
    return models, metrics_df, feature_cols


def save_outputs(
    models: Dict[str, Pipeline],
    metrics_df: pd.DataFrame,
    feature_columns: list[str],
    output_dir: Path,
) -> TrainArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)

    best_model_name = str(metrics_df.iloc[0]["Model"])
    best_auc = float(metrics_df.iloc[0]["ROC_AUC"])
    best_model = models[best_model_name]

    model_bundle = {
        "model": best_model,
        "feature_columns": feature_columns,
        "model_name": best_model_name,
    }

    model_path = output_dir / "best_model.pkl"
    metrics_path = output_dir / "metrics.csv"
    metadata_path = output_dir / "run_metadata.json"

    joblib.dump(model_bundle, model_path)
    metrics_df.to_csv(metrics_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "best_model": best_model_name,
                "best_auc": round(best_auc, 6),
                "feature_columns": feature_columns,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return TrainArtifacts(
        best_model_name=best_model_name,
        best_auc=best_auc,
        metrics=metrics_df,
        feature_columns=feature_columns,
        model_path=model_path,
        metrics_path=metrics_path,
    )


def build_single_row_from_values(
    age_years: float,
    height: float,
    weight: float,
    ap_hi: float,
    ap_lo: float,
    cholesterol: int,
    gluc: int,
    smoke: int,
    alco: int,
    active: int,
) -> pd.DataFrame:
    row = pd.DataFrame(
        [
            {
                "age_years": float(age_years),
                "height": float(height),
                "weight": float(weight),
                "ap_hi": float(ap_hi),
                "ap_lo": float(ap_lo),
                "cholesterol": int(cholesterol),
                "gluc": int(gluc),
                "smoke": int(smoke),
                "alco": int(alco),
                "active": int(active),
            }
        ]
    )

    row["BMI"] = row["weight"] / (row["height"] / 100.0) ** 2
    row["is_hypertensive"] = ((row["ap_hi"] >= 140) | (row["ap_lo"] >= 90)).astype(int)
    row["pulse_pressure"] = row["ap_hi"] - row["ap_lo"]
    row["MAP"] = (row["ap_hi"] + 2 * row["ap_lo"]) / 3.0
    row["high_cholesterol"] = (row["cholesterol"] >= 2).astype(int)
    row["high_glucose"] = (row["gluc"] >= 2).astype(int)
    row["risk_factors"] = (
        row["is_hypertensive"]
        + row["high_cholesterol"]
        + row["high_glucose"]
        + row["smoke"]
        + (row["BMI"] >= 25).astype(int)
    )
    row["risk_score"] = (
        (row["age_years"] / 75.0) * 25
        + ((row["ap_hi"] - 100) / 100).clip(0, 1) * 30
        + (row["cholesterol"] / 3) * 20
        + (row["gluc"] / 3) * 15
        + (row["smoke"] * 10)
    ).clip(0, 100)

    return row


def build_single_row_from_args(args: argparse.Namespace) -> pd.DataFrame:
    return build_single_row_from_values(
        age_years=args.age_years,
        height=args.height,
        weight=args.weight,
        ap_hi=args.ap_hi,
        ap_lo=args.ap_lo,
        cholesterol=args.cholesterol,
        gluc=args.gluc,
        smoke=args.smoke,
        alco=args.alco,
        active=args.active,
    )


def run_train(args: argparse.Namespace) -> int:
    df, source = load_dataset(args.data, args.sep)
    models, metrics_df, feature_cols = train_models(df, test_size=args.test_size)
    base_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (base_dir / output_dir).resolve()
    artifacts = save_outputs(models, metrics_df, feature_cols, output_dir)

    print("=" * 80)
    print("CARDIO MODEL TRAINING COMPLETE")
    print("=" * 80)
    print(f"Data source   : {source}")
    print(f"Rows, columns : {df.shape}")
    print(f"Best model    : {artifacts.best_model_name}")
    print(f"Best ROC-AUC  : {artifacts.best_auc:.4f}")
    print(f"Model saved   : {artifacts.model_path}")
    print(f"Metrics saved : {artifacts.metrics_path}")
    print("\nLeaderboard:")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    return 0


def run_predict(args: argparse.Namespace) -> int:
    model_bundle = joblib.load(args.model_path)
    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]
    model_name = model_bundle.get("model_name", "Unknown")

    row = build_single_row_from_args(args)
    X = row[feature_columns]

    proba = float(model.predict_proba(X)[:, 1][0])
    pred = int(proba >= 0.5)
    label = "High CVD risk" if pred == 1 else "Low CVD risk"

    print("=" * 80)
    print("CARDIO RISK PREDICTION")
    print("=" * 80)
    print(f"Model used        : {model_name}")
    print(f"Predicted class   : {pred} ({label})")
    print(f"Risk probability  : {proba:.4f}")
    print(f"Confidence margin : {abs(proba - 0.5):.4f}")

    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        return run_train(args)

    if args.command == "predict":
        return run_predict(args)

    parser.error("Unknown command")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
