from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "classification"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
TARGET_COLUMN = "readmitted_binary"
RANDOM_STATE = 42


def load_split(filename: str):
    data = pd.read_csv(DATA_DIR / filename)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    x = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]
    return x, y


def evaluate(model, x, y, split_name: str):
    y_pred = model.predict(x)

    metrics = {
        "split": split_name,
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1_score": f1_score(y, y_pred, zero_division=0),
    }

    cm = pd.DataFrame(
        confusion_matrix(y, y_pred),
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    )

    report = pd.DataFrame(classification_report(y, y_pred, output_dict=True, zero_division=0)).T
    return metrics, cm, report


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    x_train, y_train = load_split("classification_train_processed.csv")
    x_val, y_val = load_split("classification_validation_processed.csv")
    x_test, y_test = load_split("classification_test_processed.csv")

    model = LogisticRegression(
        max_iter=1000,
        solver="saga",
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train, y_train)

    all_metrics = []
    for split_name, x, y in [
        ("validation", x_val, y_val),
        ("test", x_test, y_test),
    ]:
        metrics, cm, report = evaluate(model, x, y, split_name)
        all_metrics.append(metrics)
        cm.to_csv(OUTPUT_DIR / f"{split_name}_confusion_matrix.csv")
        report.to_csv(OUTPUT_DIR / f"{split_name}_classification_report.csv")

    pd.DataFrame(all_metrics).to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)
    joblib.dump(model, OUTPUT_DIR / "logistic_regression_model.joblib")

    print(pd.DataFrame(all_metrics).to_string(index=False))
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
