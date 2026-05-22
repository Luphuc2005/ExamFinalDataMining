from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "diabetic_data_clean_common.csv"
BASELINE_RANKING_PATH = ROOT_DIR / "train" / "model_comparison" / "outputs" / "practical_model_ranking.csv"
OUTPUT_DIR = ROOT_DIR / "train" / "feature_engineering" / "eda_tuned_comparison_outputs"

TARGET_COLUMN = "readmitted_binary"
RANDOM_STATE = 42

DROP_COLUMNS = [
    "readmitted",
    "diag_1",
    "diag_2",
    "diag_3",
    "age_midpoint",
]

DRUG_COLUMNS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def fit_feature_engineering_params(x_train):
    return {
        "long_stay_q3": x_train["time_in_hospital"].quantile(0.75),
        "high_med_q3": x_train["num_medications"].quantile(0.75),
        "high_lab_q3": x_train["num_lab_procedures"].quantile(0.75),
    }


def add_eda_features(x, params):
    x_fe = x.copy()

    x_fe["total_prior_visits"] = (
        x_fe["number_outpatient"] + x_fe["number_emergency"] + x_fe["number_inpatient"]
    )
    x_fe["has_prior_inpatient"] = (x_fe["number_inpatient"] > 0).astype(int)
    x_fe["meds_per_day"] = x_fe["num_medications"] / x_fe["time_in_hospital"].clip(lower=1)
    x_fe["labs_per_day"] = x_fe["num_lab_procedures"] / x_fe["time_in_hospital"].clip(lower=1)
    x_fe["is_long_stay"] = (x_fe["time_in_hospital"] >= params["long_stay_q3"]).astype(int)
    x_fe["is_senior"] = (x_fe["age_ordinal"] >= 7).astype(int)
    x_fe["a1c_abnormal"] = x_fe["A1Cresult"].isin([">7", ">8"]).astype(int)
    x_fe["insulin_changed"] = x_fe["insulin"].isin(["Up", "Down"]).astype(int)

    available_drug_cols = [col for col in DRUG_COLUMNS if col in x_fe.columns]
    x_fe["num_diabetes_drugs_used"] = (
        x_fe[available_drug_cols].ne("No").sum(axis=1) if available_drug_cols else 0
    )

    diag_group_cols = [col for col in ["diag_1_group", "diag_2_group", "diag_3_group"] if col in x_fe.columns]
    x_fe["num_unique_diag_groups"] = (
        x_fe[diag_group_cols].replace("Unknown", np.nan).nunique(axis=1) if diag_group_cols else 0
    )

    # Interaction features from EDA: prior utilization + discharge/admission risk context.
    x_fe["prior_inpatient_long_stay"] = x_fe["has_prior_inpatient"] * x_fe["is_long_stay"]
    x_fe["insulin_change_a1c_abnormal"] = x_fe["insulin_changed"] * x_fe["a1c_abnormal"]
    x_fe["high_medication_load"] = (x_fe["num_medications"] >= params["high_med_q3"]).astype(int)
    x_fe["high_lab_load"] = (x_fe["num_lab_procedures"] >= params["high_lab_q3"]).astype(int)

    return x_fe


def build_preprocessor(x_train):
    numeric_features = x_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [col for col in x_train.columns if col not in numeric_features]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", make_one_hot_encoder(), categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def get_feature_names(preprocessor):
    try:
        return preprocessor.get_feature_names_out()
    except AttributeError:
        names = []
        for name, transformer, columns in preprocessor.transformers_:
            if name == "remainder" or transformer == "drop":
                continue
            if isinstance(transformer, Pipeline):
                transformer = transformer.steps[-1][1]
            if hasattr(transformer, "get_feature_names_out"):
                names.extend(transformer.get_feature_names_out(columns))
            else:
                names.extend(columns)
        return np.array(names)


def score_values(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(x)
    return model.predict(x)


def default_threshold(model):
    if hasattr(model, "predict_proba"):
        return 0.5
    if hasattr(model, "decision_function"):
        return 0.0
    return 0.5


def predict_with_threshold(scores, threshold):
    return (scores >= threshold).astype(int)


def evaluate_predictions(y_true, y_pred, scores=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }
    if scores is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, scores)
    else:
        metrics["roc_auc"] = np.nan
    return metrics


def tune_threshold(y_true, scores):
    candidate_thresholds = np.unique(np.quantile(scores, np.linspace(0.05, 0.95, 181)))
    best_threshold = candidate_thresholds[0]
    best_f1 = -1

    for threshold in candidate_thresholds:
        y_pred = predict_with_threshold(scores, threshold)
        current_f1 = f1_score(y_true, y_pred, zero_division=0)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold

    return best_threshold, best_f1


def save_confusion_matrix(y_true, y_pred, output_path):
    cm = pd.DataFrame(
        confusion_matrix(y_true, y_pred),
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    )
    cm.to_csv(output_path)


def make_models():
    return {
        "Random Forest + EDA": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight=None,
        ),
        "Random Forest + EDA balanced": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced_subsample",
            max_features="sqrt",
            min_samples_leaf=3,
        ),
        "Random Forest + EDA conservative": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced_subsample",
            max_features="sqrt",
            min_samples_leaf=5,
            max_depth=25,
        ),
        "SVM + EDA": LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=5000,
            dual=False,
        ),
    }


def add_baseline_rows(comparison_rows):
    if not BASELINE_RANKING_PATH.exists():
        return comparison_rows

    baseline_df = pd.read_csv(BASELINE_RANKING_PATH)
    baseline_df = baseline_df[baseline_df["model"].isin(["Random Forest", "SVM"])].copy()
    for _, row in baseline_df.iterrows():
        comparison_rows.append(
            {
                "experiment": "baseline_from_model_comparison",
                "model": row["model"],
                "threshold_strategy": "default",
                "threshold": np.nan,
                "accuracy": row["accuracy"],
                "precision": row["precision"],
                "recall": row["recall"],
                "f1_score": row["f1_score"],
                "roc_auc": row["roc_auc"],
                "train_time_sec": np.nan,
            }
        )
    return comparison_rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    y = df[TARGET_COLUMN]
    x = df.drop(columns=[TARGET_COLUMN])

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    fe_params = fit_feature_engineering_params(x_train)
    x_train_fe = add_eda_features(x_train, fe_params)
    x_val_fe = add_eda_features(x_val, fe_params)
    x_test_fe = add_eda_features(x_test, fe_params)

    x_train_model = x_train_fe.drop(columns=[col for col in DROP_COLUMNS if col in x_train_fe.columns])
    x_val_model = x_val_fe.drop(columns=[col for col in DROP_COLUMNS if col in x_val_fe.columns])
    x_test_model = x_test_fe.drop(columns=[col for col in DROP_COLUMNS if col in x_test_fe.columns])

    preprocessor = build_preprocessor(x_train_model)
    x_train_processed = preprocessor.fit_transform(x_train_model)
    x_val_processed = preprocessor.transform(x_val_model)
    x_test_processed = preprocessor.transform(x_test_model)
    feature_names = get_feature_names(preprocessor)

    pd.DataFrame({"feature_name": feature_names}).to_csv(OUTPUT_DIR / "encoded_feature_names.csv", index=False)
    pd.DataFrame([fe_params]).to_csv(OUTPUT_DIR / "feature_engineering_params.csv", index=False)
    pd.DataFrame(
        {
            "new_feature": [
                "total_prior_visits",
                "has_prior_inpatient",
                "meds_per_day",
                "labs_per_day",
                "is_long_stay",
                "is_senior",
                "a1c_abnormal",
                "insulin_changed",
                "num_diabetes_drugs_used",
                "num_unique_diag_groups",
                "prior_inpatient_long_stay",
                "insulin_change_a1c_abnormal",
                "high_medication_load",
                "high_lab_load",
            ]
        }
    ).to_csv(OUTPUT_DIR / "new_features.csv", index=False)

    comparison_rows = []
    comparison_rows = add_baseline_rows(comparison_rows)
    trained_models = {}
    best_validation = None

    for model_name, model in make_models().items():
        start = time.time()
        model.fit(x_train_processed, y_train)
        train_time_sec = time.time() - start
        trained_models[model_name] = model

        val_scores = score_values(model, x_val_processed)
        default_t = default_threshold(model)
        tuned_t, _ = tune_threshold(y_val, val_scores)

        for strategy, threshold in [("default", default_t), ("tuned_for_validation_f1", tuned_t)]:
            y_val_pred = predict_with_threshold(val_scores, threshold)
            metrics = evaluate_predictions(y_val, y_val_pred, val_scores)
            row = {
                "experiment": "eda_features_tuned_pipeline",
                "model": model_name,
                "threshold_strategy": strategy,
                "threshold": threshold,
                **metrics,
                "train_time_sec": train_time_sec,
            }
            comparison_rows.append(row)

            safe_name = model_name.lower().replace(" ", "_").replace("+", "plus")
            save_confusion_matrix(
                y_val,
                y_val_pred,
                OUTPUT_DIR / f"{safe_name}_{strategy}_validation_confusion_matrix.csv",
            )
            pd.DataFrame(
                classification_report(y_val, y_val_pred, output_dict=True, zero_division=0)
            ).T.to_csv(OUTPUT_DIR / f"{safe_name}_{strategy}_validation_classification_report.csv")

            if best_validation is None or row["f1_score"] > best_validation["f1_score"]:
                best_validation = row.copy()

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        ["f1_score", "roc_auc", "accuracy"], ascending=False
    )
    comparison_df.insert(0, "rank", range(1, len(comparison_df) + 1))
    comparison_df.to_csv(OUTPUT_DIR / "baseline_vs_eda_tuned_validation_comparison.csv", index=False)

    best_model_name = best_validation["model"]
    best_threshold = best_validation["threshold"]
    best_model = trained_models[best_model_name]
    test_scores = score_values(best_model, x_test_processed)
    y_test_pred = predict_with_threshold(test_scores, best_threshold)
    test_metrics = evaluate_predictions(y_test, y_test_pred, test_scores)

    test_result = {
        "selected_by": "best_validation_f1",
        "model": best_model_name,
        "threshold_strategy": best_validation["threshold_strategy"],
        "threshold": best_threshold,
        **test_metrics,
    }
    pd.DataFrame([test_result]).to_csv(OUTPUT_DIR / "final_test_metrics.csv", index=False)
    save_confusion_matrix(y_test, y_test_pred, OUTPUT_DIR / "final_test_confusion_matrix.csv")
    pd.DataFrame(
        classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
    ).T.to_csv(OUTPUT_DIR / "final_test_classification_report.csv")

    joblib.dump(preprocessor, OUTPUT_DIR / "preprocessor.joblib")
    joblib.dump(best_model, OUTPUT_DIR / "best_validation_model.joblib")

    print("\nValidation comparison")
    print(comparison_df.to_string(index=False))
    print("\nFinal test metrics")
    print(pd.DataFrame([test_result]).to_string(index=False))
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
