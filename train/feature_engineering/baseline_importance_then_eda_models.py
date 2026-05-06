from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from eda_tuned_end_to_end_comparison import (
    DATA_PATH,
    DROP_COLUMNS,
    OUTPUT_DIR as OLD_OUTPUT_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    add_eda_features,
    build_preprocessor,
    fit_feature_engineering_params,
    get_feature_names,
    make_one_hot_encoder,
    predict_with_threshold,
    score_values,
    tune_threshold,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "train" / "feature_engineering" / "baseline_importance_then_eda_outputs"
TOP_K_VALUES = [30, 50, 100, 150]


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


def build_models():
    return {
        "Dummy Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            solver="saga",
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced_subsample",
            max_features="sqrt",
            min_samples_leaf=3,
        ),
        "SVM": LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=5000,
            dual=False,
        ),
        "Naive Bayes": GaussianNB(),
    }


def save_confusion_matrix(y_true, y_pred, output_path):
    cm = pd.DataFrame(
        confusion_matrix(y_true, y_pred),
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    )
    cm.to_csv(output_path)


def default_threshold_for_model(model):
    if hasattr(model, "predict_proba"):
        return 0.5
    if hasattr(model, "decision_function"):
        return 0.0
    return 0.5


def fit_rf_feature_selector(x_train, y_train, feature_names):
    selector = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
        max_features="sqrt",
        min_samples_leaf=3,
    )
    selector.fit(x_train, y_train)

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": selector.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance_df.insert(0, "rank", range(1, len(importance_df) + 1))

    return selector, importance_df


def train_and_evaluate_model_set(
    x_train,
    y_train,
    x_val,
    y_val,
    experiment,
    feature_set,
    output_dir,
    tune_threshold=False,
):
    rows = []
    trained_models = {}

    for model_name, model in build_models().items():
        start = time.time()
        model.fit(x_train, y_train)
        train_time_sec = time.time() - start
        trained_models[model_name] = model

        scores = score_values(model, x_val)
        threshold = default_threshold_for_model(model)
        threshold_strategy = "default"

        if tune_threshold and model_name in ["Random Forest", "Logistic Regression", "SVM"]:
            threshold, _ = tune_threshold_fn(y_val, scores)
            threshold_strategy = "tuned_for_validation_f1"

        if hasattr(model, "predict_proba") or hasattr(model, "decision_function"):
            y_pred = predict_with_threshold(scores, threshold)
        else:
            y_pred = model.predict(x_val)

        metrics = evaluate_predictions(y_val, y_pred, scores)
        row = {
            "experiment": experiment,
            "feature_set": feature_set,
            "model": model_name,
            "threshold_strategy": threshold_strategy,
            "threshold": threshold,
            **metrics,
            "train_time_sec": train_time_sec,
        }
        rows.append(row)

        safe = f"{experiment}_{feature_set}_{model_name}".lower()
        safe = safe.replace(" ", "_").replace("+", "plus").replace("/", "_")
        save_confusion_matrix(y_val, y_pred, output_dir / f"{safe}_validation_confusion_matrix.csv")
        pd.DataFrame(
            classification_report(y_val, y_pred, output_dict=True, zero_division=0)
        ).T.to_csv(output_dir / f"{safe}_validation_classification_report.csv")

    return rows, trained_models


def tune_threshold_fn(y_true, scores):
    return tune_threshold(y_true, scores)


def prepare_baseline_data(x_train, x_val, x_test):
    x_train_model = x_train.drop(columns=[col for col in DROP_COLUMNS if col in x_train.columns])
    x_val_model = x_val.drop(columns=[col for col in DROP_COLUMNS if col in x_val.columns])
    x_test_model = x_test.drop(columns=[col for col in DROP_COLUMNS if col in x_test.columns])

    preprocessor = build_preprocessor(x_train_model)
    x_train_processed = preprocessor.fit_transform(x_train_model)
    x_val_processed = preprocessor.transform(x_val_model)
    x_test_processed = preprocessor.transform(x_test_model)
    feature_names = get_feature_names(preprocessor)

    return preprocessor, x_train_processed, x_val_processed, x_test_processed, feature_names


def prepare_eda_data(x_train, x_val, x_test):
    fe_params = fit_feature_engineering_params(x_train)
    x_train_fe = add_eda_features(x_train, fe_params)
    x_val_fe = add_eda_features(x_val, fe_params)
    x_test_fe = add_eda_features(x_test, fe_params)

    preprocessor, x_train_processed, x_val_processed, x_test_processed, feature_names = prepare_baseline_data(
        x_train_fe, x_val_fe, x_test_fe
    )

    return fe_params, preprocessor, x_train_processed, x_val_processed, x_test_processed, feature_names


def evaluate_best_on_test(best_row, trained_model, x_test, y_test, output_dir):
    scores = score_values(trained_model, x_test)
    if hasattr(trained_model, "predict_proba") or hasattr(trained_model, "decision_function"):
        y_pred = predict_with_threshold(scores, best_row["threshold"])
    else:
        y_pred = trained_model.predict(x_test)

    metrics = evaluate_predictions(y_test, y_pred, scores)
    result = {
        "selected_by": "best_validation_f1",
        "experiment": best_row["experiment"],
        "feature_set": best_row["feature_set"],
        "model": best_row["model"],
        "threshold_strategy": best_row["threshold_strategy"],
        "threshold": best_row["threshold"],
        **metrics,
    }
    pd.DataFrame([result]).to_csv(output_dir / "final_test_metrics.csv", index=False)
    save_confusion_matrix(y_test, y_pred, output_dir / "final_test_confusion_matrix.csv")
    pd.DataFrame(
        classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    ).T.to_csv(output_dir / "final_test_classification_report.csv")
    return result
