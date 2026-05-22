FEATURE ENGINEERING NOTEBOOKS

Recommended workflow if you want this exact story:

baseline_importance_then_eda_models.ipynb

This notebook follows the order:
1. Run baseline preprocessing from the clean common data.
2. Train a baseline Random Forest selector on train only.
3. Save baseline feature importance and top-K selected features.
4. Train multiple models on all baseline features and top-K baseline features.
5. Add EDA-based features after the baseline comparison.
6. Retrain the models and compare before/after feature engineering.
7. Evaluate the best validation model on the test set once.

Outputs are saved to:
train/feature_engineering/baseline_importance_then_eda_outputs/

Main files:
- baseline_rf_feature_importance.csv
- baseline_top_30_features.csv
- baseline_top_50_features.csv
- baseline_top_100_features.csv
- baseline_top_150_features.csv
- baseline_importance_then_eda_validation_comparison.csv
- eda_rf_feature_importance.csv
- final_test_metrics.csv

Recommended notebook for your report workflow:

eda_10_features_retrain.ipynb

This is the cleaner version. It adds only 10 meaningful EDA-based features, so it is easier to explain and less likely to add noise.

Features added:
- total_prior_visits
- has_prior_inpatient
- meds_per_day
- labs_per_day
- is_long_stay
- is_senior
- a1c_abnormal
- insulin_changed
- num_diabetes_drugs_used
- num_unique_diag_groups

Outputs are saved to:
train/feature_engineering/eda_10_features_outputs/

Recommended end-to-end comparison script if you want a stronger metric experiment:

eda_tuned_end_to_end_comparison.ipynb
eda_tuned_end_to_end_comparison.py

The notebook is the report-friendly version. The Python script contains the reusable pipeline functions.

This workflow runs a full validation-comparison experiment:
- loads data/diabetic_data_clean_common.csv
- splits train/validation/test before fitting feature thresholds
- adds EDA-based features and interaction flags
- preprocesses numeric and categorical columns
- trains Random Forest and SVM variants
- tunes the classification threshold on validation F1
- compares the new experiments with the baseline Random Forest and SVM ranking
- evaluates the best validation model on the test set once

Outputs are saved to:
train/feature_engineering/eda_tuned_comparison_outputs/

Main comparison table:
train/feature_engineering/eda_tuned_comparison_outputs/baseline_vs_eda_tuned_validation_comparison.csv

Final held-out test result:
train/feature_engineering/eda_tuned_comparison_outputs/final_test_metrics.csv

Next experiment if you want to combine EDA features with feature selection:

eda_10_features_rf_selection_retrain.ipynb

This notebook:
- adds the same 10 EDA-based features
- preprocesses the data
- uses Random Forest feature_importances_ on train only
- selects top K encoded features: 50, 75, 100, 150
- retrains Random Forest and SVM
- compares Baseline vs +10 EDA features vs +10 EDA features + RF selection

Outputs are saved to:
train/feature_engineering/eda_10_rf_selection_outputs/

Alternative broader notebook:

eda_feature_engineering_retrain.ipynb

This notebook follows the intended project story:

1. Baseline phase
   Use near-original cleaned data, remove meaningless columns, preprocess, and train multiple models.

2. Model comparison phase
   Select practical top models from validation metrics.
   In your current result table, the selected top models are:
   - Random Forest
   - SVM

3. EDA-based feature engineering phase
   Create new features from EDA insights, such as:
   - total_prior_visits
   - has_prior_inpatient
   - labs_per_day
   - meds_per_day
   - is_long_stay_fixed
   - is_senior
   - a1c_tested
   - a1c_abnormal
   - insulin_used
   - insulin_changed
   - num_diabetes_drugs_used
   - num_drugs_changed
   - same_diag_1_2_group
   - num_unique_diag_groups

4. Retraining phase
   Preprocess again and retrain Random Forest and SVM with the engineered features.

Important leakage rule:
The notebook splits train/validation/test first.
Any learned thresholds such as Q3 are fitted only on train, then applied to validation and test.

The broader notebook outputs are saved to:
train/feature_engineering/eda_outputs/

Older notebook:
feature_engineering_retrain.ipynb

This older notebook focuses more on feature selection using Random Forest importance.
It is still useful, but the EDA-based notebook is clearer for the workflow you described.
