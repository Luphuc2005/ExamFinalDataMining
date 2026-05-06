MODEL COMPARISON WORKFLOW

This folder handles steps 1 to 4:

1. Create a model comparison notebook.
2. Train baseline models:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - KNN
   - SVM
   - Naive Bayes
3. Evaluate each model on the validation set.
4. Select the top 2 models by validation F1-score.

Notebook:
model_comparison.ipynb

Input data:
data/classification/classification_train_processed.csv
data/classification/classification_validation_processed.csv

Target column:
readmitted_binary

Main ranking metric:
validation f1_score

Reason:
For readmission prediction, Accuracy alone can be misleading.
F1-score balances Precision and Recall for class 1.

Important:
The test set is not used in this notebook.
Keep the test set for final evaluation after feature engineering and retraining the selected top 2 models.

Saved outputs:
train/model_comparison/outputs/model_comparison_metrics.csv
train/model_comparison/outputs/top2_models.csv
train/model_comparison/outputs/*_validation_confusion_matrix.csv
train/model_comparison/outputs/*_model.joblib

Note about SVM:
The notebook uses LinearSVC instead of full kernel SVM because the dataset has more than 65,000 training rows.
Kernel SVM may be too slow for this dataset.

