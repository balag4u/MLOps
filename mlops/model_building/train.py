# -----------------------------
# train.py – FINAL CORRECTED VERSION
# -----------------------------

import os
import pandas as pd
import mlflow
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# =====================================================================================
# FIX: USE LOCAL MLFLOW TRACKING INSTEAD OF localhost:5000
# =====================================================================================
os.environ.pop("MLFLOW_TRACKING_URI", None)  # Remove outside env config
mlflow.set_tracking_uri("file:./mlruns")     # Local tracking folder
mlflow.set_experiment("mlops-training-experiment")

api = HfApi()

# =====================================================================================
# DATA PATHS FROM HUGGING FACE
# =====================================================================================
Xtrain_path = "hf://datasets/nsa9/bank-customer-churn/Xtrain.csv"
Xtest_path = "hf://datasets/nsa9/bank-customer-churn/Xtest.csv"
ytrain_path = "hf://datasets/nsa9/bank-customer-churn/ytrain.csv"  # fixed double slash
ytest_path = "hf://datasets/nsa9/bank-customer-churn/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

# =====================================================================================
# FEATURES
# =====================================================================================
numeric_features = [
    'CreditScore',
    'Age',
    'Tenure',
    'Balance',
    'NumOfProducts',
    'HasCrCard',
    'IsActiveMember',
    'EstimatedSalary'
]

categorical_features = ['Geography']

# Handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Preprocessing
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Base model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Search grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],
}

pipeline = make_pipeline(preprocessor, xgb_model)

# =====================================================================================
# TRAINING + LOGGING
# =====================================================================================
with mlflow.start_run():
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    results = grid_search.cv_results_
    
    # Log each combination as nested run
    for i in range(len(results['params'])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results['params'][i])
            mlflow.log_metric("mean_test_score", results['mean_test_score'][i])
            mlflow.log_metric("std_test_score", results['std_test_score'][i])

    # Log best params
    mlflow.log_params(grid_search.best_params_)

    best_model = grid_search.best_estimator_

    # Thresholding
    threshold = 0.45
    y_pred_train = (best_model.predict_proba(Xtrain)[:, 1] >= threshold).astype(int)
    y_pred_test = (best_model.predict_proba(Xtest)[:, 1] >= threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_recall": train_report['1']['recall'],
        "train_precision": train_report['1']['precision'],
        "train_f1": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_recall": test_report['1']['recall'],
        "test_precision": test_report['1']['precision'],
        "test_f1": test_report['1']['f1-score'],
    })

    # Save best model
    model_path = "best_churn_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log artifact
    mlflow.log_artifact(model_path, artifact_path="model")

    # =================================================================================
    # Upload model to Hugging Face Hub
    # =================================================================================
    repo_id = "nsa9/churn-model"
    repo_type = "model"

    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print("Model repo already exists.")
    except RepositoryNotFoundError:
        print("Creating new model repo...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type=repo_type,
    )

print("Training complete and model uploaded successfully.")
