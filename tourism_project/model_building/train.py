"""
train.py
--------
Model building step of the MLOps pipeline (production environment).

1. Load the train/test splits produced by the data-prep job (workflow artifact).
2. Build a preprocessing + XGBoost pipeline (scaling for numeric features,
   one-hot encoding for categorical features, class weighting for imbalance).
3. Tune the hyper-parameters with GridSearchCV (stratified 5-fold CV, F1 score).
4. Track every parameter combination in MLflow as a nested run, plus the best
   parameters and final metrics in the parent run.
5. Choose the decision threshold that maximises F1 on out-of-fold predictions
   of the training set (the test set is never used for any decision).
6. Evaluate on train and test, then save the best model and its configuration
   into tourism_project/deployment/ so the workflow can commit them to the repo.
"""
import json
import os

import joblib
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.compose import make_column_transformer
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "tourism-package-prediction"
DEPLOY_DIR = "tourism_project/deployment"
MODEL_FILE = "best_tourism_model_v1.joblib"
CONFIG_FILE = "model_config.json"
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",
]
CATEGORICAL_FEATURES = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]

# Hyper-parameter grid (2^5 = 32 combinations x 5 folds = 160 fits)
PARAM_GRID = {
    "xgbclassifier__n_estimators": [150, 300],      # number of boosting trees
    "xgbclassifier__max_depth": [4, 6],             # tree depth (model complexity)
    "xgbclassifier__learning_rate": [0.05, 0.1],    # shrinkage per boosting step
    "xgbclassifier__colsample_bytree": [0.7, 1.0],  # feature sampling per tree
    "xgbclassifier__reg_lambda": [1, 5],            # L2 regularisation
}


def compute_metrics(y_true, y_prob, threshold, prefix):
    """Return a dict of classification metrics for a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}_recall": recall_score(y_true, y_pred),
        f"{prefix}_f1": f1_score(y_true, y_pred),
        f"{prefix}_roc_auc": roc_auc_score(y_true, y_prob),
    }


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 1. Load the splits passed by the previous job -------------------------
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").squeeze()
    ytest = pd.read_csv("ytest.csv").squeeze()
    print(f"Train: {Xtrain.shape} | Test: {Xtest.shape}")

    # 2. Pipeline -------------------------------------------------------------
    # scale_pos_weight = (#negatives / #positives) compensates the imbalance
    class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

    preprocessor = make_column_transformer(
        (StandardScaler(), NUMERIC_FEATURES),
        (OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    )
    xgb_model = xgb.XGBClassifier(
        scale_pos_weight=class_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model_pipeline = make_pipeline(preprocessor, xgb_model)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    with mlflow.start_run(run_name="xgb-gridsearch") as parent_run:
        mlflow.set_tags({"stage": "production-training", "model_family": "XGBoost"})
        mlflow.log_params({
            "scale_pos_weight": round(float(class_weight), 4),
            "cv_folds": 5,
            "train_rows": Xtrain.shape[0],
            "test_rows": Xtest.shape[0],
        })

        # 3. Hyper-parameter tuning --------------------------------------------
        grid_search = GridSearchCV(
            model_pipeline,
            PARAM_GRID,
            scoring={"f1": "f1", "recall": "recall", "precision": "precision", "roc_auc": "roc_auc"},
            refit="f1",
            cv=cv,
            n_jobs=-1,
        )
        grid_search.fit(Xtrain, ytrain)

        # 4. Log every tuned combination as a nested run ---------------------
        res = grid_search.cv_results_
        for i, params in enumerate(res["params"]):
            with mlflow.start_run(run_name=f"combo_{i:02d}", nested=True):
                mlflow.log_params(params)
                mlflow.log_metrics({
                    "cv_f1": res["mean_test_f1"][i],
                    "cv_f1_std": res["std_test_f1"][i],
                    "cv_recall": res["mean_test_recall"][i],
                    "cv_precision": res["mean_test_precision"][i],
                    "cv_roc_auc": res["mean_test_roc_auc"][i],
                })

        best_model = grid_search.best_estimator_
        mlflow.log_params({f"best_{k.split('__')[1]}": v for k, v in grid_search.best_params_.items()})
        mlflow.log_metric("best_cv_f1", grid_search.best_score_)
        print("Best parameters:", grid_search.best_params_)
        print(f"Best CV F1: {grid_search.best_score_:.4f}")

        # 5. Decision threshold from out-of-fold training predictions --------
        oof_prob = cross_val_predict(
            clone(best_model), Xtrain, ytrain, cv=cv, method="predict_proba"
        )[:, 1]
        thresholds = np.round(np.arange(0.20, 0.81, 0.01), 2)
        f1s = [f1_score(ytrain, (oof_prob >= t).astype(int)) for t in thresholds]
        best_threshold = float(thresholds[int(np.argmax(f1s))])
        mlflow.log_param("decision_threshold", best_threshold)
        print(f"Selected decision threshold (max OOF F1): {best_threshold:.2f}")

        # 6. Evaluate ------------------------------------------------------------
        train_prob = best_model.predict_proba(Xtrain)[:, 1]
        test_prob = best_model.predict_proba(Xtest)[:, 1]
        metrics = {
            **compute_metrics(ytrain, train_prob, best_threshold, "train"),
            **compute_metrics(ytest, test_prob, best_threshold, "test"),
        }
        mlflow.log_metrics(metrics)

        print("\nTrain classification report:")
        print(classification_report(ytrain, (train_prob >= best_threshold).astype(int)))
        print("Test classification report:")
        print(classification_report(ytest, (test_prob >= best_threshold).astype(int)))

        # 7. Save the model + config for deployment -------------------------
        os.makedirs(DEPLOY_DIR, exist_ok=True)
        model_path = os.path.join(DEPLOY_DIR, MODEL_FILE)
        config_path = os.path.join(DEPLOY_DIR, CONFIG_FILE)
        joblib.dump(best_model, model_path)

        config = {
            "model_file": MODEL_FILE,
            "decision_threshold": best_threshold,
            "best_params": {k.split("__")[1]: v for k, v in grid_search.best_params_.items()},
            "test_metrics": {k: round(float(v), 4) for k, v in metrics.items() if k.startswith("test_")},
            "mlflow_run_id": parent_run.info.run_id,
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        mlflow.log_artifact(model_path, artifact_path="model")
        mlflow.log_artifact(config_path, artifact_path="model")
        print(f"\nModel saved to {model_path}")
        print(f"Config saved to {config_path}")
        print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
