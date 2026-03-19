import pandas as pd
import numpy as np

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)


def time_aware_train_test_split(
    df: pd.DataFrame,
    target_col: str = "overspend_target_t1",
    test_size: float = 0.2
):
    """
    Split data chronologically instead of randomly.

    Parameters
    ----------
    df : pd.DataFrame
        Model-ready dataset with a month column and target.
    target_col : str
        Name of target column.
    test_size : float
        Fraction of the newest rows to use as test set.

    Returns
    -------
    X_train, X_test, y_train, y_test, train_df, test_df
    """
    if "month" not in df.columns:
        raise ValueError("DataFrame must contain a 'month' column.")

    data = df.copy()
    data["month"] = pd.to_datetime(data["month"].astype(str), errors="coerce")
    data = data.sort_values("month").reset_index(drop=True)

    split_idx = int(len(data) * (1 - test_size))
    if split_idx <= 0 or split_idx >= len(data):
        raise ValueError("test_size produced an invalid train/test split.")

    train_df = data.iloc[:split_idx].copy()
    test_df = data.iloc[split_idx:].copy()

    # Exclude obvious leakage / non-feature columns
    drop_cols = [
        target_col,
        "month",
        "overspend_current_month"   # current month label should not be used
    ]
    feature_cols = [c for c in data.columns if c not in drop_cols]

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df[target_col]
    y_test = test_df[target_col]

    return X_train, X_test, y_train, y_test, train_df, test_df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build preprocessing for numeric and categorical columns.
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ],
        remainder="drop"
    )

    return preprocessor


def get_models(random_state: int = 42) -> dict:
    """
    Returns the planned models for overspending classification.
    """
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state
        )
    }


def evaluate_classifier(model, X_test, y_test) -> dict:
    """
    Evaluate a trained classifier.
    """
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = y_pred

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score) if len(np.unique(y_test)) > 1 else np.nan,
        "avg_precision": average_precision_score(y_test, y_score)
        if len(np.unique(y_test)) > 1 else np.nan,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, zero_division=0, output_dict=True
        )
    }

    return metrics


def train_and_compare_models(
    df: pd.DataFrame,
    target_col: str = "overspend_target_t1",
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Full modeling pipeline:
    1. chronological train/test split
    2. preprocessing
    3. train multiple models
    4. compare metrics

    Returns
    -------
    results_df : pd.DataFrame
        Table of model performance.
    fitted_models : dict
        Trained sklearn pipelines.
    split_data : dict
        Train/test partitions for later analysis.
    """
    X_train, X_test, y_train, y_test, train_df, test_df = time_aware_train_test_split(
        df=df,
        target_col=target_col,
        test_size=test_size
    )

    preprocessor = build_preprocessor(X_train)
    models = get_models(random_state=random_state)

    results = []
    fitted_models = {}

    for model_name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", clone(estimator))
            ]
        )

        pipeline.fit(X_train, y_train)
        metrics = evaluate_classifier(pipeline, X_test, y_test)

        fitted_models[model_name] = pipeline
        results.append({
            "model": model_name,
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1_score"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "roc_auc": metrics["roc_auc"],
            "avg_precision": metrics["avg_precision"]
        })

    results_df = pd.DataFrame(results).sort_values(
        by=["f1_score", "roc_auc", "avg_precision"],
        ascending=False
    ).reset_index(drop=True)

    split_data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "train_df": train_df,
        "test_df": test_df
    }

    return results_df, fitted_models, split_data