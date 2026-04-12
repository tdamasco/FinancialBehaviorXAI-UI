import pandas as pd
import numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

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
    if "period" not in df.columns:
        raise ValueError("DataFrame must contain a 'period' column.")

    data = df.copy()
    data["period"] = pd.to_datetime(data["period"].astype(str), errors="coerce")
    data = data.sort_values("period").reset_index(drop=True)

    split_idx = int(len(data) * (1 - test_size))
    if split_idx <= 0 or split_idx >= len(data):
        raise ValueError("test_size produced an invalid train/test split.")

    train_df = data.iloc[:split_idx].copy()
    test_df = data.iloc[split_idx:].copy()

    drop_cols = [
        target_col,
        "period",
        "overspend_current_month",
        "first",
        "last",
        "year",
        "month",
    ]
    feature_cols = [c for c in data.columns if c not in drop_cols]

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df[target_col]
    y_test = test_df[target_col]

    return X_train, X_test, y_train, y_test, train_df, test_df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
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
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
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


def get_models(random_state: int = 42, class_weight: dict = {0: 1, 1: 2}) -> dict:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight=class_weight,
            random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=2,
            class_weight=class_weight,
            random_state=random_state
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state
        )
    }


def evaluate_classifier(model, X_test, y_test, threshold: float = 0.5) -> dict:
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
        y_pred = (y_score >= threshold).astype(int)
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
        y_pred = (y_score >= threshold).astype(int)
    else:
        y_score = model.predict(X_test)
        y_pred = y_score

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score) if len(np.unique(y_test)) > 1 else np.nan,
        "avg_precision": average_precision_score(y_test, y_score) if len(np.unique(y_test)) > 1 else np.nan,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, zero_division=0, output_dict=True
        )
    }

    return metrics


def tune_model(model_name: str, X_train, y_train, preprocessor, n_trials: int = 50, random_state: int = 42) -> dict:
    tscv = TimeSeriesSplit(n_splits=3)

    def get_pipeline_for_trial(params):
        if model_name == "logistic_regression":
            estimator = LogisticRegression(
                C=params["C"],
                class_weight={0: 1, 1: params["class_weight_1"]},
                max_iter=1000,
                random_state=random_state
            )
        elif model_name == "random_forest":
            estimator = RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_leaf=params["min_samples_leaf"],
                class_weight={0: 1, 1: params["class_weight_1"]},
                random_state=random_state
            )
        elif model_name == "gradient_boosting":
            estimator = GradientBoostingClassifier(
                n_estimators=params["n_estimators"],
                learning_rate=params["learning_rate"],
                max_depth=params["max_depth"],
                random_state=random_state
            )
        return Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])

    def objective(trial):
        if model_name == "logistic_regression":
            params = {
                "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
                "class_weight_1": trial.suggest_float("class_weight_1", 1.0, 4.0)
            }
        elif model_name == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 400),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "class_weight_1": trial.suggest_float("class_weight_1", 1.0, 4.0)
            }
        elif model_name == "gradient_boosting":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 5)
            }

        pipeline = get_pipeline_for_trial(params)
        scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=tscv, scoring="f1", error_score=0.0
        )
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def train_and_compare_models(
    df: pd.DataFrame,
    target_col: str = "overspend_target_t1",
    test_size: float = 0.2,
    random_state: int = 42,
    class_weight: dict = {0: 1, 1: 2},
    threshold: float = 0.5,
    tune: bool = False,
    n_trials: int = 50
):
    X_train, X_test, y_train, y_test, train_df, test_df = time_aware_train_test_split(
        df=df, target_col=target_col, test_size=test_size
    )

    preprocessor = build_preprocessor(X_train)

    if tune:
        print("Running hyperparameter tuning...")
        tuned_estimators = {}
        for name in ["logistic_regression", "random_forest", "gradient_boosting"]:
            print(f"  Tuning {name}...")
            best_params = tune_model(name, X_train, y_train, preprocessor, n_trials=n_trials)
            print(f"  Best params: {best_params}")

            cw_1 = best_params.pop("class_weight_1", 2.0)
            cw = {0: 1, 1: cw_1}

            if name == "logistic_regression":
                tuned_estimators[name] = LogisticRegression(
                    C=best_params.get("C", 1.0),
                    class_weight=cw, max_iter=1000, random_state=random_state
                )
            elif name == "random_forest":
                tuned_estimators[name] = RandomForestClassifier(
                    n_estimators=best_params.get("n_estimators", 300),
                    max_depth=best_params.get("max_depth", 6),
                    min_samples_leaf=best_params.get("min_samples_leaf", 2),
                    class_weight=cw, random_state=random_state
                )
            elif name == "gradient_boosting":
                tuned_estimators[name] = GradientBoostingClassifier(
                    n_estimators=best_params.get("n_estimators", 200),
                    learning_rate=best_params.get("learning_rate", 0.05),
                    max_depth=best_params.get("max_depth", 3),
                    random_state=random_state
                )
        models = tuned_estimators
    else:
        models = get_models(random_state=random_state, class_weight=class_weight)

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
        metrics = evaluate_classifier(pipeline, X_test, y_test, threshold=threshold)

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
        by=["f1_score", "roc_auc", "avg_precision"], ascending=False
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