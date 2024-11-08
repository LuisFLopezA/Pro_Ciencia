import os
import pickle

import dagshub
import mlflow
import pandas as pd
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from mlflow import MlflowClient
from prefect import flow, task
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

DAGSHUB_URL = "https://dagshub.com/luislopez3105/Pro_Ciencia"
CSV_PATH = "data/Landmines.csv"

@task(name="Read Data")
def read_data():
    df = pd.read_csv(CSV_PATH)
    return df

@task(name="Add features")
def add_features(df: pd.DataFrame):
    X = df.drop(columns=["M"])
    y = df["M"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_val, y_train, y_val

# Hyperparameteros tuning para XGBoost
@task(name="Tuning para XGBClassifier")
def hyper_parameter_tuning_xgb(X_train, X_val, y_train, y_val):
    mlflow.sklearn.autolog()

    def objective_xgb(params):
        with mlflow.start_run(nested=True):
            mlflow.set_tag("model_family", "xgb_classifier")

            xgb_model = XGBClassifier(
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]),
                learning_rate=params["learning_rate"],
                gamma=params["gamma"],
                min_child_weight=int(params["min_child_weight"]),
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                random_state=42,
            )

            xgb_model.fit(X_train, y_train)

            y_pred = xgb_model.predict(X_val)

            accuracy = accuracy_score(y_val, y_pred)

            mlflow.log_metric("accuracy", accuracy)

            return {"loss": -accuracy, "status": STATUS_OK}

    search_space_xgb = {
        "n_estimators": scope.int(hp.quniform("n_estimators", 50, 200, 1)),
        "max_depth": scope.int(hp.quniform("max_depth", 3, 10, 1)),
        "learning_rate": hp.loguniform("learning_rate", -3, 0),
        "gamma": hp.uniform("gamma", 0, 1),
        "min_child_weight": scope.int(hp.quniform("min_child_weight", 1, 10, 1)),
        "subsample": hp.uniform("subsample", 0.5, 1),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
    }

    with mlflow.start_run(
        run_name="Hyperparameter Optimization for XGBClassifier", nested=True
    ):
        best_params = fmin(
            fn=objective_xgb,
            space=search_space_xgb,
            algo=tpe.suggest,
            max_evals=10,
            trials=Trials(),
        )

        best_params["n_estimators"] = int(best_params["n_estimators"])
        best_params["max_depth"] = int(best_params["max_depth"])
        best_params["min_child_weight"] = int(best_params["min_child_weight"])

        mlflow.log_params(best_params)

    return best_params

@task(name="Train Best XGBClassifier Model")
def train_best_model(X_train, X_val, y_train, y_val, best_params) -> None:
    with mlflow.start_run(run_name="Best XGBClassifier Model"):
        mlflow.log_params(best_params)

        xgb_model = XGBClassifier(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            learning_rate=best_params["learning_rate"],
            gamma=best_params["gamma"],
            min_child_weight=best_params["min_child_weight"],
            subsample=best_params["subsample"],
            colsample_bytree=best_params["colsample_bytree"],
            random_state=42,
        )

        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        os.makedirs("models", exist_ok=True)
        model_path = "models/xgb_model.pkl"
        with open(model_path, "wb") as f_model:
            pickle.dump(xgb_model, f_model)
        mlflow.log_artifact(model_path, artifact_path="model")

    return None

@task(name="Register Model")
def register_model():
    MLFLOW_TRACKING_URI = mlflow.get_tracking_uri()
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    df = mlflow.search_runs(order_by=["-metrics.accuracy"])

    try:
        run_id = df.loc[df["metrics.accuracy"].idxmax()]["run_id"]
        run_uri = f"runs:/{run_id}/model"

        result = mlflow.register_model(
            model_uri=run_uri, name="boom-model-xgbclassifier-perfect"
        )

        model_name = "boom-xgbclassifier-perfect"
        model_version_alias = "champion"

        client.set_registered_model_alias(
            name=model_name, alias=model_version_alias, version="1"
        )

    except mlflow.exceptions.RestException as e:
        print(f"Skipping model registration due to error: {e}")

@flow(name="Main Flow")
def main_flow() -> None:
    dagshub.init(url=DAGSHUB_URL, mlflow=True)
    mlflow.set_experiment(experiment_name="boom-xgbclassifier-prefect")
    df = pd.read_csv("data/Landmines.csv")
    X_train, X_val, y_train, y_val = add_features(df)
    best_params = hyper_parameter_tuning_xgb(X_train, X_val, y_train, y_val)
    train_best_model(X_train, X_val, y_train, y_val, best_params)
    register_model()

if __name__ == "__main__":
    main_flow()