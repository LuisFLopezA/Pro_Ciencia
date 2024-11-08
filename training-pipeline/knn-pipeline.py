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
from sklearn.neighbors import KNeighborsClassifier

# Configuration
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

@task(name="Tuning para KNeighborsClassifier")
def hyper_parameter_tuning_knn(X_train, X_val, y_train, y_val):
    mlflow.sklearn.autolog()

    def objective_knn(params):
        with mlflow.start_run(nested=True):
            mlflow.set_tag("model_family", "k_neighbors_classifier")

            knn_model = KNeighborsClassifier(
                n_neighbors=int(params["n_neighbors"]),
                weights=params["weights"],
                p=int(params["p"]),
            )

            knn_model.fit(X_train, y_train)

            y_pred = knn_model.predict(X_val)

            accuracy = accuracy_score(y_val, y_pred)

            mlflow.log_metric("accuracy", accuracy)

            return {"loss": -accuracy, "status": STATUS_OK}

    search_space_knn = {
        "n_neighbors": scope.int(hp.quniform("n_neighbors", 3, 15, 1)),
        "weights": hp.choice("weights", ["uniform", "distance"]),
        "p": scope.int(hp.quniform("p", 1, 5, 1)),  
    }

    with mlflow.start_run(
        run_name="Hyperparameter Optimization for KNeighborsClassifier", nested=True
    ):
        best_params = fmin(
            fn=objective_knn,
            space=search_space_knn,
            algo=tpe.suggest,
            max_evals=10,
            trials=Trials(),
        )

        best_params["n_neighbors"] = int(best_params["n_neighbors"])
        best_params["p"] = int(best_params["p"])

        mlflow.log_params(best_params)

    return best_params

@task(name="Train Best KNeighborsClassifier Model")
def train_best_model(X_train, X_val, y_train, y_val, best_params) -> None:
    with mlflow.start_run(run_name="Best KNeighborsClassifier Model"):
        mlflow.log_params(best_params)

        knn_model = KNeighborsClassifier(
            n_neighbors=best_params["n_neighbors"],
            weights=best_params["weights"],
            p=best_params["p"],
        )

        knn_model.fit(X_train, y_train)
        y_pred = knn_model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        os.makedirs("models", exist_ok=True)

        model_path = "models/knn_model.pkl"
        with open(model_path, "wb") as f_model:
            pickle.dump(knn_model, f_model)
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
            model_uri=run_uri, name="boom-model-kneighborsclassifier-perfect"
        )

        model_name = "boom-kneighborsclassifier-perfect"
        model_version_alias = "champion"

        client.set_registered_model_alias(
            name=model_name, alias=model_version_alias, version="1"
        )

    except mlflow.exceptions.RestException as e:
        print(f"Skipping model registration due to error: {e}")

@flow(name="Main Flow")
def main_flow() -> None:
    dagshub.init(url=DAGSHUB_URL, mlflow=True)
    mlflow.set_experiment(experiment_name="boom-kneighborsclassifier-prefect")
    df = pd.read_csv("data/Landmines.csv")
    X_train, X_val, y_train, y_val = add_features(df)
    best_params = hyper_parameter_tuning_knn(X_train, X_val, y_train, y_val)
    train_best_model(X_train, X_val, y_train, y_val, best_params)
    register_model()

if __name__ == "__main__":
    main_flow()