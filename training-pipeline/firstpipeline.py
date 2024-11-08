import pickle
import mlflow
import pathlib
import dagshub
import pandas as pd
import xgboost as xgb
import mlflow.sklearn
from mlflow import MlflowClient
from hyperopt.pyll import scope
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_extraction import DictVectorizer
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from prefect import flow, task
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np
import mlflow
import os
from sklearn.linear_model import LogisticRegression
import pickle


## df

# Configura la URL del experimento en DagsHub
DAGSHUB_URL = "https://dagshub.com/luislopez3105/Pro_Ciencia"
CSV_PATH = "data/Landmines.csv"

# Define the data reading task

    

# Definir función para agregar características
@task(name="Add features")
def add_features(df: pd.DataFrame):


    X = df.drop(columns=["M"])
    y = df["M"]
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

# Hyperparameteros tuning para Logistic Regression
@task(name="Tuning para Logistic Regression")
def hyper_parameter_tuning_lr(X_train, X_val, y_train, y_val):
    mlflow.sklearn.autolog()

    
    def objective_lr(params):
        with mlflow.start_run(nested=True):
            mlflow.set_tag("model_family", "logistic_regression")

            lr_model = LogisticRegression(
                C=params['C'],
                max_iter=int(params['max_iter']),
                solver=params['solver'],
                random_state=42
            )

            lr_model.fit(X_train, y_train)

            y_pred = lr_model.predict(X_val)

            accuracy = accuracy_score(y_val, y_pred)

            mlflow.log_metric("accuracy", accuracy)

            return {'loss': -accuracy, 'status': STATUS_OK}

    search_space_lr = {
        'C': hp.loguniform('C', -4, 2),  
        'max_iter': scope.int(hp.quniform('max_iter', 50, 300, 1)),
        'solver': hp.choice('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
    }

    with mlflow.start_run(run_name="Hyperparameter Optimization for Logistic Regression", nested=True):
        best_params = fmin(
            fn=objective_lr,
            space=search_space_lr,
            algo=tpe.suggest,
            max_evals=10,
            trials=Trials()
        )

        best_params["C"] = float(best_params["C"])
        best_params["max_iter"] = int(best_params["max_iter"])
        best_params["solver"] = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'][best_params["solver"]]
        mlflow.log_params(best_params)

    return best_params






# Función para entrenar el mejor modelo




@task(name="Train Best Logistic Regression Model")
def train_best_model(X_train, X_val, y_train, y_val, best_params) -> None:
    with mlflow.start_run(run_name="Best Logistic Regression Model"):
       
        mlflow.log_params(best_params)
        lr_model = LogisticRegression(
            C=best_params['C'],
            max_iter=int(best_params['max_iter']),
            solver=best_params['solver'],
            random_state=42
        )
        lr_model.fit(X_train, y_train)
        y_pred = lr_model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        os.makedirs("models", exist_ok=True) 

        with open("models/logistic_regression_model.pkl", "wb") as f_model:
            pickle.dump(lr_model, f_model)
        mlflow.log_artifact("models/logistic_regression_model.pkl", artifact_path="model")

    return None



@task(name='Register Model')
def register_model():
    MLFLOW_TRACKING_URI = mlflow.get_tracking_uri()
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    df = mlflow.search_runs(order_by=['-metrics.accuracy'])  
    
    try:
        run_id = df.loc[df['metrics.accuracy'].idxmax()]['run_id']
        run_uri = f"runs:/{run_id}/model"

        result = mlflow.register_model(
            model_uri=run_uri,
            name="boom-model-logisticregression-perfect"
        )

        model_name = "boom-logisticregression-perfect"
        model_version_alias = "champion"

        client.set_registered_model_alias(
            name=model_name,
            alias=model_version_alias,
            version='1'
        )

    except mlflow.exceptions.RestException as e:
        print(f"Skipping model registration due to error: {e}")




# Definir el flujo principal con Prefect


@flow(name="Main Flow")
def main_flow() -> None:
    dagshub.init(url=DAGSHUB_URL, mlflow=True)
    mlflow.set_experiment(experiment_name="boom-logisticregression-prefect")
    df = pd.read_csv("data/Landmines.csv")
    X_train, X_val, y_train, y_val = add_features(df)
    best_params = hyper_parameter_tuning_lr(X_train, X_val, y_train, y_val)  
    train_best_model(X_train, X_val, y_train, y_val, best_params)
    register_model()

main_flow()
