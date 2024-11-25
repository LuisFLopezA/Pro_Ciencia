from prefect import flow, task
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import pandas as pd
from sklearn.model_selection import train_test_split
from dagshub import init
from mlflow.tracking import MlflowClient
import pickle

# Inicializar DagsHub y configurar MLflow
init(repo_owner="Pacolaz", repo_name="Proyecto-Final", mlflow=True)

# Variables globales
best_rf_model = None
best_rf_accuracy = 0
best_lr_model = None
best_lr_accuracy = 0

# Función para cargar los datos
def load_data(file_path):
    df = pd.read_csv(file_path)
    X = df[['V', 'H']]
    y = df['M']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Espacio de búsqueda para Random Forest
search_space_rf = {
    'n_estimators': hp.quniform('n_estimators', 50, 300, 1),
    'max_depth': hp.quniform('max_depth', 5, 30, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 5, 1),
}

# Función objetivo para Random Forest
def objective_rf(params):
    global best_rf_model, best_rf_accuracy
    rf_model = RandomForestClassifier(
        n_estimators=int(params['n_estimators']),
        max_depth=int(params['max_depth']),
        min_samples_split=int(params['min_samples_split']),
        min_samples_leaf=int(params['min_samples_leaf']),
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    if accuracy > best_rf_accuracy:
        best_rf_accuracy = accuracy
        best_rf_model = rf_model

    return {'loss': -accuracy, 'status': STATUS_OK}

# Espacio de búsqueda para Logistic Regression
search_space_lr = {
    'C': hp.loguniform('C', -4, 2)
}

# Función objetivo para Logistic Regression
def objective_lr(params):
    global best_lr_model, best_lr_accuracy
    lr_model = LogisticRegression(
        C=params['C'], 
        solver='liblinear', 
        random_state=42
    )
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    if accuracy > best_lr_accuracy:
        best_lr_accuracy = accuracy
        best_lr_model = lr_model

    return {'loss': -accuracy, 'status': STATUS_OK}

@task
def register_models():
    client = MlflowClient()

    if best_rf_accuracy > best_lr_accuracy:
        champion_model, challenger_model = best_rf_model, best_lr_model
        champion_name = "Landmines-Champion-RandomForest"
        challenger_name = "Landmines-Challenger-LogisticRegression"
    else:
        champion_model, challenger_model = best_lr_model, best_rf_model
        champion_name = "Landmines-Champion-LogisticRegression"
        challenger_name = "Landmines-Challenger-RandomForest"

    # Registrar Champion
    with mlflow.start_run(run_name="Champion-Model"):
        mlflow.log_metric("accuracy", max(best_rf_accuracy, best_lr_accuracy))
        mlflow.sklearn.log_model(champion_model, "model")
        registered_champion = mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/model",
            champion_name
        )
        print(f"Modelo Champion registrado como: {registered_champion.name}")

    # Registrar Challenger
    with mlflow.start_run(run_name="Challenger-Model"):
        mlflow.log_metric("accuracy", min(best_rf_accuracy, best_lr_accuracy))
        mlflow.sklearn.log_model(challenger_model, "model")
        registered_challenger = mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/model",
            challenger_name
        )
        print(f"Modelo Challenger registrado como: {registered_challenger.name}")

@flow(name="Entrenamiento y Optimización de Modelos - Landmines")
def main_flow(file_path):
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = load_data(file_path)

    # Entrenamiento y registro de Random Forest
    with mlflow.start_run(run_name="RandomForest"):
        fmin(fn=objective_rf, space=search_space_rf, algo=tpe.suggest, max_evals=10, trials=Trials())

    # Entrenamiento y registro de Logistic Regression
    with mlflow.start_run(run_name="LogisticRegression"):
        fmin(fn=objective_lr, space=search_space_lr, algo=tpe.suggest, max_evals=10, trials=Trials())

    # Registrar los mejores modelos
    register_models()

# Ejecución
if __name__ == "__main__":
    file_path = "data/Landmines.csv"
    main_flow(file_path)
