import pickle
import mlflow
import pathlib
import dagshub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

# Cargar los datos
data_path = '../data/Landmines.csv'
df = pd.read_csv(data_path)

# Selección de características y variable objetivo
X = df.drop('M', axis=1)
y = df['M']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configurar experimentos de DagsHub
with dagshub.dagshub_logger(hparams_path='params.yml') as logger:

    # Definición de los modelos y el espacio de hiperparámetros
    models = {
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(),
            'space': {
                'n_estimators': scope.int(hp.quniform('n_estimators', 100, 1000, 1)),
                'learning_rate': hp.loguniform('learning_rate', -3, 0),
                'max_depth': scope.int(hp.quniform('max_depth', 3, 10, 1)),
                'subsample': hp.uniform('subsample', 0.5, 1)
            }
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'space': {
                'n_neighbors': hp.quniform('n_neighbors', 3, 15, 1),
                'weights': hp.choice('weights', ['uniform', 'distance']),
                'p': hp.choice('p', [1, 2])
            }
        }
    }

    # Tuning y evaluación de cada modelo
    for model_name, model_info in models.items():
        def objective(params):
            with mlflow.start_run(nested=True):
                mlflow.set_tag("model", model_name)
                mlflow.log_params(params)

                # Pipeline de escalado y modelo
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', model_info['model'].set_params(**params))
                ])

                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                mlflow.log_metric("accuracy", accuracy)

            return {'loss': -accuracy, 'status': STATUS_OK}

        # Ejecución de la búsqueda de hiperparámetros
        with mlflow.start_run(run_name=f"{model_name} Hyperparameter Tuning", nested=True):
            best_params = fmin(
                fn=objective,
                space=model_info['space'],
                algo=tpe.suggest,
                max_evals=10,
                trials=Trials()
            )

            # Entrenar el modelo con los mejores parámetros
            best_model = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model_info['model'].set_params(**best_params))
            ])
            best_model.fit(X_train, y_train)

            # Evaluación final del modelo
            y_pred = best_model.predict(X_test)
            metrics = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Missclassification rate': 1 - accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, average='weighted'),
                'Recall': recall_score(y_test, y_pred, average='weighted'),
                'F1-score': f1_score(y_test, y_pred, average='weighted')
            }

            # Registro de métricas
            for metric_name, metric_value in metrics.items():
                logger.log_metrics({f"{model_name} - {metric_name}": metric_value})
                mlflow.log_metric(metric_name, metric_value)

            # Guardar el modelo
            pathlib.Path("models").mkdir(exist_ok=True)
            with open(f'models/{model_name}_model.pkl', 'wb') as f:
                pickle.dump(best_model, f)