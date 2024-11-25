from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import pickle
import os

# Configuración de conexión a DagsHub y MLflow
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/Pacolaz/Proyecto-Final.mlflow"

# Configuración del modelo
MODEL_NAME = "Landmines-Champion-RandomForest"

# Crear la aplicación FastAPI
app = FastAPI()

# Intentar cargar el modelo desde DagsHub
try:
    print("Intentando cargar el modelo desde DagsHub...")
    # Especificar la versión del modelo
    model_uri = f"models:/{MODEL_NAME}/1"
    model = mlflow.sklearn.load_model(model_uri)
    print("Modelo Champion cargado exitosamente desde DagsHub.")
except Exception as e:
    raise RuntimeError(f"No se pudo cargar el modelo desde DagsHub: {e}")

# Modelo de entrada para la API
class PredictionRequest(BaseModel):
    V: float  # Característica predictora 1
    H: float  # Característica predictora 2

# Ruta principal para verificar el estado
@app.get("/")
def read_root():
    return {"status": "API is running", "model_status": "Loaded"}

# Ruta para realizar predicciones
@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Crear DataFrame con las características de entrada
        input_data = [[request.V, request.H]]

        # Realizar predicción
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data).max()

        return {
            "prediction": int(prediction[0]),
            "probability": float(probability)
        }
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la predicción: {e}")
