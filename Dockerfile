# Usa una imagen base de Python
FROM python:3.9-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia los archivos del proyecto al contenedor
COPY . /app

# Instala las dependencias necesarias
RUN pip install --no-cache-dir fastapi uvicorn pandas mlflow pydantic

# Expone el puerto que usará la API
EXPOSE 8000

# Comando para iniciar el servidor de la API
CMD ["uvicorn", "Api:app", "--host", "0.0.0.0", "--port", "8000"]
