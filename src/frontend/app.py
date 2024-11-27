import streamlit as st
import requests

# Configuración inicial
st.set_page_config(page_title="Land Mines Detection", layout="centered")

# Configuración de la API
API_URL = "http://api:8000/predict"

# Título de la aplicación
st.title("Land Mines Detection")

# Formulario para entrada de datos
st.header("Entrada de Características")
v_value = st.number_input("Ingrese el valor del Voltaje V:", format="%.6f")
h_value = st.number_input("Ingrese la altura del sensor H:", format="%.6f")

# Botón para realizar la predicción
if st.button("Realizar Predicción"):
    if v_value is not None and h_value is not None:  # Validar que las entradas no estén vacías
        # Realizar solicitud a la API
        payload = {"V": v_value, "H": h_value}
        try:
            with st.spinner("Realizando predicción, por favor espere..."):
                response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                prediction = data.get("prediction", "No disponible")
                probability = data.get("probability", "No disponible")
                
                # Mostrar resultados
                st.success("¡Predicción realizada exitosamente!")
                st.header("Resultados de la Predicción")
                st.write(f"**Categoría Predicha:** {prediction}")
                if probability != "No disponible":
                    st.write(f"**Probabilidad:** {probability:.2f}")
                else:
                    st.write("**Probabilidad:** No disponible")
            else:
                st.error(f"Error en la API: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Error al conectar con la API: {e}")
    else:
        st.warning("Por favor, ingrese valores válidos para las características.")

# Pie de página
st.markdown("---")
st.caption("Desarrollado por: Francisco Gonzalez y Luis Lopez")
