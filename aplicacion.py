import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import joblib
import requests

# Cargar modelo y encoder
modelo = tf.keras.models.load_model("fashion_mobilenet.keras")
encoder = joblib.load("encoder.pkl")   # cargamos el LabelEncoder guardado en el notebook
clases = encoder.classes_              # usamos las clases directamente del encoder

st.title("Clasificador de Moda con MobileNetV2")
st.write("Sube una imagen y el modelo te dirá a qué tipo de prenda pertenece.")

# Subida de imagen
archivo = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

if archivo is not None:
    imagen = Image.open(archivo).convert("RGB")
    #st.image(imagen, caption="Imagen subida", use_column_width=True)
    st.image(imagen, caption="Imagen subida", width=400)

    # Preprocesamiento idéntico al notebook
    imagen_array = np.array(imagen)
    imagen_array = tf.image.resize(imagen_array, (224,224))   # mismo resize que en el notebook
    imagen_array = preprocess_input(imagen_array)             # normalización de MobileNetV2
    imagen_array = np.expand_dims(imagen_array, axis=0)

    # Predicción
    prediccion = modelo.predict(imagen_array)
    indice = np.argmax(prediccion)
    clase = clases[indice]
    confianza = prediccion[0][indice]

    # Diccionario de traducción
    traducciones = {
    "Tshirts": "Remera",
    "Shirts": "Camiseta",
    "Casual Shoes": "Zapatillas casuales",
    "Sports Shoes": "Zapatillas deportivas",
    "Tops": "Top",
    "Handbags": "Cartera",
    "Heels": "Zapatos",
    "Sandals":"Sandalias",
    "Shorts":"Shorts",
    "Trousers":"Pantalones",
    "Jeans":"Jeans"
}

    # Traducción antes de mostrar y enviar
    clase_traducida = traducciones.get(clase, clase)

    # Mostrar en Streamlit
    st.write(f"Clase: {clase_traducida}")
    st.write(f"**Confianza:** {confianza:.2f}")


    # Enviar al Webhook con la clase ya traducida
    data = {"clase": clase_traducida, "confianza": float(confianza)}
    requests.post("https://shanafritzler9.app.n8n.cloud/webhook/c627cb40-ac06-4bde-b7ae-48006fed1342", json=data)
