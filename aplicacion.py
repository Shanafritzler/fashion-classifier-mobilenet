import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import joblib
import requests

#funcion para detectar color dominante 
def detectar_color(imagen):

    # Reducimos tamaño para acelerar

    imagen_chica = imagen.resize((50, 50))

    pixeles = np.array(imagen_chica)

 # Ignorar píxeles blancos
    mascara = ~(
        (pixeles[:,:,0] > 240) &
        (pixeles[:,:,1] > 240) &
        (pixeles[:,:,2] > 240)
    )

    pixeles_prenda = pixeles[mascara]

    # Promedio RGB
    r = np.mean(pixeles_prenda[:, 0])
    g = np.mean(pixeles_prenda[:, 1])
    b = np.mean(pixeles_prenda[:, 2])

    # Reglas simples
    if r > 180 and g > 180 and b > 180:
        return "Blanco"

    elif r < 70 and g < 70 and b < 70:
        return "Negro"

    elif abs(r - g) < 20 and abs(g - b) < 20:
        return "Gris"

    elif r > b and r > g:
        return "Rojo"

    elif g > r and g > b:
        return "Verde"

    elif b > r and b > g:
        return "Azul"

    return "Color no identificado"

 #funcion para recomendar colores
combinaciones = {
    "Negro": ["Blanco", "Gris"],
    "Blanco": ["Negro", "Azul"],
    "Azul": ["Gris", "Beige", "Blanco"],
    "Rojo": ["Gris", "Blanco", "Azul"],
    "Verde": ["Blanco", "Beige", "Negro"],
    "Gris": ["Negro", "Rosa", "Bordo"],
    "Rosa": ["Blanco", "Gris", "Beige"]}

colores_hex = {
    "Negro": "#000000",
    "Blanco": "#FFFFFF",
    "Gris": "#696767",
    "Azul": "#1A1ABBC1",
    "Rojo": "#C13535",
    "Verde": "#008000",
    "Beige": "#F8E1C2",
    "Bordo": "#800000",
    "Rosa": "#FFC0CB"
}

def mostrar_color(nombre_color):

    hex_color = colores_hex.get(
        nombre_color,
        "#CCCCCC"
    )

    st.markdown(
        f"""
        <div style="
            width:100px;
            height:100px;
            background:{hex_color};
            border:1px solid black;
            border-radius:10px;
            margin-bottom:5px;
        ">
        </div>

        <p>{nombre_color}</p>
        """,
        unsafe_allow_html=True
    )

def recomendar_colores(color):
    
    if color in combinaciones:
        return combinaciones[color]

    return ["Negro", "Blanco"]


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
    "Jeans":"Jeans"}

    #imagen = Image.open(archivo).convert("RGB")

    color = detectar_color(imagen)
    recomendados = recomendar_colores(color)


    # Traducción antes de mostrar y enviar
    clase_traducida = traducciones.get(clase, clase)

    # Mostrar en Streamlit
    st.write(f"Clase: {clase_traducida}")
    st.write(f"Color: {color}")
    st.write("Combina bien con:")

    columnas = st.columns(len(recomendados))

    for col, color_recomendado in zip(
        columnas,
        recomendados
    ):
         with col:
          mostrar_color(color_recomendado)


    # Enviar al Webhook con la clase ya traducida
    data = {"clase": clase_traducida, "confianza": float(confianza)}
    requests.post("https://shanafritzler9.app.n8n.cloud/webhook/c627cb40-ac06-4bde-b7ae-48006fed1342", json=data)
