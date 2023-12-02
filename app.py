import streamlit as st
import pandas as pd
import time
import tkinter as tk
from PIL import Image




st.set_page_config(page_title="ProyArturoJunco", page_icon="icono.png")


st.title("Proyecto de Deep Learning Avanzado")
st.title("TAREA 5 - FINAL")
st.header("Presentado: Ingeniero Reinel Tabares Soto")
st.header("Estudiante: Edmundo Arturo Junco Orduz")
st.subheader("Doctorado Ingeniería")

st.markdown("-----")
st.markdown("<h1 style='text-align: center;'>\U0001f4a1", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>Jupiter: Strategic planning model as a mediator in Forecasting for social good on Hydro-climatic variables, With Machine learning based on the Social Graphs model.</h1>", unsafe_allow_html=True)
st.success("Exitoso \U0001F604 ")

st.markdown("----- ")
# Ruta de la imagen, ajusta esto a la ubicación de tu imagen
ruta_imagen = "img\enomeno1.jpg"

# Cargar la imagen con Pillow
imagen_pil = Image.open(ruta_imagen)

# Mostrar la imagen en Streamlit
st.image(imagen_pil, caption='Visualización de la Imagen', use_column_width=True)
st.markdown("----- ")


st.write("Bienvenidos al Sistema Climatico de Prediciones (Jupiter)")
st.write("Estas decuardo")
st.checkbox("SI")
st.checkbox("NO")
st.radio("Cambio Climatico",("Consultar","Predecir","Notificar"))
st.button("ACEPTAR")
var = st.selectbox("Tipo de Fenomeno", ("Clase 1", "Clase 2","Clase 3","Clase 4","Clase 5","Clase 6"))
st.multiselect("Selecionen los Eventos", ("lluvia", "Insendio","Terremoto","Maremoto","Tormetas eléctricas"))
st.slider("Califica Nivel de panico del Clima (0 a 10)  donde 0 es normal y 10 es panico total.", 0, 10) 
st.select_slider("Selecciona la Zona", options=("Urbana", "Rural","Mar"))
st.text_input("Escribir el Comportamiento Climatico")
st.number_input("Nivel de grabedad")
st.text_area("Comentarios del Comportamiento del Clima")
st.date_input("Fecha del suceso")
st.time_input("Hora del suceso")
st.markdown("----- ")
st.markdown("Información Adicional")
st.markdown("----- ")
st.file_uploader("Cargar Datos en CVS")

st.success("Ud selecciono Clase:"+var)


st.sidebar.title("Fenomenos Cambio Climatico")
st.sidebar.radio("Evento",("Niña","Niño","Terremoto","Tormeta"))
st.sidebar.radio("Predición",("Niña","Niño","Terremoto","Tormeta"))
col1,col2 = st.columns(2)

with col1:
    st.write("Nombres")
    st.write("Edmundo Arturo")

with col2:
    st.write("Apellidos")
    st.write("Junco Orduz")    


data = {"data":[1,2,3,4,5,6,7,8,9,10]}
df = pd.DataFrame(data)
st.dataframe(df)


st.line_chart(df)
st.area_chart(df)
st.bar_chart(df)

lastet_interation = st.empty()
bar = st.progress(0)

for i in range(100):
    lastet_interation.text(f"Cargado sus Datos {i+1}")
    bar.progress(i+1)
    time.sleep(0.1)
st.write("Muchas Gracias por su Información")

#st.pyplot_chart(figure)

st.write("Made in Edmundo Arturo Junco Orduz")