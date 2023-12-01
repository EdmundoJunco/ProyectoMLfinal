import streamlit as st
import pandas as pd
import time

st.set_page_config(page_title="ProyArturoJunco", page_icon="icono.png")


st.title("Proyecto de Deep Learning Avanzado")
st.title("TAREA 5 - FINAL")
st.header("Presentado: Ingeniero Reinel Tabares Soto")
st.header("Estudiante: Edmundo Arturo Junco Orduz")
st.subheader("Doctorado Ingeniería")

st.markdown("-----")

st.success("Exitoso \U0001F604 ")

st.markdown("----- ")
st.header("\U0001f4a1")
st.write("escribir texto")


st.checkbox("checkbox")
st.radio("Radio",("Opcion1","Opcion2","Opcion3"))
st.button("Button")
var = st.selectbox("Selectbox", ("Opcion 1", "Opcion 2","Opcion 3"))
st.multiselect("Multiselect", ("Opcion 1", "Opcion 2","Opcion 3"))
st.slider("Slider", 0, 10) 
st.select_slider("Select slider", options=("Opcion1", "Opcion2","Opcion3"))
st.text_input("Text input")
st.number_input("Number input")
st.text_area("Text area")
st.date_input("Date input")
st.time_input("Time imput")
st.markdown("mas ejemplos")

st.markdown("----- ")
st.file_uploader("File uploader")

st.success("Ud selecciono:"+var)

st.sidebar.title("Esta es la Barra lateral")
st.sidebar.radio("Radio2",("Opcion1","Opcion2","Opcion3"))

col1,col2 = st.columns(2)

with col1:
    st.write("Columna 1")
    st.write("Edmundo Arturo")

with col2:
    st.write("Columna 2")
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
    lastet_interation.text(f"Interation {i+1}")
    bar.progress(i+1)
    time.sleep(0.1)


#st.pyplot_chart(figure)

st.write("Made in Edmundo Arturo Junco Orduz")