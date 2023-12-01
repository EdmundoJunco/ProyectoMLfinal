import streamlit as st
import pandas as pd

st.title("Página \U0001F604")
st.markdown("-----")
st.markdown("<h1 style='text-align: center;'>Cargar los Datos del Equipo</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>DEAM</h1>", unsafe_allow_html=True)
st.markdown("-----")
file = st.file_uploader("Sube un archivo csv o xlsx", type=["csv","xlsx"])

if file is not None:
    df = pd.read_csv(file)
    st.write("Datos del archivo:")
    st.dataframe(df)

st.write("Made in Edmundo Arturo Junco Orduz")

