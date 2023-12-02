import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


st.title("")

st.markdown("<h1 style='text-align: center;'>Forecasting for Social Good</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>Variables Hidro-climáticas</h1>", unsafe_allow_html=True)

#with open('') as archivo:
st.markdown("-----")
data = pd.read_csv("models\data\TUNJA.csv")
st.dataframe(data)
st.dataframe(data.describe())
st.markdown("-----")
# Cargar modelos desde archivos pickle
loaded_models = []

with open('model1.pkl', 'rb') as model_file:
    loaded_model1 = pickle.load(model_file)
    loaded_models.append(loaded_model1)

with open('model2.pkl', 'rb') as model_file:
    loaded_model2 = pickle.load(model_file)
    loaded_models.append(loaded_model2)

# Página principal de la aplicación Streamlit
def main():
    st.title('Streamlit Model Showcase')

    # Sidebar para seleccionar el modelo
    model_selection = st.sidebar.selectbox('Seleccionar Modelo', ['Modelo 1', 'Modelo 2'])

    # Obtener el modelo seleccionado
    selected_model = loaded_models[0] if model_selection == 'Modelo 1' else loaded_models[1]

    # Interfaz para ingresar datos y realizar predicciones
    st.header('Hacer Predicciones')

    # Ingresar datos para predicciones
    input_data = st.text_input('Ingresar datos para predicciones (separados por comas):')

    if st.button('Realizar Predicción'):
        try:
            input_array = np.array([float(x.strip()) for x in input_data.split(',')])
            prediction = selected_model.predict(input_array.reshape(1, -1))
            st.success(f'La predicción es: {prediction[0]}')
        except ValueError:
            st.error('Error al procesar los datos. Asegúrate de ingresar números separados por comas.')

# Ejecutar la aplicación Streamlit
if __name__ == '__main__':
    main()

st.write("Made in Edmundo Arturo Junco Orduz")