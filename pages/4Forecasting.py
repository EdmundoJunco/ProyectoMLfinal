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

#with open('') as archivo:

# Generar datos de ejemplo
X, y = [[1], [2], [3], [4]], [2, 4, 6, 8]

# Dividir datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Entrenar un modelo de regresión lineal
model1 = LinearRegression()
model1.fit(X_train, y_train)

# Evaluar el modelo
predictions1 = model1.predict(X_test)
mse1 = mean_squared_error(y_test, predictions1)
print(f'Modelo 1 - MSE: {mse1}')

# Guardar el modelo en un archivo usando pickle
with open('model1.pkl', 'wb') as model_file:
    pickle.dump(model1, model_file)

# Entrenar otro modelo, por ejemplo, un RandomForestRegressor
model2 = RandomForestRegressor()
model2.fit(X_train, y_train)

# Evaluar el segundo modelo
predictions2 = model2.predict(X_test)
mse2 = mean_squared_error(y_test, predictions2)
print(f'Modelo 2 - MSE: {mse2}')

# Guardar el segundo modelo en un archivo usando pickle
with open('model2.pkl', 'wb') as model_file:
    pickle.dump(model2, model_file)

# Cargar modelos desde los archivos
loaded_models = []

with open('model1.pkl', 'rb') as model_file:
    loaded_model1 = pickle.load(model_file)
    loaded_models.append(loaded_model1)

with open('model2.pkl', 'rb') as model_file:
    loaded_model2 = pickle.load(model_file)
    loaded_models.append(loaded_model2)

# Realizar predicciones con los modelos cargados
for idx, loaded_model in enumerate(loaded_models):
    loaded_predictions = loaded_model.predict(X_test)
    loaded_mse = mean_squared_error(y_test, loaded_predictions)
    print(f'Modelo {idx + 1} cargado - MSE: {loaded_mse}')

st.write("Made in Edmundo Arturo Junco Orduz")