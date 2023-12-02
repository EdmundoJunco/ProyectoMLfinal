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

st.markdown("-----")
st.markdown("<h1 style='text-align: center;'>Métricas Datos</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>DEAM</h1>", unsafe_allow_html=True)
st.markdown("-----")


st.markdown("-----")
st.markdown("<h1 style='text-align: center;'>Arquitectura Datos</h1>", unsafe_allow_html=True)
st.markdown("-----")

data = pd.read_csv("models\data\excel.csv.csv")
st.dataframe(data)

# Obtener las features and labels
features = data.drop(['NombreEstacion'], axis=1)

labels=data.NombreEstacion

features.head(), labels.head(), features.shape, labels.shape

#Particionar en conjunto entrenamiento y pruebas
X_train, X_test, y_train, y_test = train_test_split(
                                        features,
                                        labels,
                                        train_size   = 0.6,
                                        random_state=1, stratify=labels)
X_train.shape, X_test.shape, y_train.shape, y_test.shape




st.write("Made in Edmundo Arturo Junco Orduz")