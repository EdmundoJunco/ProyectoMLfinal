import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import time
from numpy.lib.shape_base import column_stack
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder


st.markdown("-----")

st.markdown("<h1 style='text-align: center;'>Análisis hidro-climático Tunja Boyacá</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>Análisis Exploratorio de Datos</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>DEAM</h1>", unsafe_allow_html=True)
st.markdown("-----")
imagen_url = "http://entreojos.co/img/intro/mapa-intro.png"
st.image(imagen_url, caption='El panorama del cambio climático en Boyacá', use_column_width=True)
#imagen_local = "ruta/a/tu/imagen.jpg"
#st.image(imagen_local, caption='Descripción de la imagen', use_column_width=True)


lastet_interation = st.empty()
bar = st.progress(0)
for i in range(100):
    lastet_interation.text(f"Cargar los Datos Hidroclimaticos {i+1}")
    bar.progress(i+1)
    #time.sleep(0.10)

st.markdown("-----")
data = pd.read_csv("models\data\excel.csv.csv")
st.dataframe(data)
st.dataframe(data.describe())
st.markdown("-----")

st.markdown("<h1 style='text-align: center;'>Análisis Componente Principales</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>PCA</h1>", unsafe_allow_html=True)
lastet_interation = st.empty()
bar = st.progress(0)
for i in range(100):
    lastet_interation.text(f"Generar PCA {i+1}")
    bar.progress(i+1)
    #time.sleep(0.10)
# Generar datos de ejemplo
np.random.seed(42)
data = np.random.rand(100, 2) * 10  # Datos bidimensionales de ejemplo

# Escalar los datos usando StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Crear una instancia de PCA
pca = PCA(n_components=2)

# Ajustar el modelo PCA y transformar los datos escalados
data_pca = pca.fit_transform(data_scaled)

# Obtener la varianza acumulada
explained_variance_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)

# Crear una gráfica de varianza acumulada
fig, ax = plt.subplots()
ax.plot(np.arange(1, len(explained_variance_ratio_cumsum) + 1), explained_variance_ratio_cumsum, marker='o')
ax.set_xlabel('Número de Componentes Principales')
ax.set_ylabel('Varianza Acumulada')
ax.set_title('Gráfica de Varianza Acumulada')

# Mostrar la gráfica en Streamlit
st.pyplot(fig)

# Crear un conjunto de datos
np.random.seed(19)
data = pd.DataFrame(np.random.randint(10,size=(100,3)),columns=['Columna 1','Columna 2','Columna 3'])
data['Columna 1']= data['Columna 1'] * 10
data['Columna 2']= data['Columna 2'] * 100
data['Columna 3']= data['Columna 3'] * 1000
data

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler_df = scaler.fit_transform(data)

scaler_df

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_df = scaler.fit_transform(data)
scaled_df = pd.DataFrame(scaled_df,columns=['Columna 1', 'Columna 2', 'Columna 3'])
scaled_df

sns.set(style="whitegrid")
sns.distplot(data['Columna 1'], kde=False, bins=20)

data = pd.DataFrame({'Columna 1': np.random.randn(100)})
sns.distplot(data['Columna 1'])
st.pyplot()

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
