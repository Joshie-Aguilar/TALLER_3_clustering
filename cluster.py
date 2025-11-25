## Clustering  cancer de mama - taller 3 IA en salud 

import pandas as pd  
import kagglehub
import os # para manejar rutas de archivos 

# - descargar y cargar datos 
print ("Descargando datos ...")
# Este comando descarga la carpeta con los datos y nos da la ruta
path_carpeta= kagglehub.dataset_download ("uciml/breast-cancer-wisconsin-data")
print ("Ruta de descarga:",path_carpeta)

# Buscamos el archivo .csv dentro de esta carpeta 
ruta_archivo_csv= os.path.join(path_carpeta,"data.csv")

# leemos el archivo CSV y lo convertimos en un DataFrame
df= pd.read_csv(ruta_archivo_csv)

# Miremos las primeras 5 filas para entender qué tenemos 
print("\n-- Primeras filas del dataser ---")
print(df.head())

# Miremos información general (columnas, tipos de datos, valores nulos)
print("\n-- Información general del dataset ---")
print(df.info())  

## Limpieza de datos 

# eliminar las columnas que no sirven para el análisis numérico 
# "id", "Unnamed: 32"
df_limpio = df.drop(["id"], axis=1)
if "Unnamed: 32" in df_limpio.columns:
    df_limpio = df_limpio.drop(["Unnamed: 32"], axis=1)

# Separar la columna de diagnóstico 
diagnostico_real = df_limpio["diagnosis"]

# creamos nuestros datos de entrenamiento (x) eliminando la columna de diagnóstico 
X= df_limpio.drop(["diagnosis"], axis=1)

print("\nDatos listos para procesar (X):", X.shape)

# Escalar los datos (Estandarización)
# sino el algoritmo solo le prestara atención a los datos grandes 

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_escalado = scaler.fit_transform(X)

# Método K-means
# para encontrar el número óptimo de clusters (K) - usamos el método del codo

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

## método del codo para hallar K óptimo 
inercia=[] # lista para guardar los resultados 
rango_k =range(1,11) # probaremos de 1 a 10

print("\nCalculando el método del codo...")
for k in rango_k:
    # Creamos el modelo KMeans
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init='auto')
    # Entrenamos con los datos escalados
    kmeans_temp.fit(X_escalado)
    # Guardamos la "inercia" (qué tan dispersos están los puntos dentro del cluster)
    inercia.append(kmeans_temp.inertia_)

# Graficar el codo
plt.figure(figsize=(8, 4))
plt.plot (rango_k,inercia,"bx-")
plt.xlabel("Cantidad de Clusters (K)")
plt.ylabel("Inercia (dispersión)")
plt.title("Método del Codo para encontrar K óptimo")
plt.show()

 # la infleccion del codo se da en 2 (tiene sentido porque los grupos son: benigno y maligno)

# como tenemos 30 columnas (dimesiones) . no podemos gráficar en 30D. usaremo el PCA (análisis de componentes principales)
# paara comprimir esas 30 dimensiones en solo 2 para poder hacer un gráfico de puntos 

kmeans_final = KMeans (n_clusters=2, random_state=42, n_init='auto')

# Ahora el modelo agrupa los datos 
etiquetas_kmeans= kmeans_final.fit_predict(X_escalado)

# Visualizacioón usando PCA para reducir a 2D
from sklearn.decomposition import PCA
import seaborn as sns

# Reducirmos las 30 columnas a solo 2 componentes principales para graficar 
pca=PCA(n_components=2)
X_pca=pca.fit_transform(X_escalado)

# Creamos un data frame 
df_pca=pd.DataFrame(data=X_pca,columns=["Componente 1","Componente 2"])

#agregamos las etiquetas que encontro K means 
df_pca["Cluster_KMeans"]=etiquetas_kmeans

# Graficamos 
plt.figure(figsize=(8,6))
sns.scatterplot(data=df_pca,x="Componente 1",y="Componente 2",hue="Cluster_KMeans",palette="viridis")
plt.title("Resultados de Cluustering K-means (visto en 2D con PCA)")
plt.show()

# Evaluacion de desempeño (comparación con la realidad)
# comparemos los grupos que encontro k- means (0 y 1) con el diagnostico real (M y B )
print("n\Comparación K-Means vs Diagnóstico Real--")

# usamos pandas crosstab para hacer una tabla cruzada
tabla_comparativa=pd.crosstab(diagnostico_real,etiquetas_kmeans, rownames=["Diagnostico Real"],colnames=["Cluster K-Means"])
print(tabla_comparativa)


# Clustering Jerárquico 
from scipy.cluster.hierarchy import dendrogram, linkage

# 'linkage' calcula las distancias entre todos los puntos para ver quién se une con quién.
# Usamos el método 'ward' que suele funcionar bien.
linked = linkage(X_escalado, method='ward')

# Graficar el dendrograma
plt.figure(figsize=(10, 5))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title("Dendrograma Jerárquico")
plt.xlabel("Índice de la muestra")
plt.ylabel("Distancia (Ward)")
# Cortamos el árbol visualmente para ver dónde se separan los grupos grandes.
# Una línea horizontal alrededor de distancia 40 o 50 suele mostrar 2 grandes ramas.
plt.axhline(y=40, color='r', linestyle='--')
plt.show()

from sklearn.cluster import AgglomerativeClustering

# A. Ejecutar el modelo jerárquico pidiendo 2 clusters
cluster_jerarquico = AgglomerativeClustering(n_clusters=2, linkage='ward')
etiquetas_jerarquico = cluster_jerarquico.fit_predict(X_escalado)

# B. Visualización (Usamos los mismos datos PCA de antes)
df_pca['Cluster_Jerarquico'] = etiquetas_jerarquico

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_pca, x='Componente 1', y='Componente 2', hue='Cluster_Jerarquico', palette='magma')
plt.title('Resultados de Clustering Jerárquico (Visto en 2D con PCA)')
plt.show()

# C. Evaluación del Desempeño
print("\n--- Comparación Jerárquico vs Diagnóstico Real ---")
tabla_comparativa_jer = pd.crosstab(diagnostico_real, etiquetas_jerarquico, rownames=['Diagnóstico Real'], colnames=['Cluster Jerárquico'])
print(tabla_comparativa_jer)