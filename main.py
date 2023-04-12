import pandas as pd
from sklearn.linear_model import LinearRegression

# Cargar el archivo csv con los datos de Transmilenio
datos_transmilenio = pd.read_csv("transmilenio.csv")

# Seleccionar las columnas correspondientes a las características y la variable objetivo
X = datos_transmilenio[["hora_del_dia", "capacidad_del_bus",
                        "temperatura", "velocidad_promedio"]]
y = datos_transmilenio["numero_de_pasajeros"]

# Convertir la columna "dia_de_la_semana" en variables binarias para que pueda ser utilizada en el modelo de regresión
X = pd.get_dummies(X)

# Crear el modelo de regresión lineal
regresion_lineal = LinearRegression()

# Entrenar el modelo utilizando los datos de Transmilenio
regresion_lineal.fit(X, y)

# Crear un dataframe con las características correspondientes a la hora de interés
hora_del_dia = 5
capacidad_del_bus = 80
temperatura = 30
velocidad_promedio = 15

datos_prediccion = pd.DataFrame({"hora_del_dia": [hora_del_dia], "capacidad_del_bus": [capacidad_del_bus],
                                 "temperatura": [temperatura], "velocidad_promedio": [velocidad_promedio]})

# Utilizar el modelo entrenado para hacer la predicción
y_prediccion = regresion_lineal.predict(datos_prediccion)

print("La cantidad de pasajeros que se espera en la hora indicada es de: ",
      round(y_prediccion[0], 2))
