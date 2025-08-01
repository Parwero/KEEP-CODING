#PRACTICA CHRISTIAN GUEVARA


import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


### 1.1 Carga del dataset y exploración de datos
# Sigo los pasos que se ha marcado la practica


# fetch dataset 
auto_mpg = fetch_ucirepo(id=9) 
  
# data (as pandas dataframes) 
X = auto_mpg.data.features 
y = auto_mpg.data.targets 
  
# metadata 
print(auto_mpg.metadata) 
  
# variable information 
print(auto_mpg.variables) 


# Ver primeras filas
print(X.head())
print(y.head())

# En este punto, puedes hacer cualquier otro análisis exploratorio que creas conveniente.
# Aqui puedo comparar si hay valores nulos en el dataset
print(X.isnull().sum())
print(y.isnull().sum())
# Solo horsepower tiene 6 nulos deberiamos eliminarlos

# Historigrama para ver la distribuicion
X['weight'].hist(bins=30)
# Parece una distribucion casi normal

# Ver variables medias, minimos y maximos
print(X.describe())
print(y.describe())
#la mayoría de autos pesa entre 2200 y 3600 libras. El más pesado pesa 5140 libras.
#la mayoría está entre 17.5 y 29 mpg. 





#A continuación, crea una función a la que le pases los dataframes de features y de target y el nombre de la variable y haga la gráfica del target
# (consumo en millas por galón) vs la variable que le ha llegado por parámetro. Puedes ver un ejemplo de gráfica más abajo, pero usa los parámetros que
# quieras en cuanto a color de los puntos, grids etc.


def visualiza(df_features, df_target, nombre_columna):
    # Calculo la correlacion
    correlacion = df_features[nombre_columna].corr(df_target.squeeze())

    # Pinto la grafica
    plt.figure(figsize=(8,5))
    plt.scatter(df_features[nombre_columna], df_target, color='red', alpha=0.6)
    plt.title(f'{nombre_columna} vs MPG (corr: {correlacion:.2f})')
    plt.xlabel(nombre_columna)
    plt.ylabel('MPG')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
    
# Llamo a la funcion con las variables
visualiza(X, y, 'horsepower')
# En el caso de housepower tiene un corr ≈ -0.78 por lo que diria que la relación entre horsepower y MPG es fuerte y negativa.
visualiza(X, y, 'acceleration')
# En el caso de acceleration tiene una corr ≈ 0.42 la relación entre acceleration y MPG es moderada y positiva.
visualiza(X, y, 'weight')
# En el caso de weight tiene una corr ≈ -0.83 La relación entre weight y MPG es fuerte y negativa.
# Al ser -0.83 creo que es la mejor eleccion para este modelo



# Hacemos la regresion lineal con weight
# Cojo la columna weight y la variable mpg
X_var = X[['weight']].values 
y_var = y.values  

# Ahora le meto una columna de 1s, porque si no el modelo no tendría el intercepto (el punto donde corta la recta)
# Esto básicamente es para que el modelo pueda tener la constante b0.
X_bias = np.c_[np.ones(X_var.shape[0]), X_var]

# Hacemos mínimos cuadrados: (X^T X)^(-1) X^T y
XtX = X_bias.T @ X_bias
XtX_inv = np.linalg.inv(XtX)
Xty = X_bias.T @ y_var
w_hat = XtX_inv @ Xty
print("Pesos calculados (primero intercepto y luego pendiente):", w_hat)
# El primer número es la constante, el segundo es la pendiente de la recta.

# Calculo las predicciones
y_pred = X_bias @ w_hat

# Hago el error promedio para ver si esta bien
rmse = np.sqrt(np.mean((y_var - y_pred)**2))
print("RMSE del modelo:", rmse)

# Hago la grafica
plt.figure(figsize=(8,5))
plt.scatter(X_var, y_var, color='blue', alpha=0.5, label='Datos')
plt.plot(X_var, y_pred, color='red', label='Recta ajustada')
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.title('Regresión Lineal: MPG vs Weight')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# La pendiente es negativa, si el coche pesa mas, gasta mas
# Viendo la gráfica, los puntos no están perfectamente alineados con la recta, lo que significa que hay variabilidad



# 1.2 Implementación del modelo
# En este punto vamos a dar un paso atrás y olvidarnos de los datos de consumo por un momento, porque el algoritmo que vamos a implementar serviría para cualquier conjunto de datos.
# Lo primero, implementa una función que devuelva los valores de wˆ dados X e y. Si se da el caso de que el dataset tiene más features que observaciones, queremos transponerlo antes de realizar las operaciones. Lo hacemos porque, como vimos en clase, los resultados no van a ser buenos si el número de observaciones es menor.

def pesos_ols(X, y):
    # Implementa la fórmula matricial

    # TODO 1: Comprueba que las dimensiones son adecuadas y corrige si no es así
    # Primero convierto X e y a arrays numpy para trabajar más fácil
    X = np.array(X)
    y = np.array(y)

    # Si hay más columnas que filas, transpongo para que cada fila sea un ejemplo.
    if X.shape[0] < X.shape[1]:
        print("Hay que trasponer")
        X = X.T

    # TODO 2: Prepara la matriz X para que tenga la forma adecuada para aplicar la fórmula
    # Aquí añado una columna de 1s para el término independiente (intercepto).
    # Sin esto, el modelo no podría aprender el punto donde la recta corta el eje.
    X_bias = np.c_[np.ones(X.shape[0]), X]

    # TODO 3: Calcula los pesos
    # Aplico la fórmula de mínimos cuadrados: (X^T X)^(-1) X^T y
    # Esta fórmula encuentra los coeficientes que minimizan el error cuadrático medio.
    XtX = X_bias.T @ X_bias  # Multiplico X transpuesta por X
    XtX_inv = np.linalg.inv(XtX)  
    Xty = X_bias.T @ y  
    w = XtX_inv @ Xty  

    return w

# Datos de prueba: 2 variables (probablemente tamaño y año) y 10 observaciones
X_test = np.array([[1710, 1262, 1786, 1717, 2198, 1362, 1694, 2090, 1774, 1077],
                   [2003, 1976, 2001, 1915, 2000, 1993, 2004, 1973, 1931, 1939]])
y_test = np.array([208500, 181500, 223500, 140000, 250000, 143000, 307000, 200000, 129900, 118000])

# - y_test parece ser el precio de una casa 
# - X_test tiene dos filas con valores tipo 1700 y 2000: podrían ser área (sq ft) y año.
# Si la relación es lógica, cuanto más área o más reciente el año, más alto el precio.

pesos = pesos_ols(X_test, y_test)
print("Pesos calculados:", pesos)
# Los pesos dan mas o menos lo mismo que el array, asi que la damos como correcta
# El intercepto (primer número) es negativo, algo normal porque es un punto de ajuste.
# La segunda cifra (alrededor de 59) significa que por cada unidad que aumenta la primera variable, el precio sube 59.
# La tercera (alrededor de 1207) indica que la segunda variable (probablemente el año) influye mucho más.




#1.3 Prueba del modelo en los datos de consumo de coches
#Ya podemos usar nuestra función pesos en el dataset real que hemos descargado, el de consumo de combustible. Para ver el ajuste, dibuja de nuevo los datos junto con la recta que has obtenido, en la misma gráfica. Tendrás que convertir los dataframes de Pandas en arrays de numpy para poder aplicar las operaciones matriciales.
#TIP: Revisa las dimensiones de todos los arrays para ver que son correctas.
# TODO Aplica el modelo y dibuja la recta junto con los datos
# Lo que voy a hacer ahora es usar los pesos que calculé y dibujar la línea para ver si todo tiene sentido.
# Si la línea se ajusta bien, genial. Si no, algo raro pasa.

# Selecciono solo la variable weight y la salida mpg (como dice la práctica)
X_var = X[['weight']].values  
y_var = y.values 

# Calculo los pesos 
w_hat = pesos_ols(X_var, y_var)
print("Pesos obtenidos con mi modelo:", w_hat)
# Espero algo como [46.2, -0.0076]
# Hago las predicciones con esos pesos
X_bias = np.c_[np.ones(X_var.shape[0]), X_var] 
# Aplico la fórmula para sacar y estimado
y_pred = X_bias @ w_hat  

# Dibujo la gráfica
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.scatter(X_var, y_var, color='blue', alpha=0.5, label='Datos')  
# Línea roja es mi modelo
plt.xlabel('Weight')
plt.plot(X_var, y_pred, color='red', linewidth=2, label='')  
plt.ylabel('MPG')
plt.title('Mi modelo OLS vs Datos')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


# La línea roja debería bajar porque (más peso = menos mpg). Si no baja, algo está mal.
# No espero que la línea pase por todos los puntos porque solo usamos una variable.
# Si la tendencia general está bien, entonces el modelo funciona.

# Ahora comparo con scikit-learn para ver si coinciden los resultados

# Creo el modelo de sklearn
lr = LinearRegression()
X_lr = X[['weight']]
y_lr = y

# Entreno con los datos
lr.fit(X_lr, y_lr)
print("Punto de corte (intercepto):", lr.intercept_)
print("Pendiente:", lr.coef_)
# Aquí debería salir algo muy parecido a lo mío


# Sklearn y mi resultado son casi iguales, osea que esta bien
# También me doy cuenta de que solo usar weight no explica todo, pero capta bien la tendencia.




# ### 1.4 Compara el método de mínimos cuadrados y el descenso del gradiente

# Defino la función de descenso del gradiente



def gradient_descent(X, y, w0, n_iter, eta):
    # Aquí aplico la fórmula: w^(t+1) = w^t - eta * gradiente
    # Para guardar la pérdida en cada iteración
    loss_iter = []  
    # Para guardar los pesos en cada paso
    # Uso copy() porque si no, todas las posiciones en la lista de pesos apuntarían al mismo objeto y acabaría viendo el mismo valor en todas
    w_iter = [w0.copy()]  
    w = w0.copy()  

    # Añadimos la columna de 1s para el intercepto
    X_bias = np.c_[np.ones(X.shape[0]), X]

    for i in range(n_iter):
        # Calculo la predicción
        y_hat = X_bias @ w

        # Calculo el gradiente
        grad = X_bias.T @ (y_hat - y)

        # Actualizo los pesos
        w = w -  eta * grad

        # (RSS)
        loss = 0.5 * np.sum((y_hat - y) ** 2)

        # Guardo los valores
        loss_iter.append(loss)
        w_iter.append(w.copy())

    return np.array(w_iter), np.array(loss_iter)

# Preparo los datos, sino me daban NAN
X_gd = X[['weight']].values
y_gd = y.values.reshape(-1, 1)

# Escalo X porque el descenso del gradiente va más rápido si normalizo los datos
X_scaled = (X_gd - X_gd.mean()) / X_gd.std()

# Punto inicial y parámetros
np.random.seed(0)
# Dos pesos: intercepto y pendiente
w0 = np.random.rand(2,1)
# Lo he puesto en 0.001 para que no de error al hacer el proceso python
eta = 0.001  
n_iter = 2000

# Ejecuto el descenso del gradiente
w_history, loss_history = gradient_descent(X_scaled, y_gd, w0, n_iter, eta)

print("Ultimos pesos:", w_history[-1])
# Lo importante: intercepto positivo, pendiente negativa → más peso = menos MPG.

# Ahora dibujo cómo disminuye la función de pérdida
import matplotlib.pyplot as plt

plt.plot(loss_history)
plt.title('Evolucion del error con descenso del gradiente')
plt.xlabel('Iteraciones')
plt.ylabel('RSS')
plt.grid()
plt.show()

# Comparo con OLS y sklearn
print("Mi modelo OLS -> Intercepto:", w_hat[0], "Pendiente:", w_hat[1])
print("Modelo sklearn -> Intercepto:", lr.intercept_, "Pendiente:", lr.coef_)
# Mi modelo OLS -> Intercepto: ~46.2165, Pendiente: ~-0.007647
# Modelo sklearn -> Intercepto: ~46.2165, Pendiente: ~-0.007647
# Son muy parecidos, por lo que la formula de minimos esta bien 
