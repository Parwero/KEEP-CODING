#PRACTICA CHRISTIAN GUEVARA 2


import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


### 2. Optimización con descenso del gradiente
# 2.1 Implementar el descenso del gradiente
#
#Completa las siguientes funciones para implementar el descenso del gradiente con la función objetivo de la regresión lineal. La función necesitará:

#    La entrada X y salida y de la regresión
#    Un punto inicial desde el que empezar a iterar
#    El número de iteraciones
#    El learning rate

#La función nos devolverá un array con las w resultantes de las iteraciones y otro con el valor de la función en cada paso del algoritmo, a la que también se conoce como loss function.


# Define la función que calcule n_iter iteraciones del descenso del gradiente
def gradient_descent(X, y, w0, n_iter, eta):
    # Inicializamos variables
    # loss_iter guardará el error en cada iteración para ver cómo evoluciona
    loss_iter = [np.inf]
    # w_iter guardará los pesos en cada paso, por si queremos analizarlos después
    w_iter = [w0]
    # Comienzo con w igual a w0 
    w = w0

    # TODO 1 Añade la columna de 1s
    # Esto es para añadir el término independiente (bias)
    X_bias = np.c_[np.ones(X.shape[0]), X]

    # TODO 2 Haz un bucle para las iteraciones
    for i in range(n_iter):
        # TODO 3 Dentro del bucle tendrás que actualizar el error y los pesos y añadirlos a las listas

        # Calculo la predicción actual: y_hat = X_bias * w
        y_hat = X_bias @ w

        # Calculo el gradiente: X^T(Xw - y)
        grad = X_bias.T @ (y_hat - y)

        # Actualizo los pesos con la regla: w = w - eta * grad
        w = w - eta * grad

        # Calculo el error RSS para ver cómo evoluciona
        loss = 0.5 * np.sum((y_hat - y) ** 2)

        # Guardo el error y los pesos
        loss_iter.append(loss)
        w_iter.append(w.copy())

    # Devuelvo resultados
    return np.array(w_iter), np.array(loss_iter)

# Learning rate
eta = 0.01    
# Número de iteraciones       
iteraciones = 2000   

# Genero los pesos iniciales
# Punto inicial desde el que empezar a iterar (pesos iniciales aleatorios)
np.random.seed(123)
w0 = np.random.rand(2).reshape((2, 1))  

# Datos 
# Entrada X e y de la regresión

y = np.array([208500, 181500, 223500, 140000, 250000]).reshape((5, 1))
X = np.array([
    [0.37020659],
    [-0.48234664],
    [0.51483616],
    [0.38352774],
    [1.29888065]
])

# Llamo a la función de descenso del gradiente
weights, loss = gradient_descent(X, y, w0, iteraciones, eta)

# Resultados
print("Ultimos pesos ", weights[-1])
print("Primeros pesos :", weights[0])

# Pinto la grafica
plt.plot(loss)
plt.title('descenso del gradiente')
plt.xlabel('Iteraciones')
plt.ylabel('RSS')
plt.grid()
plt.show()

# La curva baja muy rapido y luego se aplana, esta convergiendo
# Tiene sentido que el intercepto sea enorme (~200k) porque X está normalizado y la media de y es 200.000
# La pendiente es grande porque un cambio pequeño en X provoca cambios enormes en y.
# En mi caso salió ~4k







#2.2 Aplicar al dataset de consumo de combustible
#Leemos de nuevo los datos y aplicamos la función que acabamos de programar.


import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

# Fetch dataset
auto_mpg = fetch_ucirepo(id=9)

# Data (as pandas dataframes)
X = auto_mpg.data.features
y = auto_mpg.data.targets

# Convert to numpy
X_np = X['weight'].to_numpy().reshape((X.shape[0], 1))
y_np = y.to_numpy()

# Verifico la forma para confirmar que tengo 398 filas
print(X_np.shape)
print(y_np.shape)

# Escalamos la variable X
X_gd = (X_np - X_np.mean()) / X_np.std()
# Y la variable y la dejamos igual
y_gd = y_np

# Definimos la función de descenso del gradiente
def gradient_descent(X, y, w0, n_iter, eta):
    # Inicializo la lista para almacenar el error en cada iteración
    loss_iter = [np.inf]
    # Inicializo la lista para guardar los pesos en cada iteración
    w_iter = [w0]
    # Inicializo los pesos con el valor inicial
    w = w0
    # Añado una columna de 1s para el intercepto
    X_bias = np.c_[np.ones(X.shape[0]), X]

    # Inicio el bucle de iteraciones
    for i in range(n_iter):
        # Calculo la predicción actual
        y_hat = X_bias @ w
        
        # Calculo el gradiente usando la fórmula X^T(Xw - y)
        grad = X_bias.T @ (y_hat - y)
        
        # Actualizo los pesos con la tasa de aprendizaje
        w = w - eta * grad
        
        # Calculo el error RSS
        loss = 0.5 * np.sum((y_hat - y) ** 2)
        # Guardo el error en la lista
        
        loss_iter.append(loss)
        
        # Guardo los pesos en la lista
        w_iter.append(w.copy())

    # Devuelvo las listas como arrays
    return np.array(w_iter), np.array(loss_iter)

# TODO Punto inicial y learning rate
eta = 0.0000001
n_iter = 2000
np.random.seed(0)
w0 = np.random.rand(2, 1)

# TODO Aplicamos el algoritmo
weights, loss = gradient_descent(X_gd, y_gd, w0, n_iter, eta)

# Imprimimos los pesos iniciales y finales
print("Pesos iniciales:", weights[0])
print("Últimos pesos:", weights[-1])

# Pintamos la gráfica de la evolución del error
plt.figure(figsize=(8, 4))
plt.plot(loss)
plt.title('Evolución del error con descenso del gradiente')
plt.xlabel('Iteraciones')
plt.ylabel('RSS')
plt.grid()
plt.show()

# Calculamos las predicciones
X_bias = np.c_[np.ones(X_gd.shape[0]), X_gd]
y_pred = X_bias @ weights[-1]

# Pintamos la recta con los datos
plt.figure(figsize=(8, 5))
plt.scatter(X_np, y_gd, color='blue', alpha=0.4, label='Datos')
plt.plot(X_np, y_pred, color='red', linewidth=2, label='Recta GD')
plt.xlabel('Peso del coche')
plt.ylabel('Consumo (MPG)')
plt.title('Recta ajustada con descenso del gradiente')
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# La curva del error baja y luego se aplana, lo que indica convergencia
# El intercepto es da lo esperado
# La pendiente es negativa, significa que a mayor peso, menor consumo
# Usé un learning rate muy pequeño porque y no está normalizado
# La recta roja sigue la tendencia de los puntos, funciona el modelo
