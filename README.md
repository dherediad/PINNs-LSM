# Aplicación de Redes Neuronales Informadas por Física (PINNs) en el Método de Conjuntos de Nivel

_Proyecto de Titulación / Maestría en Ciencia de Datos USFQ_

**Autor:** Diego Heredia

## Resumen

En el presente trabajo se implementan redes neuronales informadas por física (PINNs) para la resolución de ecuaciones diferenciales obtenidas de la aplicación del método de conjuntos de nivel en problemas de evolución de interfaces. Estos problemas son la rotación del círculo, rectángulo y disco de Zalesak a través de un punto en el plano y la deformación de un círculo en un vórtice. Se realizan comparaciones con distintas configuraciones de los parámetros de las PINNs variando la cantidad de capas, neuronas, función de activación de cada neurona, puntos de colocación y número de iteraciones de entrenamiento. Para estas comparaciones se utilizan las métricas $L_1$, IoU y porcentaje de pérdida de área. Se concluyó que para estos problemas, la resolución mediante PINNs con redes neuronales completamente conectadas de 5 capas y 200 neuronas, donde se utilizan 10000 puntos de colocación, generalmente obtienen métricas robustas después de 500000 iteraciones de entrenamiento; además, la función de activación sigmoide tiene un comportamiento más estable que la función tangente hiperbólica. Esta configuración puede servir como punto de partida para investigaciones futuras sobre problemas del método de conjuntos de nivel.

La ecuación del conjunto de nivel que se resuelve con PINNs es:

$$
\begin{equation}
    \frac{\partial \phi}{\partial t} + \overrightarrow{v} \cdot \nabla \phi = 0
\end{equation}
$$

Específicamente, la ecuación para la rotación es:

$$
\begin{equation}
    \frac{\partial\phi}{\partial t} - 2\pi(y-0.5) \frac{\partial \phi}{\partial x} + 2\pi(x-0.5) \frac{\partial \phi}{\partial y} = 0
\end{equation}
$$

y la ecuación para la deformación de una interfaz en un vórtice es:

$$
\begin{equation}
    \frac{\partial\phi}{\partial t} = - 8 sin^2(\pi x) sin(2\pi y) cos(\pi t)\frac{\partial \phi}{\partial x} + 8sin(2\pi x) sin^2(\pi y) cos(\pi t)\frac{\partial \phi}{\partial y}
\end{equation}
$$

- **Rotación del Círculo:** El mejor modelo para este problema, que obtuvo el mayor valor de IoU y menor valor de $L_1$ fue el de $5$ capas con $200$ neuronas cada una, con función de activación sigmoide utilizando $100\times100=10000$ puntos de colocación y con $1700000$ iteraciones de entrenamiento. El tiempo de ejecución del entrenamiento fue $4$ horas y $31$ minutos.
<p align="center">
<img src="https://github.com/user-attachments/assets/79aa6f36-6582-4c10-a61c-e3847e970604" width="60%">
</p>

- **Rotación del Rectángulo:** El mejor modelo para este problema, que obtuvo el mayor valor de IoU y menor valor de $L_1$ fue el de $5$ capas con $200$ neuronas cada una, con función de activación sigmoide utilizando $100\times100=10000$ puntos de colocación y con $1900000$ iteraciones de entrenamiento. El tiempo de ejecución del entrenamiento fue $5$ horas y $4$ minutos.
<p align="center">
<img src="https://github.com/user-attachments/assets/86b9ad74-513a-4554-92ec-8980f7205fc8" width="60%">
</p>

- **Rotación del Disco de Zalesak:** El mejor modelo para este problema, que obtuvo el mayor valor de IoU y menor valor de $L_1$ fue el de $5$ capas con $200$ neuronas cada una, con función de activación tangente hiperbólica utilizando $80\times80=6400$ puntos de colocación y con $1750000$ iteraciones de entrenamiento. El tiempo de ejecución del entrenamiento fue $3$ horas y $44$ minutos.
<p align="center">
<img src="https://github.com/user-attachments/assets/e922bcc2-b976-47ad-a781-4feefd01d2d0" width="60%">
</p>

- **Deformación del Círculo en Vórtice:** El mejor modelo para este problema, que obtuvo el mayor valor de IoU y menor valor de $L_1$ fue el de $5$ capas con $200$ neuronas cada una, con función de activación sigmoide utilizando $80\times80=6400$ puntos de colocación y con $2000000$ iteraciones de entrenamiento. El tiempo de ejecución del entrenamiento fue $4$ horas y $59$ minutos.
<p align="center">
<img src="https://github.com/user-attachments/assets/91e7e99b-9c05-469c-ac0b-109e09e444d2" width="60%">
</p>

## Contenido del Repositorio

El repositorio consta de:

-	Carpeta de cada problema (circulo, rectangulo, zalesak y vortex )
-	graficos.py: Contiene las funciones para graficar los resultados.
-	resultados.ipynb: Cuaderno de Jupyter con la ejecución de los gráficos de los resultados de las métricas $L_1$, IoU y porcentaje de pérdida de área a través de las iteraciones.
-	resultados_evolucion_comparacion_area.ipynb: Cuaderno de Jupyter con la ejecución de los gráficos de los resultados sobre evolución de la interfaz, comparación de interfaces teóricas y predichas y evolución del porcentaje de pérdida de área en el tiempo, para los mejores modelos de cada problema.
Dentro de cada carpeta (circulo, rectangulo, zalesak y vortex) se encuentra:
-	8 carpetas para cada configuración de la PINN. En cada carpeta se encuentra un archivo .py para la ejecución de los resultados, un archivo .csv con los resultados de las métricas, un archivo .txt con los tiempos de entrenamiento en segundos y cada 500000 iteraciones, una carpeta con el .gif de la evolución de la interfaz en la iteración 2000000 y en aquella que corresponda al mejor modelo y una carpeta con todos los modelos entrenados.
-	8 archivos .py con el código para el entrenamiento de la PINN

**Nota:** Cada carpeta tiene el mismo nombre que el archivo .py del entrenamiento del modelo.

## Instrucciones de Ejecución

Se presenta un ejemplo de entrenamiento de una configuración de la PINN específica:

Para entrenar una PINN para el problema de la rotación del círculo con 100x100 puntos de colocación, cuya red neuronal consta de 5 capas y 200 neuronas; y con función de activación sigmoide en cada neurona, se abre la carpeta “circulo” y se ejecuta el archivo:

<p align="center">
r_circulo_100x100_5c_200n_sigmoid.py
</p>

Este archivo genera una carpeta de nombre “r_circulo_100x100_5c_200n_sigmoid” en el mismo directorio. En esta carpeta se generan los archivos tiempo_entrenamiento.txt y una carpeta llamada “modelos” con los modelos de PyTorch cada 50000 iteraciones. Con estos archivos se puede luego inferir los resultados. Un ejemplo de estos archivos es “r_circulo_100x100_5c_200n_sigmoid_500000it”, el cual representa el modelo en 500000 iteraciones de entrenamiento.

Dentro de “r_circulo_100x100_5c_200n_sigmoid” se ejecuta el archivo:

<p align="center">
r_circulo_100x100_5c_200n_sigmoid_resultados.py
</p>

lo que genera en el mismo directorio el archivo “resultados.csv” con una tabla donde se encuentran todos los resultados de las métricas $L_1$, IoU y porcentaje de pérdida de área. Además, la ejecución genera un conjunto de carpetas con las imágenes de la evolución de la interfaz, la función de nivel y las curvas de nivel desde $t=0$ hasta $t=1$, junto con un .gif de esta evolución. En este repositorio solo se incluyeron los gifs de la iteración 2000000 y aquella correspondiente al mejor modelo por cada problema. 

De esta forma se ejecuta cada configuración de cada problema.

Finalmente, una vez que se ejecutaron todos los entrenamientos y todos los resultados dentro de sus carpetas específicas (32 modelos en total), se ejecutan los archivos resultados.ipynb y resultados_evolucion_comparacion_area.ipynb, pues estos utilizan toda la información de los .csv para obtener los mejores modelos y graficar los resultados.


