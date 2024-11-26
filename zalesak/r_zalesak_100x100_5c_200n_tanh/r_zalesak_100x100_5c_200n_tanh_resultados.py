import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import pandas as pd
from scipy.integrate import simpson
from torch.autograd import Variable
import imageio

# Nombre del modelo:
nombre = 'r_zalesak_100x100_5c_200n_tanh'

# Importar funciones para graficos
nombre_carpeta_anterior = 'experimentos'
import sys
sys.path.append(nombre_carpeta_anterior)
from graficos import grafico_general, datosAX

# Carpetas para imagenes y gifs
nombre_carpeta_principal = f'{nombre_carpeta_anterior}/zalesak/{nombre}/'

## Predicciones
# Configuracion de la red neuronal
num_neuronas = 200
problema = 'zalesak'

# Clase de la red neuronal
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_layer1 = nn.Linear(3, num_neuronas)
        self.hidden_layer2 = nn.Linear(num_neuronas, num_neuronas)
        self.hidden_layer3 = nn.Linear(num_neuronas, num_neuronas)
        self.hidden_layer4 = nn.Linear(num_neuronas, num_neuronas)
        self.hidden_layer5 = nn.Linear(num_neuronas, num_neuronas)
        self.output_layer = nn.Linear(num_neuronas, 1)
    def forward(self, x,y,t):
        inputs = torch.cat([x,y,t],axis=1)
        layer1_out = torch.tanh(self.hidden_layer1(inputs))
        layer2_out = torch.tanh(self.hidden_layer2(layer1_out))
        layer3_out = torch.tanh(self.hidden_layer3(layer2_out))
        layer4_out = torch.tanh(self.hidden_layer4(layer3_out))
        layer5_out = torch.tanh(self.hidden_layer5(layer4_out))
        output = self.output_layer(layer5_out)
        return output

# Constantes 
# Malla para predicciones de metricas (fija para todos los casos - comparacion)
num_puntos = 2000
# Malla para graficos
num_puntos_gif = 200

# Predecir modelo en CPU
device = torch.device('cpu')
# Numero de nucleos del cpu a usar (local)
torch.set_num_threads(4)

# DataFrame para resultados
resultados = pd.DataFrame(columns=['num_iter', 'area', 'area_loss(%)', 'error_L1', 'orden_L1', 'IoU', 'orden_IoU'])

# Bucle para obtener resultados de todos los modelos guardados

# Area teorica - Disco de Zalesak = area del circulo - area del rectangulo dentro - area del segmento circular sobrante
# Area del circulo
a_circulo = np.pi * 0.15**2
# Area del rectangulo = lado_izq_der * lado sup
lado_izq_der = 0.8 - (0.75 - np.sqrt(0.15**2-0.05**2))
lado_sup = 0.1
a_rectangulo = lado_izq_der * lado_sup
# Area del segmento circular
angulo_dz = np.arccos((0.15**2+0.15**2-0.1**2)/(2*0.15**2))
a_segmento_circ = (0.15**2/2) * (angulo_dz - np.sin(angulo_dz))
# Area del Disco de Zalesak
area_teorica = a_circulo - a_rectangulo - a_segmento_circ


# Perimetro teorico - Disco de Zalesak = perimetro del circulo + (lado izq + lado sup + lado der - arco inf) [del disco de Zalesak]
# Perimetro del circulo
per_circulo = 2*np.pi*0.15
# lado izq = lado der = coord_y_punto_superior - coord_y_corte_recta_circulo
lado_izq_der = 0.8 - (0.75 - np.sqrt(0.15**2-0.05**2))
# lado sup = 0.55-0.45 = 0.1
lado_sup = 0.1
# arco inferior = radio * angulo del arco en radianes (np.arccos da angulo en radianes)
arco_inf =  0.15 * np.arccos((0.15**2+0.15**2-0.1**2)/(2*0.15**2))
# Perimetro Disco de Zalesak
perimetro_teorico = per_circulo + 2*lado_izq_der + lado_sup - arco_inf


# Phi inicial (SDF) - para calculo de IoU
x_ic = np.linspace(0,1,num_puntos)
y_ic = np.linspace(0,1,num_puntos)
# Grid
x_ic_grid, y_ic_grid = np.meshgrid(x_ic, y_ic)
## Ajuste a vector
x_ic_grid = np.ravel(x_ic_grid).reshape(-1,1)
y_ic_grid = np.ravel(y_ic_grid).reshape(-1,1)
# Circulo SDF
u_ic_circulo = np.sqrt((x_ic_grid-0.5)**2+(y_ic_grid-0.75)**2) - 0.15
# Recatgnulo SDF
rectangulo_x1 = 0.45
rectangulo_y1 = 0.50
rectangulo_x2 = 0.55
rectangulo_y2 = 0.80
u_ic_rectangulo = []
for i in range(x_ic_grid.shape[0]):
  a = max(rectangulo_x1-x_ic_grid[i], x_ic_grid[i]-rectangulo_x2)
  b = max(rectangulo_y1-y_ic_grid[i], y_ic_grid[i]-rectangulo_y2)
  distancia = min(0, max(a,b)) + np.sqrt(max(0,a)**2 + max(0,b)**2)
  u_ic_rectangulo.append(distancia)
u_ic_rectangulo = np.vstack(u_ic_rectangulo)
# Disco de Zalesak SDF: circulo - rectangulo
u_ic = [max(u_ic_circulo[i],-u_ic_rectangulo[i]) for i in range(x_ic_grid.shape[0])]
u_ic = np.vstack(u_ic)


# Métrica Interection Over Union (IoU) - argumentos son funciones phi con reshape(-1,1) [no bool todavia]
def calculate_iou(matrix1, matrix2):
    # Convertir las matrices a booleanas - menor a 0 es el area dentro de la interfaz
    bool_matrix1 = (matrix1 < 0)
    bool_matrix2 = (matrix2 < 0)
    # Calcular la intersección y la unión
    intersection = np.logical_and(bool_matrix1, bool_matrix2)
    union = np.logical_or(bool_matrix1, bool_matrix2)
    # Contar el número de celdas verdaderas en la intersección y la unión
    intersection_sum = np.sum(intersection)
    union_sum = np.sum(union)
    # Calcular IoU
    iou = intersection_sum / union_sum
    iou = round(iou, 8)
    return iou


iterations = 2000000
for epoch in range(1,iterations+1):
    if epoch % 50000 == 0:
        modelo = Net()
        modelo.load_state_dict(torch.load(f'{nombre_carpeta_principal}modelos/{nombre}_{epoch}it.pth', map_location=device))
        modelo.eval()
        # phi predicho              
        _,_,phi_pred,_ = datosAX(tiempo=1, num_puntos=num_puntos, net=modelo)
        # Metrica IoU
        valor_iou = calculate_iou(u_ic, phi_pred.ravel().reshape(-1,1))
        # Perdida de area
        phi_pred = phi_pred.reshape(num_puntos, num_puntos)
        phi_pred = (phi_pred < 0) #phi_pred es matriz booleana a partir de aqui
        area_predicha = simpson(simpson(phi_pred, x=np.linspace(0.0,1,num_puntos), dx=1/num_puntos, axis=-1),
                                x=np.linspace(0.0,1,num_puntos), dx=1/num_puntos, axis=-1)
        porcentaje_perdida = round((area_teorica-area_predicha)*100/area_teorica, 8)      
        # Error L1
        u_ic_l1 = u_ic.reshape(num_puntos, num_puntos) #integrales necesitan matrices nxn
        u_ic_l1 = (u_ic_l1 < 0)
        l1_mat = np.abs(u_ic_l1.astype(np.float32) - phi_pred.astype(np.float32))
        area_l1_mat = simpson(simpson(l1_mat, x=np.linspace(0.0,1,num_puntos), dx=1/num_puntos, axis=-1),
                              x=np.linspace(0.0,1,num_puntos), dx=1/num_puntos, axis=-1)
        valor_l1 = (1/perimetro_teorico)*(area_l1_mat)          
        # Resultados - orden se calcula al final
        res_dict = {'num_iter':epoch, 'area':round(area_predicha,8), 'area_loss(%)':porcentaje_perdida, 
                    'error_L1':round(valor_l1,8), 'orden_L1':0, 'IoU':valor_iou, 'orden_IoU':0}
        resultados = pd.concat([resultados if not resultados.empty else None,
                                pd.DataFrame([res_dict])], 
                                ignore_index=True)

# Orden - el primer valor es N/A porque no existe i-1
resultados['orden_L1'] = resultados['orden_L1'].astype(np.float64)
resultados['orden_IoU'] = resultados['orden_IoU'].astype(np.float64)
for i in range(1, len(resultados)):
    resultados.at[i, 'orden_L1'] = (np.log(resultados.at[i,'error_L1']) - np.log(resultados.at[i-1,'error_L1'])) / (np.log(resultados.at[i,'num_iter']) - np.log(resultados.at[i-1,'num_iter']))
    resultados.at[i, 'orden_IoU'] = (np.log(resultados.at[i,'IoU']) - np.log(resultados.at[i-1,'IoU'])) / (np.log(resultados.at[i,'num_iter']) - np.log(resultados.at[i-1,'num_iter']))
resultados['orden_L1'] = round(resultados['orden_L1'], 8)
resultados['orden_IoU'] = round(resultados['orden_IoU'], 8)    

# Tiempo
with open(f'{nombre_carpeta_principal}tiempo_entrenamiento.txt', 'r') as file:
    tiempo_lista = [float(line.strip()) for line in file.readlines()]

resultados['tiempo_horas'] = [int(duration // 3600) for duration in tiempo_lista]
resultados['tiempo_minutos'] = [int((duration % 3600) // 60) for duration in tiempo_lista]
resultados['tiempo_segundos'] = [duration % 60 for duration in tiempo_lista]

# Guardar resultados
resultados.to_csv(f'{nombre_carpeta_principal}resultados.csv', index=False)


### Generar gifs
for epoch in range(1,iterations+1):
    if epoch % 50000 == 0:
        modelo = Net()
        modelo.load_state_dict(torch.load(f'{nombre_carpeta_principal}modelos/{nombre}_{epoch}it.pth', map_location=device))
        modelo.eval()
        # Crear subcarpeta para el gif y las imagenes
        ruta_gif = nombre_carpeta_principal + f'gif_{epoch}it/'
        if not os.path.exists(ruta_gif):
            os.makedirs(ruta_gif)
        # generar gif
        malla = num_puntos_gif
        # Imagenes para generar gif
        png_list = []
        num_img = 1
        for i in np.linspace(0,1,21):
            tiempo_graf = round(i,2)
            graf = grafico_general(tiempo_graf, malla, problema, modelo)
            path_graf = f'{ruta_gif}{num_img}.png' # se guarda cada imagen en la carpeta del gif
            graf.savefig(path_graf)
            png_list.append(path_graf)
            num_img +=1
        # Generar gif con las imagenes generadas
        with imageio.get_writer(f'{ruta_gif}{nombre}_{epoch}it.gif', mode='I', duration=1000) as writer:
            for filename in png_list:
                image = imageio.v2.imread(filename)
                writer.append_data(image)


