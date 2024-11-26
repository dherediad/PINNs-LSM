import os
import numpy as np
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Nombre del experimento
nombre = 'r_circulo_100x100_5c_200n_sigmoid'

# Crear carpeta para guardar modelos y resultados (carpeta principal)
nombre_carpeta_principal = f'experimentos/circulo/{nombre}/'
if not os.path.exists(nombre_carpeta_principal):
  os.makedirs(nombre_carpeta_principal)
# Crear carpeta para modelos
nombre_carpeta_modelos = nombre_carpeta_principal + "modelos/"
if not os.path.exists(nombre_carpeta_modelos):
  os.makedirs(nombre_carpeta_modelos)

# Configuracion de la red neuronal
num_puntos = 100  # malla
print(f'Malla: {num_puntos}x{num_puntos}')
num_neuronas = 200
  
## Clase de la red neuronal
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
        layer1_out = torch.sigmoid(self.hidden_layer1(inputs))
        layer2_out = torch.sigmoid(self.hidden_layer2(layer1_out))
        layer3_out = torch.sigmoid(self.hidden_layer3(layer2_out))
        layer4_out = torch.sigmoid(self.hidden_layer4(layer3_out))
        layer5_out = torch.sigmoid(self.hidden_layer5(layer4_out))
        output = self.output_layer(layer5_out)
        return output
  
## Modelo
net = Net()
net = net.to(device)
mse_cost_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
print("Modelo:")
print(net)

## Loss PDE
def pde_func(x,y,t,net):
    u = net(x,y,t)
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    pde = u_t - 2*np.pi*(y-0.5)*u_x + 2*np.pi*(x-0.5)*u_y
    pde = pde.to(torch.float32).to(device)
    return pde
  
## Datos iniciales 
# SDF inicial (phi0) - Se usa malla (num_puntos) definido anteriormente
x_ic = np.linspace(0,1,num_puntos)
y_ic = np.linspace(0,1,num_puntos)
# Grid
x_ic_grid, y_ic_grid = np.meshgrid(x_ic, y_ic)
## Ajuste a vector
x_ic_grid = np.ravel(x_ic_grid).reshape(-1,1)
y_ic_grid = np.ravel(y_ic_grid).reshape(-1,1)
# Circulo SDF
u_ic_circulo = np.sqrt((x_ic_grid-0.5)**2+(y_ic_grid-0.75)**2) - 0.15
u_ic = u_ic_circulo.ravel().reshape(-1,1)

# Tiempo 0
t_ic = np.zeros((num_puntos*num_puntos,1))

# Numero de puntos para vector x, y collocation
num_puntos_x = num_puntos
num_puntos_y = num_puntos
# Vector de zeros
all_zeros = np.zeros((num_puntos_x*num_puntos_y,1))

loss_list = []
duration_list = []

start_time = time.time()

### Training / Fitting
iterations = 2000000
for epoch in range(1,iterations+1):
    optimizer.zero_grad() 
    ## Loss Condiciones Iniciales
    pt_x_ic = Variable(torch.from_numpy(x_ic_grid).float(), requires_grad=False).to(device)
    pt_y_ic = Variable(torch.from_numpy(y_ic_grid).float(), requires_grad=False).to(device)
    pt_t_ic = Variable(torch.from_numpy(t_ic).float(), requires_grad=False).to(device)
    pt_u_ic = Variable(torch.from_numpy(u_ic).float(), requires_grad=False).to(device)
    net_ic_out = net(pt_x_ic, pt_y_ic, pt_t_ic) 
    mse_u = mse_cost_function(net_ic_out, pt_u_ic)
    ## Loss PDE
    x_collocation = np.random.uniform(low=0.0, high=1.0, size=(num_puntos_x*num_puntos_y,1))
    y_collocation = np.random.uniform(low=0.0, high=1.0, size=(num_puntos_x*num_puntos_y,1))
    t_collocation = np.random.uniform(low=0.0, high=1.0, size=(num_puntos_x*num_puntos_y,1))
    pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True).to(device)
    pt_y_collocation = Variable(torch.from_numpy(y_collocation).float(), requires_grad=True).to(device)
    pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True).to(device)
    pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)
    f_out = pde_func(pt_x_collocation, pt_y_collocation, pt_t_collocation, net) 
    mse_f = mse_cost_function(f_out, pt_all_zeros)
    # Combinar loss
    loss = mse_u + mse_f
    loss.backward()
    optimizer.step()
        
    with torch.autograd.no_grad():
      loss_list.append(loss.data.item())
      if epoch % 50000 == 0:
          # Almacenar tiempo de entrenamiento (en segundos) en la lista
          end_time = time.time()
          duration = end_time - start_time
          duration_list.append(duration)
          # Guardar modelo:
          path_modelo = f'{nombre_carpeta_modelos}{nombre}_{epoch}it.pth'
          torch.save(net.state_dict(), path_modelo)
          print(f'Iteracion: {epoch}, Loss: {loss.data}')


# Guardar tiempo
tiempo_archivo = nombre_carpeta_principal + 'tiempo_entrenamiento.txt'
with open(tiempo_archivo, 'w') as file:
   for item in duration_list:
        file.write(str(item) + '\n')
   
# Guardar loss
loss_archivo = nombre_carpeta_principal + 'loss.txt'
with open(loss_archivo, 'w') as file:
    for item in loss_list:
        file.write(str(item) + '\n')

