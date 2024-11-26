# Script para graficos
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
device = torch.device('cpu')

def datosAX(tiempo, num_puntos, net):
    x=np.linspace(0,1,num_puntos)
    y=np.linspace(0,1,num_puntos)
    ms_x, ms_y = np.meshgrid(x, y)
    x = np.ravel(ms_x).reshape(-1,1)
    y = np.ravel(ms_y).reshape(-1,1)
    t = np.ones((num_puntos*num_puntos,1)).reshape(-1,1)*tiempo
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    pt_y = Variable(torch.from_numpy(y).float(), requires_grad=True).to(device)
    pt_t = Variable(torch.from_numpy(t).float(), requires_grad=True).to(device)
    pt_u = net(pt_x,pt_y,pt_t)
    u = pt_u.cpu().detach().numpy()
    u = u.reshape(num_puntos,num_puntos)
    return ms_x, ms_y, u, tiempo


def graficoAX(z_graf, tiempo, ax, num_puntos):
    ax.contour(np.linspace(0,1,num_puntos),
               np.linspace(0,1,num_puntos),
               z_graf,
               levels=np.linspace(-0.5, 1, 110),
               vmin=-0.2, vmax=0.6, 
               norm = colors.Normalize(vmin=-0.20, vmax=0.60), 
               cmap='jet')
    ax.contour(np.linspace(0,1,num_puntos),
               np.linspace(0,1,num_puntos),
               z_graf, levels=[0], colors='black')
    ax.set_title(f'Curvas de Nivel en $t={tiempo:.2f}$')
    ax.axis("scaled")


def vf_contourAX(u_ic_contorno, tiempo, ax, problema):
    ### Campo Vectorial - circulo / zalesak / rectangulo o vortex
    if problema in ('circulo','zalesak','rectangulo'):    
        x_vf, y_vf = np.meshgrid(np.linspace(0, 1, 10),
                                 np.linspace(0, 1, 10))
        u_vf = -2*np.pi * (y_vf - 0.5)
        v_vf = 2*np.pi * (x_vf -0.5)
        ax.quiver(x_vf, y_vf, u_vf, v_vf, color='blue', pivot="middle", scale=40, alpha=0.6)
    elif problema == 'vortex':
        x_vf, y_vf = np.meshgrid(np.linspace(0, 1, 10),
                                 np.linspace(0, 1, 10))
        u_vf = -(-(np.sin(np.pi*x_vf)**2)*np.sin(2*np.pi*y_vf)*np.cos(np.pi*tiempo/1))
        v_vf = -(np.sin(2*np.pi*x_vf)*(np.sin(np.pi*y_vf)**2)*np.cos(np.pi*tiempo/1))
        ax.quiver(x_vf, y_vf, u_vf, v_vf, color='blue', pivot="middle",
                  #scale=45,
                  alpha=0.6)
    ### Contorno
    # Ejecutar primero datosAX y usar u como u_ic_contorno
    ax.contour(np.linspace(0,1,u_ic_contorno.shape[0]),
               np.linspace(0,1,u_ic_contorno.shape[0]),
               u_ic_contorno,
               levels=[0], colors="black")
    ax.set_title(f'Campo Vectorial y Curva de Nivel 0 en $t={tiempo:.2f}$')
    ax.axis("scaled")


def grafico_phiAX(x_graf, y_graf, z_graf, tiempo, ax, fig):
    vmin = -0.20 
    vmax = 0.60
    # funcion de nivel
    surf = ax.plot_surface(x_graf, y_graf, z_graf,
                           cmap='jet',
                           linewidth=0, antialiased=False, alpha=0.4,
                           vmin = vmin, vmax =vmax, 
                           norm = colors.Normalize(vmin=vmin, vmax=vmax))
    # curva de nivel 0
    ax.contour(x_graf, y_graf, z_graf, levels=[0],
               colors='black', alpha=1, linestyles='solid')
    ax.set_zlim(vmin, vmax)
    ax.zaxis.set_major_formatter('{x:.02f}')
    # colorbar
    fig.colorbar(surf, shrink=0.9, aspect=10, pad=0.1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # Vista del grafico
    # Default: elev=30, azim=-60
    ax.view_init(elev=40, azim=-30)
    ax.set_title(fr'Función $\phi$ y Curva de Nivel 0 en $t={tiempo:.2f}$')


def grafico_general(t, num_pts, problema, red):
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 4)
    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2:], projection='3d')
    ax3 = fig.add_subplot(gs[1, 1:3:])
    # subplots
    x_graf, y_graf, z_graf, t = datosAX(tiempo = t, num_puntos = num_pts, net = red)
    graficoAX(z_graf, t, ax1, num_pts)
    vf_contourAX(z_graf, t, ax3, problema)
    grafico_phiAX(x_graf, y_graf, z_graf, t, ax2, fig)
    # titulo
    fig.suptitle(f'Resultados en $t={t:.2f}$', color='maroon', fontsize=14)
    plt.tight_layout()
    grafico_final = fig
    plt.close()
    return grafico_final