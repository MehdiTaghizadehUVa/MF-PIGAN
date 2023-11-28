import sys
sys.path.insert(0, '../../Utilities/')
import argparse
import os
import torch
from collections import OrderedDict
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import seaborn as sns
import pylab as py
import time
from pyDOE import lhs
import warnings
sys.path.insert(0, '../../Scripts/')
from models_pde import Generator, Discriminator, Q_Net, Net
from mf_fo_pidgan_burgers import *
from pinn_burgers import *
# from ../Scripts/helper import *

warnings.filterwarnings('ignore')

np.random.seed(1234)
torch.manual_seed(1234)

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


num_epochs = 30000
lambda_phy = 1

noise = 0.0

## Network Architecture
hid_dim = 50
num_layer = 4

N_u = 300
N_i = 50
N_b = 100
N_f = 10000
data = scipy.io.loadmat('../../datasets/burgers_LF.mat')

t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = np.real(data['usol']).T

X, T = np.meshgrid(x,t)

T_u = T[1:, 1:-1]
X_u = X[1:, 1:-1]
x_u = np.hstack((X_u.flatten()[:, None], T_u.flatten()[:, None]))
U = Exact[1:, 1:-1].flatten()[:, None]

# selecting N_u internal points for training
idx0 = np.random.choice(x_u.shape[0], N_u, replace=False)
X_u_train = x_u[idx0, :]
u_u_train = U[idx0, :]


X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_star = Exact.flatten()[:,None]

# Doman bounds
lb = X_star.min(0)
ub = X_star.max(0)

# initial conditions t = 0
xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T))
uu1 = Exact[0:1,:].T

# boundary conditions x = lb
xx2 = np.hstack((X[:,0:1], T[:,0:1]))
uu2 = Exact[:,0:1]

# boundary conditions, x = ub
xx3 = np.hstack((X[:,-1:], T[:,-1:]))
uu3 = Exact[:,-1:]

X_b_train = np.vstack([xx2, xx3])
u_b_train = np.vstack([uu2, uu3])

X_f_train = lb + (ub-lb)*lhs(2, N_f)
X_f_train = np.vstack([X_f_train, X_b_train, xx1])

# selecting N_b boundary points for training
idx = np.random.choice(X_b_train.shape[0], N_b, replace=False)
X_b_train = X_b_train[idx, :]
u_b_train = u_b_train[idx,:]

# selecting N_i initial points for training
idx = np.random.choice(xx1.shape[0], N_i, replace=False)
X_i_train = xx1[idx, :]
u_i_train = uu1[idx, :]

# adding boundary and initial points
X_u_train = np.vstack([X_u_train, X_b_train, X_i_train])
u_train = np.vstack([u_u_train, u_b_train, u_i_train])

net = Net(in_dim = 2, out_dim = 1, hid_dim = hid_dim, num_layers = num_layer).to(device)

burgers_LF = Burgers_PINN(X_u_train, u_train, X_f_train, X_star, u_star, net, device, num_epochs, lambda_phy, noise)

burgers_LF.train()

u_pred, f_pred = burgers_LF.predict(X_star)

error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
print('Error u: %e' % (error_u))
print('Residual: %e' % (f_pred**2).mean())
U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
# U_dev = griddata(X_star, u_dev.flatten(), (X, T), method='cubic')
Error = np.abs(Exact - U_pred)


""" The aesthetic setting has changed. """

####### Row 0: u(t,x) ##################
X_u_train_ = X_u_train
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]

h = ax.imshow(Exact.T, interpolation='nearest', cmap='rainbow',
              extent=[t.min(), t.max(), x.min(), x.max()],
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h, cax=cax)
cbar.ax.tick_params(labelsize=15)

# ax.plot(
#     X_u_train_[:,1],
#     X_u_train_[:,0],
#     'kx', label = 'Data (%d points)' % (u_train.shape[0]),
#     markersize = 4,  # marker size doubled
#     clip_on = False,
#     alpha=1.0
# )


ax.set_xlabel('$t$', size=20)
ax.set_ylabel('$x$', size=20)
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.9, -0.05),
    ncol=5,
    frameon=False,
    prop={'size': 15}
)
ax.set_title('$u(t,x)- Eaxct$', fontsize = 20) # font size doubled
ax.tick_params(labelsize=15)

plt.savefig('./Figures/MF-FO-PIDGAN-Exact-LF.png', dpi = 600)
####### Row 0: u(t,x) ##################

fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(111)
t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]

h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
              extent=[t.min(), t.max(), x.min(), x.max()],
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h, cax=cax)
cbar.ax.tick_params(labelsize=15)

ax.plot(
    X_u_train_[:,1],
    X_u_train_[:,0],
    'kx', label = 'Data (%d points)' % (u_train.shape[0]),
    markersize = 4,  # marker size doubled
    clip_on = False,
    alpha=1.0
)


ax.set_xlabel('$t$', size=20)
ax.set_ylabel('$x$', size=20)
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.9, -0.05),
    ncol=5,
    frameon=False,
    prop={'size': 15}
)
ax.set_title('$u(t,x) - Prediction$', fontsize = 20) # font size doubled
ax.tick_params(labelsize=15)

plt.savefig('./Figures/MF-FO-PIDGAN-Prediction-LF.png', dpi = 600)

####### Row 0: u(t,x) ##################

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)
t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]

h = ax.imshow(np.abs(Exact.T - U_pred.T) , interpolation='nearest', cmap='rainbow',
              extent=[t.min(), t.max(), x.min(), x.max()],
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h, cax=cax)
cbar.ax.tick_params(labelsize=15)


# ax.plot(
#     X_u_train_[:,1],
#     X_u_train_[:,0],
#     'kx', label = 'Data (%d points)' % (u_train.shape[0]),
#     markersize = 4,  # marker size doubled
#     clip_on = False,
#     alpha=1.0
# )

ax.set_xlabel('$t$', size=20)
ax.set_ylabel('$x$', size=20)
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.9, -0.05),
    ncol=5,
    frameon=False,
    prop={'size': 15}
)
ax.set_title('$u(t,x) - Error$', fontsize = 20) # font size doubled
ax.tick_params(labelsize=15)

plt.savefig('./Figures/MF-FO-PIDGAN-Error-LF.png', dpi = 600)


num_epochs = 30000
lambda_phy = 1.0
lambda_q = 1.5

lambda_mse = 1
lambda_mse_d = 10

noise = 0.0


#architecture for the models
d_hid_dim = 64
d_num_layer = 2

g_hid_dim = 64
g_num_layer = 4

q_hid_dim = 64
q_num_layer = 4


N_b = 100
N_i = 50
N_u = 100
N_f = 10000
data = scipy.io.loadmat('../../datasets/burgers_HF.mat')

t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = np.real(data['usol']).T

X, T = np.meshgrid(x,t)

T_u = T[1:, 1:-1]
X_u = X[1:, 1:-1]
x_u = np.hstack((X_u.flatten()[:, None], T_u.flatten()[:, None]))
U = Exact[1:, 1:-1].flatten()[:, None]

# selecting N_u internal points for training
idx0 = np.random.choice(x_u.shape[0], N_u, replace=False)
X_u_train = x_u[idx0, :]
u_u_train = U[idx0, :]


X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))

# Doman bounds
lb = X_star.min(0)
ub = X_star.max(0)

u_pred, f_pred = burgers_LF.predict(X_star)
X_star = np.hstack([X_star, u_pred])
####
u_star = Exact.flatten()[:,None]

# initial conditions t = 0
xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T))
uu1 = Exact[0:1,:].T

# boundary conditions x = lb
xx2 = np.hstack((X[:,0:1], T[:,0:1]))
uu2 = Exact[:,0:1]

# boundary conditions, x = ub
xx3 = np.hstack((X[:,-1:], T[:,-1:]))
uu3 = Exact[:,-1:]

X_b_train = np.vstack([xx2, xx3])
u_b_train = np.vstack([uu2, uu3])

X_f_train = lb + (ub-lb)*lhs(2, N_f)
X_f_train = np.vstack([X_f_train, X_b_train, xx1])



u_pred, f_pred = burgers_LF.predict(X_f_train)

X_f_train = np.hstack([X_f_train, u_pred])
####

# selecting N_u boundary points for training
idx = np.random.choice(X_b_train.shape[0], N_b, replace=False)
X_b_train = X_b_train[idx, :]
u_b_train = u_b_train[idx,:]

# selecting N_i initial points for training
idx = np.random.choice(xx1.shape[0], N_i, replace=False)
X_i_train = xx1[idx, :]
u_i_train = uu1[idx, :]

# adding boundary and initial points
X_u_train = np.vstack([X_u_train, X_b_train, X_i_train])


u_pred, f_pred = burgers_LF.predict(X_u_train)
X_u_train = np.hstack([X_u_train, u_pred])
####

u_train = np.vstack([u_u_train, u_b_train, u_i_train])

D = Discriminator(in_dim = 5, out_dim = 1, hid_dim = d_hid_dim, num_layers = d_num_layer).to(device)
G = Generator(in_dim = 4, out_dim = 2, hid_dim = g_hid_dim, num_layers = g_num_layer).to(device)
Q = Q_Net(in_dim = 5, out_dim = 1, hid_dim = q_hid_dim, num_layers = q_num_layer).to(device)



burgers = Burgers_PIDGAN(X_u_train, u_train, X_f_train, X_star, u_star, G, D, Q, device, num_epochs, lambda_phy, lambda_mse,  lambda_mse_d, lambda_q, noise)


Error_history = burgers.train()


epoch_count = Error_history[:, 0]
Error_train = Error_history[:, 1]
Error_test = Error_history[:, 2]
fig = plt.figure(figsize=(8, 5))
plt.plot(epoch_count, Error_train, 'r-')
plt.plot(epoch_count, Error_test, 'g-')

plt.legend(['Training Error', 'Test Error'], fontsize = 10)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.yscale("log")
plt.savefig('./Figures/MF-FO-PIDGAN-ErrorCurves.png', dpi = 600)
np.savetxt('./Outputs/MF-FO-PIDGAN_Error_history.csv', Error_history, delimiter=',')

nsamples = 500
u_pred_list = []
f_pred_list = []
for run in range(nsamples):
    u_pred, f_pred = burgers.predict(X_star)
    u_pred_list.append(u_pred)
    f_pred_list.append(f_pred)

u_pred_arr = np.array(u_pred_list)
f_pred_arr = np.array(f_pred_list)
u_pred = u_pred_arr.mean(axis=0)
f_pred = f_pred_arr.mean(axis=0)
u_dev = u_pred_arr.var(axis=0)
f_dev = f_pred_arr.var(axis=0)


error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
print('Error u: %e' % (error_u))
print('Residual: %e' % (f_pred**2).mean())
U_pred = griddata(X_star[:,0:2], u_pred.flatten(), (X, T), method='cubic')
U_dev = griddata(X_star[:,0:2], u_dev.flatten(), (X, T), method='cubic')
Error = np.abs(Exact - U_pred)


""" The aesthetic setting has changed. """

####### Row 0: u(t,x) ##################
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['text.usetex'] = True

X_u_train_ = X_u_train
fig = plt.figure(figsize=(12.5, 7.5))
ax = fig.add_subplot(111)
t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]

h = ax.imshow(Exact.T, interpolation='nearest', cmap='rainbow',
              extent=[t.min(), t.max(), x.min(), x.max()],
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h, cax=cax)
cbar.ax.tick_params(labelsize=24)


ax.set_xlabel(r'$t$', size=28)
ax.set_ylabel(r'$x$', size=28)
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.9, -0.05),
    ncol=5,
    frameon=False,
    prop={'size': 24}
)

ax.tick_params(labelsize=24)

plt.savefig('./Figures/MF-FO-PIDGAN-Exact.png', dpi = 600)
####### Row 0: u(t,x) ##################

fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(111)
t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]

h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
              extent=[t.min(), t.max(), x.min(), x.max()],
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h, cax=cax)
cbar.ax.tick_params(labelsize=15)

ax.plot(
    X_u_train_[:,1],
    X_u_train_[:,0],
    'kx', label = 'Data (%d points)' % (u_train.shape[0]),
    markersize = 4,  # marker size doubled
    clip_on = False,
    alpha=1.0
)

ax.set_xlabel('$t$', size=20)
ax.set_ylabel('$x$', size=20)
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.9, -0.05),
    ncol=5,
    frameon=False,
    prop={'size': 15}
)
ax.set_title('$u(t,x): Prediction$', fontsize = 20) # font size doubled
ax.tick_params(labelsize=15)

plt.savefig('./Figures/MF-FO-PIDGAN-Prediction.png', dpi = 600)
####### Row 0: u(t,x) ##################

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]

h = ax.imshow(U_dev.T, interpolation='nearest', cmap='rainbow',
              extent=[t.min(), t.max(), x.min(), x.max()],
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h, cax=cax)
cbar.ax.tick_params(labelsize=15)

# ax.plot(
#     X_u_train_[:,1],
#     X_u_train_[:,0],
#     'kx', label = 'Data (%d points)' % (u_train.shape[0]),
#     markersize = 4,  # marker size doubled
#     clip_on = False,
#     alpha=1.0
# )
ax.set_xlabel('$t$', size=20)
ax.set_ylabel('$x$', size=20)
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.9, -0.05),
    ncol=5,
    frameon=False,
    prop={'size': 15}
)
ax.tick_params(labelsize=15)

plt.savefig('./Figures/MF-FO-PIDGAN-StDv.png', dpi = 600)
####### Row 0: u(t,x) ##################

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['text.usetex'] = True

fig = plt.figure(figsize=(12.5, 7.5))
ax = fig.add_subplot(111)
t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]

vmin_value = 0  # Set your minimum value
vmax_value = 0.08  # Set your maximum value
h = ax.imshow(np.abs(Exact.T - U_pred.T), interpolation='nearest', cmap='rainbow',
              extent=[t.min(), t.max(), x.min(), x.max()],
              origin='lower', aspect='auto',
              vmin=vmin_value, vmax=vmax_value)  # Set vmin and vmax here
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h, cax=cax)
cbar.ax.tick_params(labelsize=24)


ax.set_xlabel(r'$t$', size=24)
ax.set_ylabel(r'$x$', size=24)
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.9, -0.05),
    ncol=5,
    frameon=False,
    prop={'size': 24}
)
# ax.set_title('$u(t,x): Error.$', fontsize = 20) # font size doubled
ax.tick_params(labelsize=24)

plt.savefig('./Figures/MF-FO-PIDGAN-Error.eps', dpi=266, bbox_inches='tight', format='eps')
plt.show()