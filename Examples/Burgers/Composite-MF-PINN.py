import sys

sys.path.insert(0, '../../Utilities/')

import torch
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
import scipy.io
from scipy.interpolate import griddata
from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import timeit

np.random.seed(1234)
torch.manual_seed(1234)

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# the deep neural network
class Net(torch.nn.Module):
    def __init__(self, layers, islinear=False):
        super(Net, self).__init__()
        # parameters
        self.depth = len(layers) - 1
        #activation function
        if islinear:
            self.activation = torch.nn.Identity
        else:
            self.activation = torch.nn.Tanh

        # set up layer order dict
        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out


class Composite_PINN():
    def __init__(self, X_LF, u_LF, X_HF, u_HF, X_F, X_star, U_star, Net_LF, Net_HF_L, Net_HF_NL, lambda_reg, lambda_phy,
                 device):
        # data
        self.Xmean, self.Xstd = X_F.mean(0), X_F.std(0)
        self.x_F = (X_F - self.Xmean) / self.Xstd
        self.x_lf = (X_LF - self.Xmean) / self.Xstd
        self.x_hf = (X_HF - self.Xmean) / self.Xstd
        self.x_star = (X_star - self.Xmean) / self.Xstd

        self.u_LF = u_LF
        self.u_HF = u_HF
        self.u_star = U_star

        # Jacobian of the PDE because of normalization
        self.Jacobian_X = 1 / self.Xstd[0]
        self.Jacobian_T = 1 / self.Xstd[1]

        self.device = device

        self.train_x_u_lf = torch.tensor(self.x_lf, requires_grad=True).float().to(self.device)
        self.train_x_u_hf = torch.tensor(self.x_hf, requires_grad=True).float().to(self.device)
        self.train_y_lf = torch.tensor(self.u_LF, requires_grad=True).float().to(self.device)
        self.train_y_hf = torch.tensor(self.u_HF, requires_grad=True).float().to(self.device)
        self.train_x_f = torch.tensor(self.x_F, requires_grad=True).float().to(self.device)

        # deep neural networks
        self.Net_LF = Net_LF
        self.Net_HF_L = Net_HF_L
        self.Net_HF_NL = Net_HF_NL
        params = list(self.Net_LF.parameters()) + list(self.Net_HF_L.parameters()) + list(self.Net_HF_NL.parameters())

        self.optimizer_Adam = torch.optim.Adam(params, lr=1e-4, betas=(0.9, 0.999))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_Adam, step_size=1000, gamma=0.9)

        self.optimizer = torch.optim.LBFGS(
            params,
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"  # can be "strong_wolfe"
        )

        self.iter = 0
        self.lambda_reg = lambda_reg
        self.lambda_phy = lambda_phy

    def phy_residual(self, x, t, u, nu=(0.01 / np.pi)):
        """ The pytorch autograd version of calculating residual """

        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]

        f = (self.Jacobian_T) * u_t + (self.Jacobian_X) * u * u_x - nu * (self.Jacobian_X ** 2) * u_xx
        return f

    def n_phy_prob(self, X):
        # xl = torch.tensor(X[:, 0:1], requires_grad=False).float().to(self.device)
        # tl = torch.tensor(X[:, 1:2], requires_grad=False).float().to(self.device)
        xl = X[:, 0:1].clone().detach().requires_grad_(False).to(self.device)
        tl = X[:, 1:2].clone().detach().requires_grad_(False).to(self.device)
        u_lf = self.Net_LF(torch.cat([xl, tl], dim=1))

        # xh = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)
        # th = torch.tensor(X[:, 1:2], requires_grad=True).float().to(self.device)
        xh = X[:, 0:1].clone().detach().requires_grad_(True).to(self.device)
        th = X[:, 1:2].clone().detach().requires_grad_(True).to(self.device)
        u_hf_l = self.Net_HF_L(torch.cat([xh, th, u_lf], dim=1))
        u_hf_nl = self.Net_HF_NL(torch.cat([xh, th, u_lf], dim=1))

        u = u_hf_l + u_hf_nl

        residual = self.phy_residual(xh, th, u)
        return residual, u

    def compute_error(self):

        u_pred, residual = self.predict(self.x_star)
        error_u = np.linalg.norm(self.u_star - u_pred, 2) / np.linalg.norm(self.u_star, 2)

        residual = np.linalg.norm(residual, 2)

        u_pred, residual_train = self.predict(self.x_hf)
        error_u_train = np.linalg.norm(self.u_HF - u_pred, 2) / np.linalg.norm(self.u_HF, 2)

        residual_train = np.linalg.norm(residual_train, 2)

        return error_u_train, residual_train, error_u, residual

    def loss_func(self):

        u_lf = self.Net_LF(self.train_x_u_lf)
        _, u_hf = self.n_phy_prob(self.train_x_u_hf)

        loss_lf = torch.nn.functional.mse_loss(self.train_y_lf, u_lf)
        loss_hf = torch.nn.functional.mse_loss(self.train_y_hf, u_hf)

        l2_norm = sum(p.pow(2.0).sum()
                      for p in self.Net_HF_NL.parameters())

        residual, _, _ = self.n_phy_prob(self.train_x_f)
        loss_phy = torch.mean(residual ** 2)
        loss = loss_lf + loss_hf + self.lambda_reg * l2_norm + self.lambda_phy * loss_phy

        self.optimizer.zero_grad()
        loss.backward()

        self.iter += 1
        if self.iter % n_print == 0:
            error_u_train, residual_train, error_u, residual = self.compute_error()
            print(
                'Iter: %d, Loss: %.3e, Loss_LF: %.3e, Loss_HF: %.3e, Loss_reg: %.3e , Loss_phy: %.3e, Error_Train: %.3e, Error_Test: %.3e, Residual_Train: %.3e, Residual_Test: %.3e' %
                (
                    epoch,
                    loss.item(),
                    loss_lf.item(),
                    loss_hf.item(),
                    loss_reg.item(),
                    loss_phy.item(),
                    error_u_train,
                    error_u,
                    residual_train,
                    residual
                )
            )
        return loss

    def train(self, nIter, n_print=100):
        counter = 0
        Error_history = np.zeros((nIter // n_print, 3))
        start_time = timeit.default_timer()
        for epoch in range(nIter):
            u_lf = self.Net_LF(self.train_x_u_lf)
            _, u_hf = self.n_phy_prob(self.train_x_u_hf)

            loss_lf = torch.nn.functional.mse_loss(self.train_y_lf, u_lf)
            loss_hf = torch.nn.functional.mse_loss(self.train_y_hf, u_hf)

            l2_norm_HF = sum(p.pow(2.0).sum()
                             for p in self.Net_HF_NL.parameters())

            l2_norm_LF = sum(p.pow(2.0).sum()
                             for p in self.Net_LF.parameters())

            loss_reg = self.lambda_reg * (l2_norm_HF + l2_norm_LF)

            residual, _, = self.n_phy_prob(self.train_x_f)
            loss_phy = torch.mean(residual ** 2)
            loss = loss_lf + loss_hf + loss_reg + self.lambda_phy * loss_phy
            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer_Adam.step()

            self.scheduler.step()

            if epoch % 100 == 0:
                elapsed = timeit.default_timer() - start_time
                error_u_train, residual_train, error_u, residual = self.compute_error()
                Error_history[counter, 0] = epoch
                Error_history[counter, 1] = error_u_train
                Error_history[counter, 2] = error_u
                counter += 1
                print(
                    'Epoch: %d, Loss: %.3e, Loss_LF: %.3e, Loss_HF: %.3e, Loss_reg: %.3e , Loss_phy: %.3e, Error_Train: %.3e, Error_Test: %.3e, Residual_Train: %.3e, Residual_Test: %.3e, Time per Epoch: %f' %
                    (
                        epoch,
                        loss.item(),
                        loss_lf.item(),
                        loss_hf.item(),
                        loss_reg.item(),
                        loss_phy.item(),
                        error_u_train,
                        error_u,
                        residual_train,
                        residual,
                        elapsed / 100
                    )
                )
                start_time = timeit.default_timer()
        #         self.optimizer.step(self.loss_func)
        return Error_history

    def predict(self, X):
        xl = torch.tensor(X[:, 0:1], requires_grad=False).float().to(self.device)
        tl = torch.tensor(X[:, 1:2], requires_grad=False).float().to(self.device)

        u_lf = self.Net_LF(torch.cat([xl, tl], dim=1))

        xh = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)
        th = torch.tensor(X[:, 1:2], requires_grad=True).float().to(self.device)

        u_hf_l = self.Net_HF_L(torch.cat([xh, th, u_lf], dim=1))
        u_hf_nl = self.Net_HF_NL(torch.cat([xh, th, u_lf], dim=1))

        u = u_hf_l + u_hf_nl
        residual = self.phy_residual(xh, th, u)

        u = u.detach().cpu().numpy()
        residual = residual.detach().cpu().numpy()

        return u, residual


num_epochs = 30000
noise = 0.0

layers_LF = [2, 64, 64, 1]
# layers_LF = [2, 20, 20, 1]
layers_HF_L = [3, 1]
# layers_HF_NL = [3 , 20, 20, 1]
layers_HF_NL = [3 , 64, 64, 1]

lambda_reg = 1e-5
lambda_phy = 10

N_b = 100
N_i = 50
N_u = 300

data = scipy.io.loadmat('../../datasets/burgers_LF.mat')

t = data['t'].flatten()[:, None]
x = data['x'].flatten()[:, None]
Exact = np.real(data['usol']).T

X, T = np.meshgrid(x, t)

T_u = T[1:, 1:-1]
X_u = X[1:, 1:-1]
x_u = np.hstack((X_u.flatten()[:, None], T_u.flatten()[:, None]))
U = Exact[1:, 1:-1].flatten()[:, None]

# selecting N_u internal points for training
idx0 = np.random.choice(x_u.shape[0], N_u, replace=False)
X_u_train = x_u[idx0, :]
u_u_train = U[idx0, :]

X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_star = Exact.flatten()[:, None]

# Doman bounds
lb = X_star.min(0)
ub = X_star.max(0)

# initial conditions t = 0
xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))
uu1 = Exact[0:1, :].T

# boundary conditions x = lb
xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))
uu2 = Exact[:, 0:1]

# boundary conditions, x = ub
xx3 = np.hstack((X[:, -1:], T[:, -1:]))
uu3 = Exact[:, -1:]

X_b_train = np.vstack([xx2, xx3])
u_b_train = np.vstack([uu2, uu3])

# selecting N_b boundary points for training
idx1 = np.random.choice(X_b_train.shape[0], N_b, replace=False)
X_b_train = X_b_train[idx1, :]
u_b_train = u_b_train[idx1, :]

# selecting N_i initial points for training
idx2 = np.random.choice(xx1.shape[0], N_i, replace=False)
X_i_train = xx1[idx2, :]
u_i_train = uu1[idx2, :]

# adding boundary and initial points
X_u_train_lf = np.vstack([X_u_train, X_b_train, X_i_train])
u_train_lf = np.vstack([u_u_train, u_b_train, u_i_train])

N_b = 100
N_i = 50
N_u = 100
N_f = 10000
data = scipy.io.loadmat('../../datasets/burgers_HF.mat')

t = data['t'].flatten()[:, None]
x = data['x'].flatten()[:, None]
Exact = np.real(data['usol']).T

X, T = np.meshgrid(x, t)

T_u = T[1:, 1:-1]
X_u = X[1:, 1:-1]
x_u = np.hstack((X_u.flatten()[:, None], T_u.flatten()[:, None]))
U = Exact[1:, 1:-1].flatten()[:, None]

# selecting N_u internal points for training
idx0 = np.random.choice(x_u.shape[0], N_u, replace=False)
X_u_train = x_u[idx0, :]
u_u_train = U[idx0, :]

X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_star = Exact.flatten()[:, None]

# Doman bounds
lb = X_star.min(0)
ub = X_star.max(0)

# initial conditions t = 0
xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))
uu1 = Exact[0:1, :].T

# boundary conditions x = lb
xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))
uu2 = Exact[:, 0:1]

# boundary conditions, x = ub
xx3 = np.hstack((X[:, -1:], T[:, -1:]))
uu3 = Exact[:, -1:]

X_b_train = np.vstack([xx2, xx3])
u_b_train = np.vstack([uu2, uu3])

X_f_train = lb + (ub - lb) * lhs(2, N_f)
X_f_train = np.vstack([X_f_train, X_b_train, xx1])

# selecting N_b boundary points for training
idx1 = np.random.choice(X_b_train.shape[0], N_b, replace=False)
X_b_train = X_b_train[idx1, :]
u_b_train = u_b_train[idx1, :]

# selecting N_i initial points for training
idx2 = np.random.choice(xx1.shape[0], N_i, replace=False)
X_i_train = xx1[idx2, :]
u_i_train = uu1[idx2, :]

# adding boundary and initial points
X_u_train_hf = np.vstack([X_u_train, X_b_train, X_i_train])
u_train_hf = np.vstack([u_u_train, u_b_train, u_i_train])

Net_LF = Net(layers_LF).to(device)
Net_HF_L = Net(layers_HF_L, islinear=True).to(device)
Net_HF_NL = Net(layers_HF_NL).to(device)

MF_PINN = Composite_PINN(X_u_train_lf, u_train_lf, X_u_train_hf, u_train_hf, X_f_train, X_star, u_star, Net_LF,
                         Net_HF_L, Net_HF_NL, lambda_reg, lambda_phy, device)

Error_history = MF_PINN.train(num_epochs)


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
plt.savefig('MF-PINN-ErrorCurves.png', dpi = 600)
np.savetxt('MF-PINN-Error_history.csv', Error_history, delimiter=',')


Xmean = MF_PINN.Xmean
Xstd = MF_PINN.Xstd
X_star_norm = (X_star - Xmean) / Xstd

u_pred, f_pred = MF_PINN.predict(X_star_norm)


error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
print('Error u: %e' % (error_u))
print('Residual: %e' % (f_pred ** 2).mean())
U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
# U_dev = griddata(X_star, u_dev.flatten(), (X, T), method='cubic')
Error = np.abs(Exact - U_pred)


####### Row 0: u(t,x) ##################
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['text.usetex'] = True

X_u_train_ = X_u_train_hf
fig = plt.figure(figsize=(12.5, 7.5))
ax = fig.add_subplot(111)
t = data['t'].flatten()[:, None]
x = data['x'].flatten()[:, None]

h = ax.imshow(Exact.T, interpolation='nearest', cmap='rainbow',
              extent=[t.min(), t.max(), x.min(), x.max()],
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h, cax=cax)
cbar.ax.tick_params(labelsize=24)

# ax.plot(
#     X_u_train_[:, 1],
#     X_u_train_[:, 0],
#     'kx', label='Data (%d points)' % (X_u_train_.shape[0]),
#     markersize=4,  # marker size doubled
#     clip_on=False,
#     alpha=1.0
# )

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

# plt.savefig('MF-PINN-Exact.png', dpi = 600)
plt.savefig('MF-PINN-Exact.eps', dpi=266, bbox_inches='tight', format='eps')
####### Row 0: u(t,x) ##################

fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(111)
t = data['t'].flatten()[:, None]
x = data['x'].flatten()[:, None]

h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
              extent=[t.min(), t.max(), x.min(), x.max()],
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h, cax=cax)
cbar.ax.tick_params(labelsize=15)

ax.plot(
    X_u_train_[:, 1],
    X_u_train_[:, 0],
    'kx', label='Data (%d points)' % (X_u_train_.shape[0]),
    markersize=4,  # marker size doubled
    clip_on=False,
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
ax.set_title('$u(t,x) - Prediction$', fontsize=20)  # font size doubled
ax.tick_params(labelsize=15)

plt.savefig('MF-PINN-Prediction.png', dpi = 600)
####### Row 0: u(t,x) ##################

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['text.usetex'] = True

fig = plt.figure(figsize=(12.5, 7.5))
ax = fig.add_subplot(111)
t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]

vmin_value = 0  # Set your minimum value
vmax_value = 0.6  # Set your maximum value
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
plt.savefig('MF-PINN-Error.eps', dpi=266, bbox_inches='tight', format='eps')