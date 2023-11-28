import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io
from scipy.interpolate import griddata
from torch.utils.data import TensorDataset, DataLoader


class Burgers_PIGAN():
    def __init__(self, x_u, y_u, x_f, X_star, u_star, G, D, Q, device, nepochs, lambda_phy=1.0, lambda_mse=0.0,
                 lambda_q=1.5, noise=0.0):
        super(Burgers_PIGAN, self).__init__()

        # Normalize data
        self.Xmean, self.Xstd = x_f.mean(0), x_f.std(0)
        self.x_f = (x_f - self.Xmean) / self.Xstd
        # self.Xmean, self.Xstd = x_u.mean(0), x_u.std(0)
        self.x_u = (x_u - self.Xmean) / self.Xstd
        self.X_star = X_star
        self.u_star = u_star

        # Jacobian of the PDE because of normalization
        self.Jacobian_X = 1 / self.Xstd[0]
        self.Jacobian_T = 1 / self.Xstd[1]

        self.y_u = y_u + noise * np.std(y_u) * np.random.randn(y_u.shape[0], y_u.shape[1])

        self.G = G
        self.D = D
        self.Q = Q

        def weights_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

        self.G = self.G.apply(weights_init)
        self.D = self.D.apply(weights_init)
        self.Q = self.Q.apply(weights_init)

        self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr=1e-4, betas=(0.9, 0.999))
        params = list(self.G.parameters()) + list(self.Q.parameters())
        self.G_optimizer = torch.optim.Adam(params, lr=1e-4, betas=(0.9, 0.999))
        
        self.D_scheduler = torch.optim.lr_scheduler.StepLR(self.D_optimizer, step_size=5000, gamma=0.9)
        self.G_scheduler = torch.optim.lr_scheduler.StepLR(self.G_optimizer, step_size=5000, gamma=0.9)
        
        self.device = device

        # numpy to tensor
        self.train_x_u = torch.tensor(self.x_u, requires_grad=True).float().to(self.device)
        self.train_y_u = torch.tensor(self.y_u, requires_grad=True).float().to(self.device)
        self.train_x_f = torch.tensor(self.x_f, requires_grad=True).float().to(self.device)

        self.nepochs = nepochs
        self.lambda_phy = lambda_phy
        self.lambda_mse = lambda_mse
        self.lambda_q = lambda_q

        # Ratio of training for generator and discriminator in each iteration: k1 for discriminator, k2 for generator
        self.k1 = 1
        self.k2 = 5

        self.batch_size = 150
        shuffle = True
        self.train_loader = DataLoader(
            list(zip(self.train_x_u, self.train_y_u)), batch_size=self.batch_size, shuffle=shuffle
        )

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.constant_(m.bias, 0)

    def discriminator_loss(self, logits_real_u, logits_fake_u):
        loss = -torch.mean(torch.log(1 - torch.sigmoid(logits_real_u) + 1e-8) + \
                           torch.log(torch.sigmoid(logits_fake_u) + 1e-8))
        return loss

    def generator_loss(self, logits_fake_u):
        gen_loss = torch.mean(logits_fake_u)
        return gen_loss

    def sample_noise(self, number, size=1):
        noises = torch.randn((number, size)).float().to(self.device)
        return noises

    def get_residual(self, x, t, u, nu=0.01 / np.pi):
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

    def generate_fake(self, X, noise):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(self.device)
        q = torch.tensor(X[:, 2:3], requires_grad=True).float().to(self.device)
        u = self.G(torch.cat([x, t, q, noise], dim=1))
        residual = self.get_residual(x, t, u)
        return residual, u

    def predict(self, X):
        X = (X - self.Xmean) / self.Xstd
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(self.device)
        q = torch.tensor(X[:, 2:3], requires_grad=True).float().to(self.device)
        noise = self.sample_noise(number=x.shape[0])
        u = self.G(torch.cat([x, t, q, noise], dim=1))
        f = self.get_residual(x, t, u)
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        return u, f

    def average_prediction(self, nsamples=50):
        u_pred_list = []
        f_pred_list = []
        for run in range(nsamples):
            u_pred, f_pred = self.predict(self.X_star)
            u_pred_list.append(u_pred)
            f_pred_list.append(f_pred)

        u_pred_arr = np.array(u_pred_list)
        f_pred_arr = np.array(f_pred_list)

        u_pred = u_pred_arr.mean(axis=0)
        f_pred = f_pred_arr.mean(axis=0)

        u_dev = u_pred_arr.var(axis=0)
        f_dev = f_pred_arr.var(axis=0)

        error_u = np.linalg.norm(self.u_star - u_pred, 2) / np.linalg.norm(self.u_star, 2)
        residual = (f_pred ** 2).mean()
        
        x_u = self.x_u * self.Xstd + self.Xmean
        u_pred_train, f_pred_train = self.predict(x_u)

        error_u_train = np.linalg.norm(self.y_u - u_pred_train, 2) / np.linalg.norm(self.y_u, 2)

        residual_train = (f_pred_train ** 2).mean()

        return error_u, residual, error_u_train, residual_train

    def train_disriminator(self, x, y, Z_u):
        real_logits = self.D(torch.cat([x, y], dim=1))

        # physics loss for boundary points
        _, u = self.generate_fake(x, Z_u)
        # u = torch.tensor(u.detach(), requires_grad=True).float().to(self.device)
        fake_logits_u = self.D(torch.cat([x, u.detach()], dim=1))

        l2_lambda = 1e-3
        l2_norm = sum(p.pow(2.0).sum()
                      for p in self.D.parameters())

        reg_loss = l2_lambda * l2_norm

        # discriminator loss
        bce_loss = self.discriminator_loss(real_logits, fake_logits_u)

        d_loss = bce_loss + reg_loss

        return d_loss, bce_loss, reg_loss

    def train_generator(self, x, y, Z_u, Z_f):

        # physics loss for collocation points
        residual_f, _ = self.generate_fake(self.train_x_f, Z_f)

        # physics loss for boundary points
        _, y_pred = self.generate_fake(x, Z_u)
        fake_logits_u = self.D(torch.cat([x, y_pred], dim=1))

        l2_lambda = 1e-4
        l2_norm = sum(p.pow(2.0).sum()
                      for p in self.G.parameters())

        reg_loss = l2_lambda * l2_norm

        z_pred = self.Q(torch.cat([x, y_pred], dim=1))
        loss_q = - torch.nn.functional.mse_loss(z_pred, Z_u)
        loss_q = (1 - self.lambda_q) * loss_q

        mse_loss = self.lambda_mse * torch.nn.functional.mse_loss(y_pred, y)
        adv_loss = self.generator_loss(fake_logits_u)

        phy_loss = torch.mean(torch.square(residual_f))
        phy_loss = self.lambda_phy * phy_loss

        g_loss = adv_loss + loss_q + phy_loss + reg_loss + mse_loss
        #################################
        return g_loss, adv_loss, mse_loss, phy_loss, reg_loss, loss_q

    def train(self, n_print=100):
        Adv_loss = np.zeros(self.nepochs)
        G_loss = np.zeros(self.nepochs)
        D_loss = np.zeros(self.nepochs)
        Q_loss = np.zeros(self.nepochs)

        MSE_loss = np.zeros(self.nepochs)
        PHY_loss = np.zeros(self.nepochs)

        Reg_loss_G = np.zeros(self.nepochs)
        Reg_loss_D = np.zeros(self.nepochs)

        model_history = np.zeros((self.nepochs // n_print, 3))
        counter = 0
        for epoch in range(self.nepochs):
            for i, (x, y) in enumerate(self.train_loader):

                Z_u = self.sample_noise(x.shape[0])
                Z_f = self.sample_noise(self.train_x_f.shape[0])

                for j in range(self.k1):
                    self.D_optimizer.zero_grad()
                    d_loss, bce_loss, loss_reg_d = self.train_disriminator(x, y, Z_u)
                    d_loss.backward(retain_graph=True)
                    self.D_optimizer.step()

                # Run the Tensorflow session to minimize the loss
                for i in range(self.k2):
                    self.G_optimizer.zero_grad()
                    g_loss, adv_loss, mse_loss, phy_loss, loss_reg_g, q_loss = self.train_generator(x, y, Z_u, Z_f)
                    g_loss.backward(retain_graph=True)
                    self.G_optimizer.step()
                
                self.D_scheduler.step()
                self.G_scheduler.step()

                Adv_loss[epoch] += adv_loss.detach().cpu().numpy()
                MSE_loss[epoch] += mse_loss.detach().cpu().numpy()
                PHY_loss[epoch] += phy_loss.detach().cpu().numpy()
                G_loss[epoch] += g_loss.detach().cpu().numpy()
                D_loss[epoch] += d_loss.detach().cpu().numpy()
                Q_loss[epoch] += q_loss.detach().cpu().numpy()
                Reg_loss_G[epoch] += loss_reg_g.detach().cpu().numpy()
                Reg_loss_D[epoch] += loss_reg_d.detach().cpu().numpy()

            Adv_loss[epoch] = Adv_loss[epoch] / len(self.train_loader)
            MSE_loss[epoch] = MSE_loss[epoch] / len(self.train_loader)
            PHY_loss[epoch] = PHY_loss[epoch] / len(self.train_loader)
            G_loss[epoch] = G_loss[epoch] / len(self.train_loader)
            D_loss[epoch] = D_loss[epoch] / len(self.train_loader)
            Q_loss[epoch] = Q_loss[epoch] / len(self.train_loader)
            Reg_loss_G[epoch] = Reg_loss_G[epoch] / len(self.train_loader)
            Reg_loss_D[epoch] = Reg_loss_D[epoch] / len(self.train_loader)

            if (epoch % n_print == 0):
                error_u, residual, error_u_train, residual_train = self.average_prediction()
                model_history[counter, 0] = epoch
                model_history[counter, 1] = error_u_train
                model_history[counter, 2] = error_u
                counter += 1
                print(
                    "[Epoch %d/%d] [MSE loss: %f] [PHY loss: %f] [Adv G loss: %f] [Q loss: %f] [G loss: %f] [D loss: %f]"
                    "[Reg loss - G: %f] [Reg loss - D: %f] [Err u_train: %e] [Residual_f train: %e] [Err u_test: %e]"
                    "[Residual_f_test: %e]"
                    % (epoch, self.nepochs, MSE_loss[epoch], PHY_loss[epoch], Adv_loss[epoch], Q_loss[epoch],
                       G_loss[epoch], D_loss[epoch], Reg_loss_G[epoch], Reg_loss_D[epoch],
                       error_u_train, residual_train, error_u, residual)
                )
        return model_history
