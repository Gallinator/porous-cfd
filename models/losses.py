import torch
from torch import nn
from torch.nn.functional import mse_loss
import lightning as L
from dataset.foam_dataset import StandardScaler, Normalizer


class LossLogger:
    def __init__(self, module: L.LightningModule, *loss_labels):
        super().__init__()
        self.loss_labels = loss_labels
        self.module = module

    def log(self, batch_size, *losses):
        if len(losses) != len(self.loss_labels):
            print('Mismatching losses!')
        for label, loss in zip(self.loss_labels, losses):
            self.module.log(label, loss, on_step=False, on_epoch=True, batch_size=batch_size)


class ContinuityLoss(nn.Module):
    def func(self, d_ux_x, d_uy_y):
        return d_ux_x + d_uy_y

    def forward(self, d_ux_x, d_uy_y):
        pde = self.func(d_ux_x, d_uy_y)
        return mse_loss(pde, torch.zeros_like(pde))


class MomentumLossFluid(nn.Module):
    def __init__(self, mu):
        super().__init__()
        self.mu = mu

    def func(self, ui, uj, d_p_i, d_ui_i, d_ui_j, dd_ui_i, dd_ui_j, f_i):
        return d_ui_i * ui + d_ui_j * uj - self.mu * (dd_ui_i + dd_ui_j) + d_p_i - f_i

    def forward(self, *args):
        pde = self.func(*args)
        return mse_loss(pde, torch.zeros_like(pde))


class MomentumLoss2d(nn.Module):
    def __init__(self, mu, d, f):
        super().__init__()
        self.mu = mu
        self.d = d
        self.f = f

    def func(self, ui, uj, d_p_i, zones_ids, fi, d_ui_i, d_ui_j, dd_ui_i, dd_ui_j):
        source = ui * (self.d * self.mu + 1 / 2 * torch.sqrt(ui ** 2 + uj ** 2) * self.f)
        return d_ui_i * ui + d_ui_j * uj - self.mu * (dd_ui_i + dd_ui_j) + d_p_i + source * zones_ids - fi

    def forward(self, *args):
        res = self.func(*args)
        return mse_loss(res, torch.zeros_like(res))


class MomentumLoss2dScaled(nn.Module):
    def __init__(self, i, j, mu, d, f, u_scaler: StandardScaler, points_scaler: StandardScaler,
                 p_scaler: StandardScaler):
        super().__init__()
        self.i = i
        self.j = j
        self.mu = mu
        self.d = d
        self.f = f
        self.u_scaler = u_scaler
        self.points_scaler = points_scaler
        self.p_scaler = p_scaler

    def func(self, ui, uj, d_p_i, zones_ids, d_ui_i, d_ui_j, dd_ui_i, dd_ui_j):
        i, j = self.i, self.j
        norm_d_ui_i = (self.u_scaler.std[i] / self.points_scaler.std[i])
        norm_d_ui_j = (self.u_scaler.std[i] / self.points_scaler.std[j])
        norm_dd_ui_i = norm_d_ui_i * (1 / self.points_scaler.std[i])
        norm_dd_ui_j = norm_d_ui_j * (1 / self.points_scaler.std[j])
        ui_raw = ui * self.u_scaler.std[i] + self.u_scaler.mean[i]
        uj_raw = uj * self.u_scaler.std[j] + self.u_scaler.mean[j]

        source = ui_raw * (self.d * self.mu + 1 / 2 * torch.sqrt(ui_raw ** 2 + uj_raw ** 2) * self.f)

        return (norm_d_ui_i * d_ui_i * ui_raw + norm_d_ui_j * d_ui_j * uj_raw -
                self.mu * (norm_dd_ui_i * dd_ui_i + norm_dd_ui_j * dd_ui_j) +
                (self.p_scaler.std / self.points_scaler.std[i]) * d_p_i + source * zones_ids)

    def forward(self, *args):
        pde = self.func(*args)
        return mse_loss(pde, torch.zeros_like(pde))


class MomentumLoss3dScaled(nn.Module):
    def __init__(self, i, j, k, mu, u_scaler: StandardScaler, points_scaler: StandardScaler, p_scaler: StandardScaler,
                 d_scaler: Normalizer, f_scaler: Normalizer):
        super().__init__()
        self.i = i
        self.j = j
        self.k = k
        self.mu = mu
        self.u_scaler = u_scaler
        self.points_scaler = points_scaler
        self.p_scaler = p_scaler
        self.d_scaler = d_scaler
        self.f_scaler = f_scaler

    def func(self, ui, uj, uk, d_p_i, zones_ids, d, f, d_ui_i, d_ui_j, d_ui_k, dd_ui_i, dd_ui_j, dd_ui_k):
        i, j, k = self.i, self.j, self.k
        norm_d_ui_i = (self.u_scaler.std[i] / self.points_scaler.std[i])
        norm_d_ui_j = (self.u_scaler.std[i] / self.points_scaler.std[j])
        norm_d_ui_k = (self.u_scaler.std[i] / self.points_scaler.std[k])
        norm_dd_ui_i = norm_d_ui_i * (1 / self.points_scaler.std[i])
        norm_dd_ui_j = norm_d_ui_j * (1 / self.points_scaler.std[j])
        norm_dd_ui_k = norm_d_ui_k * (1 / self.points_scaler.std[k])
        ui_raw = ui * self.u_scaler.std[i] + self.u_scaler.mean[i]
        uj_raw = uj * self.u_scaler.std[j] + self.u_scaler.mean[j]
        uk_raw = uk * self.u_scaler.std[k] + self.u_scaler.mean[k]

        d_i = self.d_scaler[i].inverse_transform(d[..., i:i + 1])
        f_i = self.f_scaler[i].inverse_transform(f[..., i:i + 1])

        source = ui_raw * (d_i * self.mu + 1 / 2 * torch.sqrt(ui_raw ** 2 + uj_raw ** 2 + uk_raw ** 2) * f_i)

        return (norm_d_ui_i * d_ui_i * ui_raw + norm_d_ui_j * d_ui_j * uj_raw + norm_d_ui_k * d_ui_k * uk_raw -
                self.mu * (norm_dd_ui_i * dd_ui_i + norm_dd_ui_j * dd_ui_j + norm_dd_ui_k * dd_ui_k) +
                (self.p_scaler.std / self.points_scaler.std[i]) * d_p_i + source * zones_ids)

    def forward(self, *args):
        pde = self.func(*args)
        return mse_loss(pde, torch.zeros_like(pde))


class ContinuityLossScaled(nn.Module):
    def __init__(self, u_scaler: StandardScaler, points_scaler: StandardScaler):
        super().__init__()
        self.u_scaler = u_scaler
        self.points_scaler = points_scaler

    def func(self, *args):
        """
          :param args: Partial derivatives in the following order: x,y,z.
          :return: the continuity residual
          """
        pde = [(self.u_scaler[i].std / self.points_scaler[i].std) * d for i, d in enumerate(args)]
        return torch.sum(torch.stack(pde), dim=0)

    def forward(self, *args):
        """
        :param args: Partial derivatives in the following order: x,y,z.
        :return: the mse loss
        """
        res = self.func(*args)
        return mse_loss(res, torch.zeros_like(res))
