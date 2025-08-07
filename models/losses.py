import torch
from torch import nn, Tensor
from torch.nn.functional import mse_loss
import lightning as L

from foam_dataset import StandardScaler


class LossLogger:
    def __init__(self, module: L.LightningModule, *loss_labels):
        super().__init__()
        self.loss_labels = loss_labels
        self.module = module

    def log(self, *losses):
        if len(losses) != len(self.loss_labels):
            print('Mismatching losses!')
        for label, loss in zip(self.loss_labels, losses):
            self.module.log(label, loss, on_step=False, on_epoch=True)


class MomentumLoss(nn.Module):
    def __init__(self, i: int, j: int, mu, d, f, n_internal,
                 u_scaler: StandardScaler, points_scaler: StandardScaler, p_scaler: StandardScaler):
        super().__init__()
        self.mu = mu
        self.d = d
        self.f = f
        self.n_internal = n_internal
        self.u_scaler = u_scaler
        self.points_scaler = points_scaler
        self.p_stats = p_scaler
        self.i = i
        self.j = j

    def func(self, ui, uj, d_p_i, zones_ids, d_ui_i, d_ui_j, dd_ui_i, dd_ui_j):
        i, j = self.i, self.j
        norm_d_ui_i = (self.u_scaler.std[i] / self.points_scaler.std[i])
        norm_d_ui_j = (self.u_scaler.std[i] / self.points_scaler.std[j])
        norm_dd_ui_i = norm_d_ui_i * (1 / self.points_scaler.std[i])
        norm_dd_ui_j = norm_d_ui_j * (1 / self.points_scaler.std[j])
        ui_raw = ui * self.u_scaler.std[i] + self.u_scaler.mean[i]
        uj_raw = uj * self.u_scaler.std[j] + self.u_scaler.mean[j]

        source = ui_raw * (self.d * self.mu * zones_ids + 1 / 2 * torch.sqrt(ui ** 2 + uj ** 2) * self.f)

        return (norm_d_ui_i * d_ui_i * ui_raw + norm_d_ui_j * d_ui_j * uj_raw -
                self.mu * (norm_dd_ui_i * dd_ui_i + norm_dd_ui_j * dd_ui_j) +
                (self.p_stats.std / self.points_scaler.std[i]) * d_p_i + source)

    def forward(self, *args):
        res = self.func(*args)
        res = res[:, :self.n_internal, :]
        return mse_loss(res, torch.zeros_like(res))


class ContinuityLoss(nn.Module):
    def __init__(self, n_internal, u_scaler: StandardScaler, points_scaler: StandardScaler):
        super().__init__()
        self.n_internal = n_internal
        self.u_scaler = u_scaler
        self.points_scaler = points_scaler

    def f(self, d_ux_x, d_uy_y):
        return ((self.u_scaler[0].std / self.points_scaler[0].std) * d_ux_x +
                (self.u_scaler[1].std / self.points_scaler[1].std) * d_uy_y)

    def forward(self, *args):
        res = self.f(*args)
        res = res[:, :self.n_internal, :]
        return mse_loss(res, torch.zeros_like(res))


class BoundaryLoss(nn.Module):
    def __init__(self, n_internal: int):
        super().__init__()
        self.n_internal = n_internal

    def forward(self, input: Tensor, target: Tensor):
        return mse_loss(input[:, self.n_internal:, :], target[:, self.n_internal:, :])
