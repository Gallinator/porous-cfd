import torch
from torch import nn, Tensor
from torch.nn.functional import mse_loss
import lightning as L
from dataset.foam_dataset import StandardScaler, Normalizer


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
    def __init__(self, i, j, k, mu,
                 u_scaler: StandardScaler, points_scaler: StandardScaler, p_scaler: StandardScaler,
                 d_scaler: Normalizer, f_scaler: Normalizer):
        super().__init__()
        self.mu = mu
        self.u_scaler = u_scaler
        self.points_scaler = points_scaler
        self.p_stats = p_scaler
        self.d_scaler = d_scaler
        self.f_scaler = f_scaler
        self.i = i
        self.j = j
        self.k = k

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
                (self.p_stats.std / self.points_scaler.std[i]) * d_p_i + source * zones_ids)

    def forward(self, *args):
        res = self.func(*args)
        return mse_loss(res, torch.zeros_like(res))


class ContinuityLoss(nn.Module):
    def __init__(self, u_scaler: StandardScaler, points_scaler: StandardScaler):
        super().__init__()
        self.u_scaler = u_scaler
        self.points_scaler = points_scaler

    def func(self, d_ux_x, d_uy_y, d_uz_z):
        return ((self.u_scaler[0].std / self.points_scaler[0].std) * d_ux_x +
                (self.u_scaler[1].std / self.points_scaler[1].std) * d_uy_y +
                (self.u_scaler[2].std / self.points_scaler[2].std) * d_uz_z)

    def forward(self, *args):
        res = self.func(*args)
        return mse_loss(res, torch.zeros_like(res))
