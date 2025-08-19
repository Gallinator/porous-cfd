import torch
from torch import nn, Tensor
from torch.nn.functional import mse_loss
import lightning as L


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
    def __init__(self, mu, d, f, n_internal):
        super().__init__()
        self.mu = mu
        self.d = d
        self.f = f
        self.n_internal = n_internal

    def forward(self, ui, d_ui_i, d_ui_j, uj, dd_ui_i, dd_ui_j, d_p_i, fi, zones_ids):
        source = ui * (self.d * self.mu + 1 / 2 * torch.sqrt(ui ** 2 + uj ** 2) * self.f)
        res = d_ui_i * ui + d_ui_j * uj - self.mu * (dd_ui_i + dd_ui_j) + d_p_i + source * zones_ids - fi
        res = res[:, :self.n_internal, :]
        return mse_loss(res, torch.zeros_like(res))


class ContinuityLoss(nn.Module):
    def __init__(self, n_internal):
        super().__init__()
        self.n_internal = n_internal

    def forward(self, d_ux_x, d_uy_y):
        res = d_ux_x + d_uy_y
        res = res[:, :self.n_internal, :]
        return mse_loss(res, torch.zeros_like(res))


class BoundaryLoss(nn.Module):
    def __init__(self, n_internal: int):
        super().__init__()
        self.n_internal = n_internal

    def forward(self, input: Tensor, target: Tensor):
        return mse_loss(input[:, self.n_internal:, :], target[:, self.n_internal:, :])
