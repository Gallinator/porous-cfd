import torch
from torch import nn
from torch.nn.functional import mse_loss
import lightning as L


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
    def func(self, d_ui_i, d_uj_j):
        return d_ui_i + d_uj_j

    def forward(self, d_ui_i, d_uj_j):
        pde = self.func(d_ui_i, d_uj_j)
        return mse_loss(pde, torch.zeros_like(pde))


class MomentumLoss(nn.Module):
    def __init__(self, nu):
        super().__init__()
        self.mu = nu

    def func(self, ui, uj, d_p_i, d_ui_i, d_ui_j, dd_ui_i, dd_ui_j):
        return d_ui_i * ui + d_ui_j * uj - self.mu * (dd_ui_i + dd_ui_j) + d_p_i

    def forward(self, *args):
        pde = self.func(*args)
        return mse_loss(pde, torch.zeros_like(pde))
