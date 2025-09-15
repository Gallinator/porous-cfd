from typing import Any, Self
import lightning as L
import torch
from torch import Tensor, autograd
from torch.nn.functional import mse_loss, l1_loss

from dataset.foam_data import FoamData
from models.losses import vector_loss, LossLogger


def calculate_gradients(outputs: Tensor, inputs: Tensor) -> Tensor:
    return autograd.grad(outputs, inputs,
                         grad_outputs=torch.ones_like(outputs),
                         retain_graph=True, create_graph=True)[0]


def get_jacobian(points: Tensor, u: Tensor):
    dims = points.shape[-1]
    jacobian = []
    for d in range(dims):
        u_grad = calculate_gradients(u[..., d: d + 1], points)
        jacobian.append(u_grad)
    return torch.stack(jacobian, dim=-2)


def get_laplacian(points: Tensor, jacobian: Tensor):
    dims = points.shape[-1]
    laplacian = []
    for i in range(dims):
        dd_u_i = []
        for j in range(dims):
            d_u_ij = jacobian[..., i:i + 1, j]
            dd_u_i.append(calculate_gradients(d_u_ij, points)[..., j:j + 1])
        laplacian.append(torch.concatenate(dd_u_i, -1))
    return torch.stack(laplacian, dim=-2)


def enable_internal_autograd(batch: FoamData) -> tuple[Tensor, Tensor]:
    internal_points = batch['internal']['C']
    internal_points.requires_grad = True
    return internal_points, torch.cat([internal_points, batch['boundary']['C']], dim=-2)


class PorousPinnBase(L.LightningModule):
    def __init__(self, out_features, nu, enable_data_loss=True, loss_scaler=None):
        super().__init__()
        self.nu = nu
        self.verbose_predict = False
        self.enable_data_loss = enable_data_loss

        # Assume U, p outputs
        self.dims = out_features - 1

        physics_losses_labels = ['Train loss continuity',
                                 'Train loss momentum x',
                                 'Train loss momentum y',
                                 'Train loss momentum z'][:out_features]
        observation_losses_labels = ['Obs loss p',
                                     'Obs loss ux',
                                     'Obs loss uy',
                                     'Obs loss uz'][:out_features] if enable_data_loss else []
        error_losses = ['p error',
                        'ux error',
                        'uy error',
                        'uz error'][:out_features]

        self.training_loss_togger = LossLogger(self, 'Train loss',
                                               *physics_losses_labels,
                                               'Train loss p',
                                               'Train loss ux',
                                               'Train loss uy',
                                               *observation_losses_labels,
                                               *[f'Train {l}' for l in error_losses])

        self.val_loss_logger = LossLogger(self, *[f'Val {l}' for l in error_losses])
        self.predicted_labels = self.get_predicted_labels()
        self.extra_labels = self.get_extra_labels()
        self.loss_scaler = loss_scaler

    def to(self, *args: Any, **kwargs: Any) -> Self:
        super().to(*args, **kwargs)
        if self.loss_scaler is not None:
            self.loss_scaler = self.loss_scaler.to(*args, **kwargs)
        return self

    def get_predicted_labels(self) -> dict:
        u_labels = ['Ux', 'Uy', 'Uz'][:self.dims]
        labels = dict.fromkeys(u_labels, None)
        labels['p'] = None
        labels['U'] = u_labels
        return labels

    def get_extra_labels(self) -> dict:
        moment_labels = ['Momentumx', 'Momentumy', 'Momentumz'][:self.dims]
        labels = dict.fromkeys(moment_labels, None)
        labels['div'] = None
        labels['Momentum'] = moment_labels
        return labels

    def postprocess_out(self, u, p) -> tuple[Tensor, Tensor]:
        """
        This function is applied to the outputs and targets before logging the error metrics.
        Useful if using normalization.
        :param u:
        :param p:
        :return: u and p in that order
        """
        return u, p

    def transfer_batch_to_device(self, batch: FoamData, device: torch.device, dataloader_idx: int) -> FoamData:
        dev_data = batch.data.to(device)
        dev_domain = {d: s.to(device) for d, s in batch.domain.items()}
        return FoamData(dev_data, batch.labels, dev_domain)

    def calculate_errors(self, input: FoamData, predicted: FoamData) -> tuple[Tensor, Tensor]:
        processed_predicted_u, processed_predicted_p = self.postprocess_out(predicted['U'], predicted['p'])
        processed_input_u, processed_input_p = self.postprocess_out(input['U'], input['p'])

        u_error = vector_loss(processed_predicted_u, processed_input_u, l1_loss)
        p_error = l1_loss(processed_predicted_p, processed_input_p)
        return u_error, p_error

    def training_step(self, batch: FoamData, batch_idx: int):
        internal_grad_points, all_points = enable_internal_autograd(batch)
        predicted = self.forward(all_points, batch)

        boundary_p_loss = mse_loss(predicted['boundary']['p'], batch['boundary']['p'])
        boundary_u_loss = vector_loss(predicted['boundary']['U'], batch['boundary']['U'], mse_loss)

        jacobian = get_jacobian(internal_grad_points, predicted['internal']['U'])
        laplacian = get_laplacian(internal_grad_points, predicted['internal']['U'])
        d_p = calculate_gradients(predicted['internal']['p'], internal_grad_points)

        continuity_loss = self.continuity_loss(jacobian)
        momentum_loss = self.momentum_loss(batch['internal'], predicted['internal']['U'], jacobian, laplacian, d_p)

        obs_losses = []
        if self.enable_data_loss:
            obs_u_loss = vector_loss(predicted['obs']['U'], batch['obs']['U'], mse_loss)
            obs_p_loss = mse_loss(predicted['obs']['p'], batch['obs']['p'])
            obs_losses = [obs_p_loss, *obs_u_loss]

        losses = torch.stack([continuity_loss, *momentum_loss, *boundary_u_loss, boundary_p_loss, *obs_losses])

        if self.loss_scaler is not None:
            losses = self.loss_scaler(self, losses)

        loss = torch.sum(losses)

        u_error, p_error = self.calculate_errors(batch, predicted)

        self.training_loss_togger.log(len(batch.data), loss, *losses, p_error, *u_error)

        return loss

    def validation_step(self, batch: FoamData):
        predicted = self.forward(batch['C'], batch)
        u_error, p_error = self.calculate_errors(batch, predicted)
        self.val_loss_logger.log(len(batch.data), p_error, *u_error)

    def predict_step(self, batch: FoamData) -> tuple[FoamData, FoamData] | FoamData:
        if self.verbose_predict:
            torch.set_grad_enabled(True)

            internal_grad_points, all_points = enable_internal_autograd(batch)
            predicted = self.forward(all_points, batch)

            jacobian = get_jacobian(internal_grad_points, predicted['internal']['U'])
            laplacian = get_laplacian(internal_grad_points, predicted['internal']['U'])
            d_p = calculate_gradients(predicted['internal']['p'], internal_grad_points)

            div = self.continuity_loss.func(jacobian)
            momentum_error = self.momentum_loss.func(batch['internal'],
                                                     predicted['internal']['U'],
                                                     jacobian,
                                                     laplacian,
                                                     d_p)

            torch.set_grad_enabled(False)
            residuals = torch.cat([momentum_error, div.unsqueeze(-1)], dim=-1)

            return predicted, FoamData(residuals, self.extra_labels, batch.domain)
        else:
            return self.forward(batch['C'], batch)
