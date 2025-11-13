from typing import Any, Self
import lightning as L
import torch
from torch import Tensor, autograd
from torch.nn.functional import mse_loss, l1_loss

from dataset.foam_data import FoamData
from models.losses import vector_loss, LossLogger


def calculate_gradients(outputs: Tensor, inputs: Tensor) -> Tensor:
    """
    Utility function to calculate gradients retaining the computational graph.
    :param outputs: Output tensor
    :param inputs: Input tensor
    :return: The gradient of input with respect to output.
    """
    return autograd.grad(outputs, inputs,
                         grad_outputs=torch.ones_like(outputs),
                         retain_graph=True, create_graph=True)[0]


def get_jacobian(points: Tensor, u: Tensor):
    """
    Calculates the jacobian matrix. Supports batching.
    :param points: Input points.
    :param u: Output velocity.
    :return: The jacobian of the velocity.
    """
    dims = points.shape[-1]
    jacobian = []
    for d in range(dims):
        u_grad = calculate_gradients(u[..., d: d + 1], points)
        jacobian.append(u_grad)
    return torch.stack(jacobian, dim=-2)


def get_laplacian(points: Tensor, jacobian: Tensor):
    """
    Calculates the Laplace operator using the Jacobian matrix. Supports batching.
    :param points: The input points.
    :param jacobian: The jacobian matrix.
    :return: The Laplace operator.
    """
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
    """
    Enables autograd on the internal domain data.

    This is needed because the internal points are used to compute the physics losses while respecting the computational graph hierarchy.
    :param batch: The input batched data.
    :return: The internal points and the points of the while domain.
    """
    internal_points = batch['internal']['C']
    internal_points.requires_grad = True
    return internal_points, torch.cat([internal_points, batch['boundary']['C']], dim=-2)


class PorousPinnBase(L.LightningModule):
    """
    Base class for porous PINNs.

    This class abstracts the subdomain management and output computation to allow easy model definition.
    This class supports data loss disabling, training and inference functions, loss logging, output postprocessing.
    Setting verbose_predict = True allows to obtain the equation residuals from the predict_step() function at inference time.
    """

    def __init__(self, out_features: int, enable_data_loss=True, loss_scaler=None):
        """
        :param out_features: Total number of outputs. Usually n_dims+1.
        :param enable_data_loss: Pass True to enable the data loss.
        :param loss_scaler: Optional loss scaler.
        """
        super().__init__()
        self.verbose_predict = False
        self.enable_data_loss = enable_data_loss

        # Assume U, p outputs
        self.dims = out_features - 1

        physics_losses_labels = ['Continuity loss',
                                 'Momentum x loss',
                                 'Momentum y loss',
                                 'Momentum z loss'][:out_features]
        observation_losses_labels = ['Observations loss p',
                                     'Observations loss ux',
                                     'Observations loss uy',
                                     'Observations loss uz'][:out_features] if enable_data_loss else []
        boundary_losses_labels = ['Boundary loss p',
                                  'Boundary loss ux',
                                  'Boundary loss uy',
                                  'Boundary loss uz', ][:out_features]
        error_losses = ['error p',
                        'error ux',
                        'error uy',
                        'error uz'][:out_features]

        self.training_loss_togger = LossLogger(self, 'Total loss',
                                               *physics_losses_labels,
                                               *boundary_losses_labels,
                                               *observation_losses_labels,
                                               *[f'Train {l}' for l in error_losses])

        self.val_loss_logger = LossLogger(self, *[f'Validation {l}' for l in error_losses])
        self.predicted_labels = self.get_predicted_labels()
        self.extra_labels = self.get_extra_labels()
        self.loss_scaler = loss_scaler

    def to(self, *args: Any, **kwargs: Any) -> Self:
        """
        Calls to() on the loss scaler object.
        """
        super().to(*args, **kwargs)
        if self.loss_scaler is not None:
            self.loss_scaler = self.loss_scaler.to(*args, **kwargs)
        return self

    def get_predicted_labels(self) -> dict:
        """
        Constructs the labels required by FoamData.
        :return: The labels for the predicted variables.
        """
        u_labels = ['Ux', 'Uy', 'Uz'][:self.dims]
        labels = dict.fromkeys(u_labels, None)
        labels['p'] = None
        labels['U'] = u_labels
        return labels

    def get_extra_labels(self) -> dict:
        """
        Same as get_predicted_labels() but for the residuals fields.
        """
        moment_labels = ['Momentumx', 'Momentumy', 'Momentumz'][:self.dims]
        labels = dict.fromkeys(moment_labels, None)
        labels['div'] = None
        labels['Momentum'] = moment_labels
        return labels

    def postprocess_out(self, u: Tensor, p: Tensor) -> tuple[Tensor, Tensor]:
        """
        This function is applied to the outputs and targets before logging the error metrics.
        Useful if using normalization.
        :param u: Tensor of shape (B,N,D).
        :param p: Tensor of shape (B,N,1).
        :return: u and p in that order.
        """
        return u, p

    def transfer_batch_to_device(self, batch: FoamData, device: torch.device, dataloader_idx: int) -> FoamData:
        """
        Moves a FoamData batch to a device. See DataHooks.transfer_batch_to_device().
        :return: batch on a new device.
        """
        return batch.to(device)

    def calculate_errors(self, target: FoamData, predicted: FoamData) -> tuple[Tensor, Tensor]:
        """
        Utility function to calculate MAEs of all output fields.
        :param target: Ground truth data.
        :param predicted: Predicted data.
        :return: The velocity (D) and pressure (1) error vectors.
        """
        processed_predicted_u, processed_predicted_p = self.postprocess_out(predicted['U'], predicted['p'])
        processed_input_u, processed_input_p = self.postprocess_out(target['U'], target['p'])

        u_error = vector_loss(processed_predicted_u, processed_input_u, l1_loss)
        p_error = l1_loss(processed_predicted_p, processed_input_p)
        return u_error, p_error

    def training_step(self, batch: FoamData, batch_idx: int):
        """
        Main training function, called on each batch of data.

        This function performs: loss calculation. loss scaling, errors calculation, loss and error logging to Tensorboard.
        """
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
            obs_losses = [*obs_u_loss, obs_p_loss]

        losses = torch.stack([continuity_loss, *momentum_loss, *boundary_u_loss, boundary_p_loss, *obs_losses])

        if self.loss_scaler is not None:
            losses = self.loss_scaler(self, losses)

        loss = torch.sum(losses)

        u_error, p_error = self.calculate_errors(batch, predicted)

        self.training_loss_togger.log(len(batch.data), loss, *losses, p_error, *u_error)

        return loss

    def validation_step(self, batch: FoamData):
        """
        Performs a validation step during training with logging to Tensorboard.
        """
        predicted = self.forward(batch['C'], batch)
        u_error, p_error = self.calculate_errors(batch, predicted)
        self.val_loss_logger.log(len(batch.data), p_error, *u_error)

    def predict_step(self, batch: FoamData) -> tuple[FoamData, FoamData] | FoamData:
        """
        Performs a prediction on batch. If verbose_predict is enabled returns the equation residuals along with the predicted output.
        """
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
