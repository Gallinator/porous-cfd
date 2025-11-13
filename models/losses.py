import torch
from lightning import LightningModule
from torch import nn, Tensor
from torch.nn.functional import mse_loss
import lightning as L
from dataset.foam_data import FoamData
from dataset.foam_dataset import StandardScaler, Normalizer


def vector_loss(input: Tensor, target: Tensor, loss_fn) -> Tensor:
    """
    Vectorized loss over multiple dimensions.
    :param input: (B,N,D).
    :param target: (B,N,D).
    :param loss_fn: supports mse_loss, l1_loss.
    :return: (1,D).
    """
    loss = loss_fn(input, target, reduction='none')
    loss = loss.reshape((-1, loss.shape[-1]))
    return torch.mean(loss, dim=-2, keepdim=True).squeeze()


class LossScaler(nn.Module):
    """
    Base loss scaler class.

    Returns unscaled losses by default.
    """

    def forward(self, model: LightningModule, losses: Tensor):
        """
        :param model: The model.
        :param losses: A Tensor of losses of size (N).
        :return: Scaled losses.
        """
        return losses


class FixedLossScaler(LossScaler):
    """
    Applies fixed loss coefficients to the losses. Supported losses names are continuity, momentum, boundary and observations.

    IMPORTANT: the losses must be passed to the forward function in the same order of the loss_weight dictionary.
    """

    def __init__(self, loss_weights: dict[str, list]):
        super().__init__()
        self.weights = loss_weights['continuity']
        self.weights.extend(loss_weights['momentum'])
        self.weights.extend(loss_weights['boundary'])
        if 'observations' in loss_weights:
            self.weights.extend(loss_weights['observations'])
        self.weights = torch.tensor(self.weights, dtype=torch.float)

    def forward(self, model: LightningModule, losses: Tensor):
        return losses * self.weights

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weights = self.weights.to(*args, **kwargs)
        return self


class RelobraloScaler(LossScaler):
    """
    Implements ReLoBRaLo scaling. In this implementation the value of alpha is (1-alpha) with respect to the original paper.
    This implementation averages the losses over each epoch to calculate the weights.

    Adapted from `physics-nemo-sym`_.

    .. _physics-nemo-sym: https://github.com/NVIDIA/physicsnemo-sym/blob/main/physicsnemo/sym/loss/aggregator.py
    """

    def __init__(self, num_losses: int, alpha=0.95, beta=0.99, tau=1.0, eps=1e-8):
        super().__init__()
        self.num_losses = num_losses
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.eps = eps
        self.register_buffer("init_losses", torch.zeros(self.num_losses))
        self.register_buffer("prev_losses", torch.zeros(self.num_losses))
        self.register_buffer("lambda_ema", torch.ones(self.num_losses))
        self.init_loss: torch.Tensor = torch.tensor(0.0)

    def forward(self, model: LightningModule, losses: Tensor):
        """
        Weights and aggregates the losses using the ReLoBRaLo algorithm. The losses order must be consistent throughout the training.
        """
        batch_size = model.trainer.train_dataloader.batch_size

        # Aggregate losses by summation at step 0
        if model.global_step == 0:
            self.init_losses = losses.detach().clone()
            self.prev_losses = losses.detach().clone()
            return losses

        # Aggregate losses using ReLoBRaLo for step > 0
        else:
            if model.global_step % batch_size == 0:
                self.prev_losses = self.prev_losses / batch_size
                normalizer_prev = (losses / (self.tau * self.prev_losses)).max()
                normalizer_init = (losses / (self.tau * self.init_losses)).max()
                rho = torch.bernoulli(torch.tensor(self.beta))
                with torch.no_grad():
                    lambda_prev = torch.exp(losses / (self.tau * self.prev_losses + self.eps) - normalizer_prev)
                    lambda_init = torch.exp(losses / (self.tau * self.init_losses + self.eps) - normalizer_init)
                    lambda_prev *= self.num_losses / (lambda_prev.sum() + self.eps)
                    lambda_init *= self.num_losses / (lambda_init.sum() + self.eps)

                # Compute the exponential moving average of weights and aggregate losses
                with torch.no_grad():
                    self.lambda_ema = self.alpha * (
                            rho * self.lambda_ema.detach().clone() + (1.0 - rho) * lambda_init)
                    self.lambda_ema += (1.0 - self.alpha) * lambda_prev
                self.prev_losses = losses.detach().clone()

                logger = model.training_loss_togger
                model.logger.experiment.add_scalars('Loss weights',
                                                    dict(zip(logger.loss_labels[1:], self.lambda_ema)),
                                                    model.global_step)
            else:
                self.prev_losses += losses.detach().clone()
            return self.lambda_ema.detach().clone() * losses


class LossLogger:
    """
    Utility class to log losses using TensorBoard.

    The loss labels must be passed in the same order they are logged using log().
    """

    def __init__(self, module: L.LightningModule, *loss_labels: str):
        super().__init__()
        self.loss_labels = loss_labels
        self.module = module

    def log(self, batch_size: int, *losses: Tensor):
        """
        Logs the losses.
        """
        if len(losses) != len(self.loss_labels):
            print('Mismatching losses!')
        for label, loss in zip(self.loss_labels, losses):
            self.module.log(label, loss, on_step=False, on_epoch=True, batch_size=batch_size)


class ContinuityLoss(nn.Module):
    """
    Continuity loss for non standardized outputs.
    """

    def func(self, jacobian):
        terms = torch.diagonal(jacobian, 0, -1, -2)
        return torch.sum(terms, dim=-1)

    def forward(self, *args):
        """
        :param args: The jacobian of the velocity.
        :return: The mse loss.
        """
        res = self.func(*args)
        return mse_loss(res, torch.zeros_like(res))


class ContinuityLossStandardized(nn.Module):
    """
    Continuity loss for standardized outputs.
    """

    def __init__(self, u_scaler: StandardScaler, points_scaler: StandardScaler):
        super().__init__()
        self.u_scaler = u_scaler
        self.points_scaler = points_scaler

    def func(self, jacobian):
        """
        See forward().
        """
        terms = torch.diagonal(jacobian, 0, -1, -2) * self.u_scaler.std / self.points_scaler.std
        return torch.sum(terms, dim=-1)

    def forward(self, *args):
        """
        :param args: The jacobian of the velocity.
        :return: The mse loss.
        """
        res = self.func(*args)
        return mse_loss(res, torch.zeros_like(res))


class MomentumLossManufactured(nn.Module):
    """
    Momentum loss for non standardized outputs.
    """

    def __init__(self, nu: float, d: float, f: float):
        """
        :param nu: Kinematic viscosity.
        :param d: Darcy coefficient.
        :param f: Forchheimer coefficient.
        """
        super().__init__()
        self.nu = nu
        self.d = d
        self.f = f

    def func(self, internal_input: FoamData, u: Tensor, u_jac: Tensor, u_laplace: Tensor, p_grad: Tensor):
        """
        See forward().
        """
        source = u * (self.d * self.nu + 1 / 2 * torch.norm(u, dim=-1, keepdim=True) * self.f)
        return (torch.matmul(u_jac, u.unsqueeze(-1)).squeeze() -
                self.nu * torch.matmul(u_laplace, torch.ones_like(u).unsqueeze(-1)).squeeze() +
                p_grad +
                source * internal_input['cellToRegion'] - internal_input['f'])

    def forward(self, *args):
        """
        :param args: Internal domain inputs, output velocity, velocity jacobian, velocity laplace operator, pressure gradient.
        :return: Tensor of shape (D).
        """
        res = self.func(*args)
        return vector_loss(res, torch.zeros_like(res), mse_loss)


class MomentumLossFixed(nn.Module):
    """
    Momentum loss with support for fixed porosity coefficients. Uses standardized outputs.
    """

    def __init__(self,
                 nu: float,
                 d: float,
                 f: float,
                 u_scaler: StandardScaler,
                 points_scaler: StandardScaler,
                 p_scaler: StandardScaler):
        """
        :param nu: Kinematic viscosity.
        :param d: Darcy coefficient.
        :param f: Forchheimer coefficient.
        :param u_scaler: Scaler for the velocity fields.
        :param points_scaler: Scaler for the point coordinates.
        :param p_scaler: Scaler for the pressure field.
        """
        super().__init__()
        self.nu = nu
        self.d = d
        self.f = f
        self.u_scaler = u_scaler
        self.points_scaler = points_scaler
        self.p_scaler = p_scaler

    def func(self, internal_input: FoamData, u: Tensor, u_jac: Tensor, u_laplace: Tensor, p_grad: Tensor):
        """
        See forward().
        """
        u_raw = self.u_scaler.inverse_transform(u)
        source = u_raw * (self.d * self.nu + 1 / 2 * torch.norm(u_raw, dim=-1, keepdim=True) * self.f)
        convection = torch.matmul(u_jac, (u_raw / self.points_scaler.std).unsqueeze(-1)).squeeze() * self.u_scaler.std
        viscosity = (self.nu * torch.matmul(u_laplace, (1 / self.points_scaler.std ** 2).unsqueeze(-1)).squeeze()
                     * self.u_scaler.std)
        pressure = (self.p_scaler.std / self.points_scaler.std) * p_grad
        return convection - viscosity + pressure + source * internal_input['cellToRegion']

    def forward(self, *args):
        pde = self.func(*args)
        return vector_loss(pde, torch.zeros_like(pde), mse_loss)


class MomentumLossVariable(nn.Module):
    """
    Momentum loss with support for variable porosity coefficients. Uses standardized outputs.
    """

    def __init__(self,
                 nu: float,
                 u_scaler: StandardScaler,
                 points_scaler: StandardScaler,
                 p_scaler: StandardScaler,
                 d_scaler: Normalizer,
                 f_scaler: Normalizer):
        """
        :param nu: Kinematic viscosity.
        :param u_scaler: Scaler for the velocity fields.
        :param points_scaler: Scaler for the point coordinates.
        :param p_scaler: Scaler for the pressure field.
        :param d_scaler: Scaler for the Darcy coefficients.
        :param f_scaler: Scaler for the Forchheimer coefficients.
        """
        super().__init__()
        self.nu = nu
        self.u_scaler = u_scaler
        self.points_scaler = points_scaler
        self.p_scaler = p_scaler
        self.d_scaler = d_scaler
        self.f_scaler = f_scaler

    def func(self, internal_input: FoamData, u: Tensor, u_jac: Tensor, u_laplace: Tensor, p_grad: Tensor):
        u_raw = self.u_scaler.inverse_transform(u)
        d_raw = self.d_scaler.inverse_transform(internal_input['d'])
        f_raw = self.f_scaler.inverse_transform(internal_input['f'])

        source = u_raw * (d_raw * self.nu + 1 / 2 * torch.norm(u_raw, dim=-1, keepdim=True) * f_raw)
        convection = torch.matmul(u_jac, (u_raw / self.points_scaler.std).unsqueeze(-1)).squeeze() * self.u_scaler.std
        viscosity = (self.nu * torch.matmul(u_laplace, (1 / self.points_scaler.std ** 2).unsqueeze(-1)).squeeze()
                     * self.u_scaler.std)
        pressure = (self.p_scaler.std / self.points_scaler.std) * p_grad
        return convection - viscosity + pressure + source * internal_input['cellToRegion']

    def forward(self, *args):
        """
        :param args: Internal domain inputs, output velocity, velocity jacobian, velocity laplace operator, pressure gradient.
        :return: Tensor of shape (D).
        """
        pde = self.func(*args)
        return vector_loss(pde, torch.zeros_like(pde), mse_loss)
