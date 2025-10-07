import torch
from lightning import LightningModule
from torch import nn, Tensor
from torch.nn.functional import mse_loss
import lightning as L
from dataset.foam_data import FoamData
from dataset.foam_dataset import StandardScaler, Normalizer


def vector_loss(input: Tensor, target: Tensor, loss_fn) -> Tensor:
    """
    Vectorized mse loss
    :param input: (B,N,D)
    :param target: (B,N,D)
    :param loss_fn: supports mse_loss, l1_loss
    :return: (1,D)
    """
    loss = loss_fn(input, target, reduction='none')
    loss = loss.reshape((-1, loss.shape[-1]))
    return torch.mean(loss, dim=-2, keepdim=True).squeeze()


class LossScaler(nn.Module):
    def forward(self, model: LightningModule, losses: Tensor):
        return losses


class FixedLossScaler(LossScaler):
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
    """
    Continuity loss.
    """

    def func(self, jacobian):
        terms = torch.diagonal(jacobian, 0, -1, -2)
        return torch.sum(terms, dim=-1)

    def forward(self, *args):
        """
        :param args: Partial derivatives in the following order: x,y,z.
        :return: the mse loss
        """
        res = self.func(*args)
        return mse_loss(res, torch.zeros_like(res))


class ContinuityLossStandardized(nn.Module):
    """
    Continuity loss that supports standardized outputs.
    """

    def __init__(self, u_scaler: StandardScaler, points_scaler: StandardScaler):
        super().__init__()
        self.u_scaler = u_scaler
        self.points_scaler = points_scaler

    def func(self, jacobian):
        terms = torch.diagonal(jacobian, 0, -1, -2) * self.u_scaler.std / self.points_scaler.std
        return torch.sum(terms, dim=-1)

    def forward(self, *args):
        """
        :param args: Partial derivatives in the following order: x,y,z.
        :return: the mse loss
        """
        res = self.func(*args)
        return mse_loss(res, torch.zeros_like(res))


class MomentumLossManufactured(nn.Module):
    """
    Momentum loss for manufactures solutions
    """

    def __init__(self, nu, d, f):
        super().__init__()
        self.mu = nu
        self.d = d
        self.f = f

    def func(self, internal_input: FoamData, u: Tensor, u_jac: Tensor, u_laplace: Tensor, p_grad: Tensor):
        source = u * (self.d * self.mu + 1 / 2 * torch.norm(u, dim=-1, keepdim=True) * self.f)
        return (torch.matmul(u_jac, u.unsqueeze(-1)).squeeze() -
                self.mu * torch.matmul(u_laplace, torch.ones_like(u).unsqueeze(-1)).squeeze() +
                p_grad +
                source * internal_input['cellToRegion'] - internal_input['f'])

    def forward(self, *args):
        res = self.func(*args)
        return vector_loss(res, torch.zeros_like(res), mse_loss)


class MomentumLossFixed(nn.Module):
    """
    Momentum loss with support for fixed porosity coefficients. Uses standardized outputs
    """

    def __init__(self, nu, d, f, u_scaler: StandardScaler, points_scaler: StandardScaler, p_scaler: StandardScaler):
        super().__init__()
        self.nu = nu
        self.d = d
        self.f = f
        self.u_scaler = u_scaler
        self.points_scaler = points_scaler
        self.p_scaler = p_scaler

    def func(self, internal_input: FoamData, u: Tensor, u_jac: Tensor, u_laplace: Tensor, p_grad: Tensor):
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
    Momentum loss with support for variable porosity coefficients. Uses standardized outputs
    """

    def __init__(self,
                 mu,
                 u_scaler: StandardScaler,
                 points_scaler: StandardScaler,
                 p_scaler: StandardScaler,
                 d_scaler: Normalizer,
                 f_scaler: Normalizer):
        super().__init__()
        self.mu = mu
        self.u_scaler = u_scaler
        self.points_scaler = points_scaler
        self.p_scaler = p_scaler
        self.d_scaler = d_scaler
        self.f_scaler = f_scaler

    def func(self, internal_input: FoamData, u: Tensor, u_jac: Tensor, u_laplace: Tensor, p_grad: Tensor):
        u_raw = self.u_scaler.inverse_transform(u)
        d_raw = self.d_scaler.inverse_transform(internal_input['d'])
        f_raw = self.f_scaler.inverse_transform(internal_input['f'])

        source = u_raw * (d_raw * self.mu + 1 / 2 * torch.norm(u_raw, dim=-1, keepdim=True) * f_raw)
        convection = torch.matmul(u_jac, (u_raw / self.points_scaler.std).unsqueeze(-1)).squeeze() * self.u_scaler.std
        viscosity = (self.mu * torch.matmul(u_laplace, (1 / self.points_scaler.std ** 2).unsqueeze(-1)).squeeze()
                     * self.u_scaler.std)
        pressure = (self.p_scaler.std / self.points_scaler.std) * p_grad
        return convection - viscosity + pressure + source * internal_input['cellToRegion']

    def forward(self, *args):
        pde = self.func(*args)
        return vector_loss(pde, torch.zeros_like(pde), mse_loss)
