import torch
from torch import Tensor
from dataset.foam_data import FoamData
from dataset.foam_dataset import Normalizer, StandardScaler
from models.losses import ContinuityLossStandardized, MomentumLossVariable, LossScaler
from models.model_base import PorousPinnBase


class PiGanoBase(PorousPinnBase):
    """
    Base class for PI-GANO models.

    This model uses outputs scaling and enables the data loss by default. All modules must be defined in the subclasses.
    The scalers must contain the U, p, c, d, anf keys.
    The variable boundaries to use are set in the variable_boundaries dict. It must contain two sub dictionaries:
    Subdomain specifies in which subdomain the boundary conditions are variable, Features sets the variable fields and coefficients.
    """

    def __init__(self,
                 nu: float,
                 out_features: int,
                 scalers: dict[str, Normalizer | StandardScaler],
                 loss_scaler: LossScaler,
                 variable_boundaries: dict[str:list[str]]):
        """
        :param nu: Kinematic viscosity.
        :param out_features: Number of output features.
        :param scalers: Dictionary of output features scalers.
        :param loss_scaler: Scaler applied to the losses.
        :param variable_boundaries: Dictionary containing the variable boundaries subdomains and features.
        """
        super().__init__(out_features, True, loss_scaler)
        self.save_hyperparameters()

        self.u_scaler = scalers['U']
        self.p_scaler = scalers['p']
        self.points_scaler = scalers['C']
        self.d_scaler = scalers['d']
        self.f_scaler = scalers['f']

        self.continuity_loss = ContinuityLossStandardized(self.u_scaler, self.points_scaler)
        self.momentum_loss = MomentumLossVariable(nu,
                                                  self.u_scaler,
                                                  self.points_scaler,
                                                  self.p_scaler,
                                                  self.d_scaler,
                                                  self.f_scaler)

        self.variable_boundaries = variable_boundaries

    def to(self, *args, **kwargs) -> 'PiGanoBase':
        super().to(*args, *kwargs)
        self.u_scaler.to(*args, *kwargs).to(torch.float)
        self.p_scaler.to(*args, *kwargs).to(torch.float)
        self.points_scaler.to(*args, *kwargs).to(torch.float)
        self.d_scaler.to(*args, *kwargs).to(torch.float)
        self.f_scaler.to(*args, *kwargs).to(torch.float)
        return self

    def get_parameters(self, x: FoamData) -> Tensor:
        """
        Extracts variable boundary features for each subdomain.
        :param x: Input data.
        :return: A tensor of shape (B,N,F) where F is the number of variable boundary features.
        """
        param_data = []
        for subdomain in self.variable_boundaries['Subdomains']:
            # Always use points coordinates
            boundary_data = [x[subdomain]['C']]
            for feature in self.variable_boundaries['Features']:
                boundary_data.append(x[subdomain][feature])
            param_data.append(torch.cat(boundary_data, dim=-1))
        return torch.cat(param_data, dim=-2)

    def postprocess_out(self, u, p) -> tuple[Tensor, Tensor]:
        return self.u_scaler.inverse_transform(u), self.p_scaler.inverse_transform(p)
