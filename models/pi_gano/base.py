import torch
from torch import Tensor
from dataset.foam_data import FoamData
from dataset.foam_dataset import Normalizer, StandardScaler
from models.losses import ContinuityLossStandardized, MomentumLossVariable
from models.model_base import PorousPinnBase


class PiGanoBase(PorousPinnBase):
    def __init__(self, nu,
                 out_features,
                 scalers: dict[str, Normalizer | StandardScaler],
                 loss_scaler,
                 variable_boundaries):
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

    def to(self, *args, **kwargs):
        super().to(*args, *kwargs)
        self.u_scaler.to(*args, *kwargs).to(torch.float)
        self.p_scaler.to(*args, *kwargs).to(torch.float)
        self.points_scaler.to(*args, *kwargs).to(torch.float)
        self.d_scaler.to(*args, *kwargs).to(torch.float)
        self.f_scaler.to(*args, *kwargs).to(torch.float)
        return self

    def get_parameters(self, x: FoamData) -> Tensor:
        """
        Extracts boundary features for each subdomain.
        :param x:
        :return:
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
