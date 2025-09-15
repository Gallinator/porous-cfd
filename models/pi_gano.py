import torch
from torch import nn, Tensor
from torch.nn import Tanh
from torch.optim.lr_scheduler import ExponentialLR
from dataset.foam_dataset import StandardScaler, FoamData, Normalizer
from models.losses import ContinuityLossStandardized, MomentumLossVariable
from models.model_base import PorousPinnBase
from models.modules import GeometryEncoder, Branch, MLP, NeuralOperatorSequential


class PiGano(PorousPinnBase):
    def __init__(self,
                 nu,
                 n_dims,
                 out_features,
                 branch_features,
                 branch_layers,
                 geometry_layers,
                 local_layers,
                 n_operators,
                 operator_dropout,
                 scalers: dict[str, StandardScaler | Normalizer],
                 variable_boundaries: dict[str, list],
                 loss_scaler=None):
        super().__init__(out_features, nu, True, loss_scaler)

        self.branch = Branch(branch_features, branch_layers)
        self.geometry_encoder = GeometryEncoder(n_dims + 1, geometry_layers)
        self.points_encoder = MLP(n_dims, local_layers, Tanh, None)

        operator_features = geometry_layers[-1] + local_layers[-1]
        self.neural_ops = NeuralOperatorSequential(n_operators, operator_features, operator_dropout)
        self.reduction = nn.Linear(operator_features, out_features)

        self.u_scaler = scalers['U']
        self.p_scaler = scalers['p']
        self.points_scaler = scalers['C']
        self.d_scaler = scalers['d']
        self.f_scaler = scalers['f']

        self.continuity_loss = ContinuityLossStandardized(self.u_scaler, self.points_scaler)
        self.momentum_loss = MomentumLossVariable(nu, self.u_scaler, self.points_scaler, self.p_scaler, self.d_scaler,
                                                  self.f_scaler)

        self.variable_boundaries = variable_boundaries
        self.save_hyperparameters()

    def to(self, *args, **kwargs):
        super().to(*args, *kwargs)
        self.u_scaler.to(*args, *kwargs).to(torch.float)
        self.p_scaler.to(*args, *kwargs).to(torch.float)
        self.points_scaler.to(*args, *kwargs).to(torch.float)
        self.d_scaler.to(*args, *kwargs).to(torch.float)
        self.f_scaler.to(*args, *kwargs).to(torch.float)
        return self

    def postprocess_out(self, u, p) -> tuple[Tensor, Tensor]:
        return self.u_scaler.inverse_transform(u), self.p_scaler.inverse_transform(p)

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

    def forward(self, autograd_points: Tensor, x: FoamData) -> FoamData:
        # Prepare inputs
        zones_ids = x['cellToRegion']
        param_features = self.get_parameters(x)

        geom_embedding = self.geometry_encoder.forward(autograd_points.detach(), zones_ids)
        geom_embedding = geom_embedding.repeat((1, autograd_points.shape[-2], 1))

        local_embedding = self.points_encoder.forward(autograd_points)

        operator_input = torch.cat([local_embedding, geom_embedding], dim=-1)
        par_embedding = self.branch.forward(param_features)

        y = self.neural_ops(operator_input, par_embedding)
        y = self.reduction.forward(y)
        return FoamData(y, self.predicted_labels, x.domain)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = ExponentialLR(optimizer, 0.999)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
