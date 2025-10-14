import torch
from torch import nn, Tensor
from torch.nn import Tanh, SiLU
from torch.optim.lr_scheduler import ExponentialLR
from dataset.foam_dataset import StandardScaler, FoamData, Normalizer
from models.modules import Branch, NeuralOperatorSequential, MLP, GeometryEncoder
from models.pi_gano.base import PiGanoBase


class PiGano(PiGanoBase):
    def __init__(self,
                 nu,
                 out_features,
                 branch_layers,
                 geometry_layers,
                 local_layers,
                 n_operators,
                 operator_dropout,
                 scalers: dict[str, StandardScaler | Normalizer],
                 variable_boundaries: dict[str, list],
                 loss_scaler=None,
                 activation=SiLU):
        super().__init__(nu, out_features, scalers, loss_scaler, variable_boundaries)

        self.branch = Branch(branch_layers, activation)
        self.geometry_encoder = GeometryEncoder(geometry_layers, activation)
        self.points_encoder = MLP(local_layers, None, activation)

        operator_features = geometry_layers[-1] + local_layers[-1]
        self.neural_ops = NeuralOperatorSequential(n_operators, operator_features, operator_dropout, activation)
        self.reduction = nn.Linear(operator_features, out_features)

    def forward(self, autograd_points: Tensor, x: FoamData) -> FoamData:
        # Prepare inputs
        geom_in = torch.cat([x['boundaryId'], x['sdf']], dim=-1)
        param_features = self.get_parameters(x)

        geom_embedding = self.geometry_encoder.forward(geom_in, autograd_points.detach())
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


class PiGanoFull(PiGano):
    def __init__(self,
                 nu,
                 out_features,
                 branch_layers,
                 geometry_layers,
                 local_layers,
                 n_operators,
                 operator_dropout,
                 scalers: dict[str, StandardScaler | Normalizer],
                 variable_boundaries: dict[str, list],
                 loss_scaler=None,
                 activation=SiLU):
        super().__init__(nu, out_features, branch_layers, geometry_layers, local_layers, n_operators, operator_dropout,
                         scalers, variable_boundaries, loss_scaler, activation)
        operator_features = geometry_layers[-1] + local_layers[-1]

        # This is a hacky solution to allow torch module registration
        self.neural_ops = nn.Sequential(
            *[NeuralOperatorSequential(n_operators, operator_features, operator_dropout, activation) for _
              in range(out_features)])
        self.reduction = None

    def forward(self, autograd_points: Tensor, x: FoamData) -> FoamData:
        # Prepare inputs
        geom_in = torch.cat([x['boundaryId'], x['sdf']], dim=-1)
        param_features = self.get_parameters(x)

        geom_embedding = self.geometry_encoder.forward(geom_in, autograd_points.detach())
        geom_embedding = geom_embedding.repeat((1, autograd_points.shape[-2], 1))

        local_embedding = self.points_encoder.forward(autograd_points)

        operator_input = torch.cat([local_embedding, geom_embedding], dim=-1)
        par_embedding = self.branch.forward(param_features)

        y = [op(operator_input, par_embedding) for op in self.neural_ops]
        y = [f.sum(dim=-1, keepdims=True) for f in y]
        y = torch.cat(y, dim=-1)

        return FoamData(y, self.predicted_labels, x.domain)
