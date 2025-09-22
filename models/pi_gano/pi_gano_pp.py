import torch
from torch import Tensor, nn
from torch.nn import Tanh
from torch.optim.lr_scheduler import ExponentialLR

from dataset.foam_data import FoamData
from dataset.foam_dataset import Normalizer, StandardScaler
from models.modules import Branch, GeometryEncoderPp, BatchedDecorator, MLP, NeuralOperatorSequential
from models.pi_gano.base import PiGanoBase


class PiGanoPp(PiGanoBase):
    def __init__(self,
                 nu,
                 out_features,
                 branch_layers,
                 geometry_layers,
                 geometry_radius,
                 geometry_fraction,
                 local_layers,
                 n_operators,
                 operator_dropout,
                 scalers: dict[str, StandardScaler | Normalizer],
                 variable_boundaries: dict[str, list],
                 loss_scaler=None):
        super().__init__(nu, out_features, scalers, loss_scaler, variable_boundaries)

        self.branch = Branch(branch_layers)
        self.geometry_encoder = GeometryEncoderPp(geometry_fraction, geometry_radius, geometry_layers)
        self.points_encoder = MLP(local_layers, None, Tanh)

        operator_features = geometry_layers[-1][-1] + local_layers[-1]
        self.neural_ops = NeuralOperatorSequential(n_operators, operator_features, operator_dropout)
        self.reduction = nn.Linear(operator_features, out_features)

    def forward(self, autograd_points: Tensor, x: FoamData) -> FoamData:
        # Prepare inputs
        zones_ids = x['cellToRegion']
        param_features = self.get_parameters(x)

        geom_embedding = self.geometry_encoder(zones_ids, autograd_points.detach())
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
