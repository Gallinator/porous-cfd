import torch
from torch import nn, Tensor, Module
from torch.nn import SiLU
from torch.optim.lr_scheduler import ExponentialLR
from dataset.foam_dataset import StandardScaler, FoamData, Normalizer
from models.losses import LossScaler
from models.modules import Branch, NeuralOperatorSequential, MLP, GeometryEncoder
from models.pi_gano.base import PiGanoBase


class PiGano(PiGanoBase):
    """PI-GANO implementation. See PiGanoBase for further details."""

    def __init__(self,
                 nu: float,
                 out_features: int,
                 branch_layers: list[int],
                 geometry_layers: list[int],
                 local_layers: list[int],
                 n_operators: int,
                 operator_dropout: list[float],
                 scalers: dict[str, StandardScaler | Normalizer],
                 variable_boundaries: dict[str, list],
                 loss_scaler: LossScaler = None,
                 activation: type[Module] = SiLU):
        """
        :param nu: Kinematic viscosity.
        :param out_features: Number of output features.
        :param scalers: Dictionary of output features scalers.
        :param loss_scaler: Scaler applied to the losses.
        :param variable_boundaries: Dictionary containing the variable boundaries subdomains and features.
        :param branch_layers: List of branch layers sizes.
        :param geometry_layers: List of geometry layers sizes.
        :param local_layers: List of local layers sizes.
        :param n_operators: Number of Neural Operators to use.
        :param operator_dropout: List of dropout probabilities for the Neural Operators.
        :param activation: Activation function.
        """
        super().__init__(nu, out_features, scalers, loss_scaler, variable_boundaries)

        self.branch = Branch(branch_layers, activation)
        self.geometry_encoder = GeometryEncoder(geometry_layers, activation)
        self.points_encoder = MLP(local_layers, None, activation)

        operator_features = geometry_layers[-1] + local_layers[-1]
        self.neural_ops = NeuralOperatorSequential(n_operators, operator_features, operator_dropout, activation)
        self.reduction = nn.Linear(operator_features, out_features)

    def forward(self, autograd_points: Tensor, x: FoamData) -> FoamData:
        """
        :param autograd_points: Internal points to use for autograd gradients computations. Must be passed through the model.
        :param x: Input features.
        :return: The predicted values.
        """
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
    """
    Implementation of the original PI-GANO.

    This model uses a branch of Neural Operators for each output variable.
    """

    def __init__(self,
                 nu: float,
                 out_features: int,
                 branch_layers: list[int],
                 geometry_layers: list[int],
                 local_layers: list[int],
                 n_operators: int,
                 operator_dropout: list[float],
                 scalers: dict[str, StandardScaler | Normalizer],
                 variable_boundaries: dict[str, list],
                 loss_scaler: LossScaler = None,
                 activation: type[Module] = SiLU):
        """
        :param nu: Kinematic viscosity.
        :param out_features: Number of output features.
        :param scalers: Dictionary of output features scalers.
        :param loss_scaler: Scaler applied to the losses.
        :param variable_boundaries: Dictionary containing the variable boundaries subdomains and features.
        :param branch_layers: List of branch layers sizes.
        :param geometry_layers: List of geometry layers sizes.
        :param local_layers: List of local layers sizes.
        :param n_operators: Number of Neural Operators to use.
        :param operator_dropout: List of dropout probabilities for the Neural Operators.
        :param activation: Activation function.
        """
        super().__init__(nu, out_features, branch_layers, geometry_layers, local_layers, n_operators, operator_dropout,
                         scalers, variable_boundaries, loss_scaler, activation)
        operator_features = geometry_layers[-1] + local_layers[-1]

        # This is a hacky solution to allow torch module registration
        self.neural_ops = nn.Sequential(
            *[NeuralOperatorSequential(n_operators, operator_features, operator_dropout, activation, False) for _
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
        y = [torch.sum(f, dim=-1, keepdim=True) for f in y]
        y = torch.cat(y, dim=-1)

        return FoamData(y, self.predicted_labels, x.domain)
