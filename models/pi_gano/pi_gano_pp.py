import torch
from torch import Tensor
from torch.nn import SiLU, Module, Linear
from torch.optim.lr_scheduler import ExponentialLR

from dataset.foam_data import FoamData
from dataset.foam_dataset import Normalizer, StandardScaler
from models.losses import LossScaler
from models.modules import Branch, GeometryEncoderPp, MLP, NeuralOperatorSequential
from models.pi_gano.base import PiGanoBase


class PiGanoPp(PiGanoBase):
    """
    Implementation of the PI-GANO++.

    This model replace the branch network with SetAbstraction layers.
    """

    def __init__(self,
                 nu: float,
                 out_features: int,
                 branch_layers: list[int],
                 geometry_layers: list[list[int]],
                 geometry_radius: list[int],
                 geometry_fraction: list[float],
                 local_layers: list[int],
                 n_operators: int,
                 operator_dropout: list[float],
                 scalers: dict[str, StandardScaler | Normalizer],
                 variable_boundaries: dict[str, list],
                 loss_scaler: LossScaler = None,
                 activation: type[Module] = SiLU,
                 max_neighbors=64):
        """
        :param nu: Kinematic viscosity.
        :param out_features: Number of output features.
        :param branch_layers: List of branch layers sizes.
        :param geometry_layers: List of layers to use for the geometry network. Must contain a list for each Set Abstraction to add.
        :param geometry_radius: List of radii to use in SetAbstraction layers. Set N-1 elements to add a Global Set Abstraction layer at the end.
        :param geometry_fraction: List of fractions to use in the SetAbstraction layers.
        :param local_layers: List of local layers sizes.
        :param n_operators: Number of Neural Operators to use.
        :param operator_dropout: List of dropout probabilities for the Neural Operators.
        :param scalers: Dictionary of output features scalers.
        :param variable_boundaries: Dictionary containing the variable boundaries subdomains and features.
        :param loss_scaler: Scaler applied to the losses.
        :param activation: Activation function.
        :param max_neighbors: Maximum number of neighbours used by the SetAbstraction layers aggregation.
        """
        super().__init__(nu, out_features, scalers, loss_scaler, variable_boundaries)

        self.branch = Branch(branch_layers, activation)
        self.geometry_encoder = GeometryEncoderPp(geometry_fraction, geometry_radius, geometry_layers, activation,
                                                  max_neighbors)
        self.points_encoder = MLP(local_layers, None, activation)

        operator_features = geometry_layers[-1][-1] + local_layers[-1]
        self.neural_ops = NeuralOperatorSequential(n_operators, operator_features, operator_dropout, activation)
        self.reduction = Linear(operator_features, out_features)

    def forward(self, autograd_points: Tensor, x: FoamData) -> FoamData:
        """
        :param autograd_points: Internal points to use for autograd gradients computations. Must be passed through the model.
        :param x: Input features.
        :return: The predicted values.
        """
        # Prepare inputs
        param_features = self.get_parameters(x)

        geom_in = torch.cat([x['boundary']['C'].detach(), x['boundary']['boundaryId']], dim=-1)
        geom_embedding = self.geometry_encoder(geom_in.detach(), x['boundary']['C'].detach())
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
