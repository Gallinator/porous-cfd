import torch
from torch import Tensor
from torch.nn import Tanh, Mish, Module
from torch.optim.lr_scheduler import ExponentialLR
from torch_geometric.utils import unbatch

from dataset.foam_data import FoamData
from dataset.foam_dataset import StandardScaler
from models.losses import ContinuityLossStandardized, MomentumLossFixed, LossScaler
from models.model_base import PorousPinnBase
from models.modules import FeaturePropagationSeq, SetAbstractionSeq, \
    get_batch, SetAbstractionMrgSeq, MLP, PointNetFeatureExtractPp, PointNetFeatureExtract


class PipnFoamBase(PorousPinnBase):
    """
    Base class for the modified PIPN with support for mixed porous-fluid medium, feature scaling and dropout.

    The scalers dictionary must contain the U, p and C keys. The data loss is enabled by default.
    """

    def __init__(self,
                 nu: float,
                 d: float,
                 f: float,
                 out_features: int,
                 scalers: dict[str, StandardScaler],
                 loss_scaler: LossScaler = None):
        """
        :param nu: Kinematic viscosity.
        :param d: Darcy coefficient.
        :param f: Forchheimer coefficient.
        :param out_features: Number of output features.
        :param scalers: Feature scaling dictionary.
        :param loss_scaler: Optional loss scaler.
        """
        super().__init__(out_features, True, loss_scaler)
        self.save_hyperparameters()

        self.u_scaler = scalers['U']
        self.p_scaler = scalers['p']
        self.points_scaler = scalers['C']

        self.momentum_loss = MomentumLossFixed(nu, d, f, self.u_scaler, self.points_scaler, self.p_scaler)
        self.continuity_loss = ContinuityLossStandardized(self.u_scaler, self.points_scaler)

    def to(self, *args, **kwargs):
        """Calls to() on loss scaler and feature scaler objects."""
        super().to(*args, *kwargs)
        self.u_scaler.to(*args, *kwargs).to(torch.float)
        self.p_scaler.to(*args, *kwargs).to(torch.float)
        self.points_scaler.to(*args, *kwargs).to(torch.float)
        return self

    def postprocess_out(self, u, p) -> tuple[Tensor, Tensor]:
        return self.u_scaler.inverse_transform(u), self.p_scaler.inverse_transform(p)


class PipnFoam(PipnFoamBase):
    def __init__(self,
                 nu: float,
                 d: float,
                 f: float,
                 fe_local_layers: list[int],
                 fe_global_layers: list[int],
                 seg_layers: list[int],
                 scalers: dict[str, StandardScaler],
                 loss_scaler: LossScaler = None,
                 seg_dropout: list[float] = None,
                 activation: type[Module] = Tanh):
        """
        :param nu: Kinematic viscosity.
        :param d: Darcy coefficient.
        :param f: Forchheimer coefficient.
        :param fe_local_layers: List of sizes for each layer in the first shared MLP.
        :param fe_global_layers: List of sizes for each layer in MLP before the max pooling.
        :param seg_layers: List of sizes for each layer in the final shared MLP.
        :param scalers: Feature scaling dictionary.
        :param loss_scaler: Optional loss scaler.
        :param seg_dropout: List of dropout values to use in the last shared MLP.
        :param activation: Activation function.
        """
        super().__init__(nu, d, f, seg_layers[-1], scalers, loss_scaler)
        self.feature_extract = PointNetFeatureExtract(fe_local_layers, fe_global_layers, activation)
        self.decoder = MLP(seg_layers, seg_dropout, activation, False)

    def forward(self, autograd_points: Tensor, x: FoamData) -> FoamData:
        """
           :param autograd_points: Internal points to use for autograd gradients computations. Must be passed through the model.
           :param x: Input features.
           :return: The predicted values.
        """
        global_in = torch.cat([x['boundaryId'], x['sdf']], dim=-1)
        local_features, global_feature = self.feature_extract(global_in, autograd_points)

        exp_global = global_feature.repeat(1, local_features.shape[-2], 1)
        seg_input = torch.cat([local_features, exp_global], dim=-1)
        y = self.decoder(seg_input)

        return FoamData(y, self.predicted_labels, x.domain)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = ExponentialLR(optimizer, 0.999)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]


class PipnFoamPp(PipnFoamBase):
    """
    PIPN++ implementation with support to feature scaling and dropout.
    """

    def __init__(self,
                 nu: float,
                 d: float,
                 f: float,
                 fe_local_layers: list[int],
                 fe_global_layers: list[list[int]],
                 fe_radius,
                 fe_fraction,
                 seg_layers: list[int],
                 scalers: dict[str, StandardScaler],
                 loss_scaler: LossScaler = None,
                 seg_dropout: list[float] = None,
                 activation: type[Module] = Tanh,
                 max_neighbors=64):
        """
        :param nu: Kinematic viscosity.
        :param d: Darcy coefficient.
        :param f: Forchheimer coefficient.
        :param fe_local_layers: List of sizes for each layer in the first shared MLP.
        :param fe_global_layers: List of sizes for each layer in the geometry encoding. Must contain a list of layers for each Set Abstraction layer.
        :param fe_radius: List of radii to use in SetAbstraction layers. Set N-1 elements to add a Global Set Abstraction layer at the end.
        :param fe_fraction: List of fractions to use in the SetAbstraction layers.
        :param seg_layers: List of sizes for each layer in the final shared MLP.
        :param scalers: Feature scaling dictionary.
        :param loss_scaler: Optional loss scaler.
        :param seg_dropout: List of dropout values to use in the last shared MLP.
        :param activation: Activation function.
        :param max_neighbors: Maximum number of neighbours used by the SetAbstraction layers aggregation.
        """
        super().__init__(nu, d, f, seg_layers[-1], scalers, loss_scaler)

        self.feature_extract = PointNetFeatureExtractPp(fe_local_layers, fe_global_layers, fe_fraction, fe_radius,
                                                        activation, max_neighbors)
        self.decoder = MLP(seg_layers, seg_dropout, activation, False)

    def forward(self, autograd_points: Tensor, x: FoamData) -> FoamData:
        """
           :param autograd_points: Internal points to use for autograd gradients computations. Must be passed through the model.
           :param x: Input features.
           :return: The predicted values.
        """
        geom_features = torch.cat([x['boundary']['C'], x['boundary']['boundaryId']], dim=-1)
        local_features, global_feature = self.feature_extract(geom_features, x['boundary']['C'], autograd_points)

        exp_global = global_feature.repeat(1, local_features.shape[-2], 1)
        seg_input = torch.cat([local_features, exp_global], dim=-1)
        y = self.decoder(seg_input)

        return FoamData(y, self.predicted_labels, x.domain)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = ExponentialLR(optimizer, 0.999)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]


class PipnFoamPpMrg(PipnFoamBase):
    """
    PIPN++ MRG implementation with support to feature scaling, dropout and Multi Resolution Grouping.
    """

    def __init__(self,
                 n_dims: int,
                 mrg_in_features: int,
                 nu: float,
                 d: float,
                 f: float,
                 fe_local_layers: list[int],
                 seg_layers: list[int],
                 scalers: dict[str, StandardScaler],
                 loss_scaler: LossScaler = None,
                 seg_dropout: list[float] = None,
                 activation: type[Module] = Mish,
                 max_neighbors=64):
        """
        :param n_dims: Coordinates dimensionality.
        :param mrg_in_features: Number of input feature of the MRG geometry encoder.
        :param nu: Kinematic viscosity.
        :param d: Darcy coefficient.
        :param f: Forchheimer coefficient.
        :param fe_local_layers: List of sizes for each layer in the first shared MLP.
        :param seg_layers: List of sizes for each layer in the final shared MLP.
        :param scalers: Feature scaling dictionary.
        :param loss_scaler: Optional loss scaler.
        :param seg_dropout: List of dropout values to use in the last shared MLP.
        :param activation: Activation function.
        :param max_neighbors: Maximum number of neighbours used by the SetAbstraction layers aggregation.
        """
        super().__init__(nu, d, f, seg_layers[-1], scalers, loss_scaler)
        self.global_fe = SetAbstractionMrgSeq(mrg_in_features, n_dims, activation, max_neighbors)
        self.local_fe = MLP(fe_local_layers, activation=activation)

        self.decoder = MLP(seg_layers, seg_dropout, activation, False)

    def forward(self, autograd_points: Tensor, x: FoamData) -> FoamData:
        """
           :param autograd_points: Internal points to use for autograd gradients computations. Must be passed through the model.
           :param x: Input features.
           :return: The predicted values.
        """
        local_features = self.local_fe(autograd_points)

        global_in = torch.cat([x['boundary']['boundaryId'], x['boundary']['C']], dim=-1)
        global_feature = self.global_fe(global_in, x['boundary']['C'])
        exp_global = global_feature.repeat(1, local_features.shape[-2], 1)
        seg_input = torch.cat([local_features, exp_global], dim=-1)
        y = self.decoder.forward(seg_input)

        return FoamData(y, self.predicted_labels, x.domain)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = ExponentialLR(optimizer, 0.999)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]


class PipnFoamPpFull(PipnFoamBase):
    """
    Implementation of the experimental PIPN++ with Feature Propagation layers.

    Supports feature scaling and dropout.
    """

    def __init__(self,
                 nu: float,
                 d: float,
                 f: float,
                 enc_layers: list[list[int]],
                 enc_radius: list[float],
                 enc_fraction: list[float],
                 dec_layers: list[list[int]],
                 dec_k: list[int],
                 scalers: dict[str, StandardScaler],
                 loss_scaler: LossScaler = None,
                 activation: type[Module] = Mish,
                 max_neighbors=64,
                 dec_dropout: list[list[float]] = None):
        """
        :param nu: Kinematic viscosity.
        :param d: Darcy coefficient.
        :param f: Forchheimer coefficient.
        :param enc_layers: List of sizes for each layer in the geometry encoding. Must contain a list of layers for each Set Abstraction layer.
        :param enc_radius: List of radii to use in SetAbstraction layers. Set N-1 elements to add a Global Set Abstraction layer at the end.
        :param enc_fraction: List of fractions to use in the SetAbstraction layers.
        :param dec_layers: List of Feature Propagation layer sizes. Must contain a list of size for each Feature Propagation layer.
        :param dec_k: List of k values to use in the Feature Propagation layers.
        :param scalers: Feature scaling dictionary.
        :param loss_scaler: Optional loss scaler.
        :param activation: Activation function.
        :param max_neighbors: Maximum number of neighbours used by the SetAbstraction layers aggregation.
        :param dec_dropout: List of dropouts to use in the decoder layers. Must contain a list for each Feature Propagation layer.
        """
        super().__init__(nu, d, f, dec_layers[-1][-1], scalers, loss_scaler)
        self.encoder = SetAbstractionSeq(enc_fraction, enc_radius, enc_layers,
                                         activation=activation, max_neighbors=max_neighbors)
        self.decoder = FeaturePropagationSeq(dec_layers, dec_k, dec_dropout, activation)

    def forward(self, autograd_points: Tensor, x: FoamData) -> FoamData:
        """
           :param autograd_points: Internal points to use for autograd gradients computations. Must be passed through the model.
           :param x: Input features.
           :return: The predicted values.
        """
        batch = get_batch(autograd_points)
        pos = autograd_points.flatten(0, 1)
        x_in = torch.cat([x['sdf'], x['boundaryId']], dim=-1)
        x_in = x_in.flatten(0, 1)
        x_in = torch.cat([x_in, pos], dim=-1)

        out, skips = self.encoder(x_in, pos, batch)
        y, _, batch = self.decoder(*out, *skips)
        y = torch.stack(unbatch(y, batch))
        return FoamData(y, self.predicted_labels, x.domain)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = ExponentialLR(optimizer, 0.999)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
