import torch
from torch import Tensor, Module
from torch.nn import Tanh
from torch.optim.lr_scheduler import ExponentialLR

from dataset.foam_data import FoamData
from models.losses import ContinuityLoss, MomentumLossManufactured
from models.model_base import PorousPinnBase
from models.modules import PointNetFeatureExtract, PointNetFeatureExtractPp, MLP


class PipnManufactured(PorousPinnBase):
    """
    PIPN modified for the mixed fluid-porous task, using the manufactured solutions losses.

    Thi model does not use features scaling. The data loss is disabled by default.
    """

    def __init__(self,
                 nu: float,
                 d: float,
                 f: float,
                 fe_local_layers: list[int],
                 fe_global_layers: list[int],
                 seg_layers: list[int],
                 activation: type[Module] = Tanh):
        """
        :param nu: Kinematic viscosity.
        :param d: Darcy coefficient.
        :param f: Forchheimer coefficient.
        :param fe_local_layers: List of sizes for each layer in the first shared MLP.
        :param fe_global_layers: List of sizes for each layer in MLP before the max pooling.
        :param seg_layers: List of sizes for each layer in the final shared MLP.
        :param activation: Activation function.
        """
        super().__init__(seg_layers[-1], False, None)
        self.save_hyperparameters()

        self.feature_extract = PointNetFeatureExtract(fe_local_layers, fe_global_layers)
        self.decoder = MLP(seg_layers, None, activation, False)

        self.momentum_loss = MomentumLossManufactured(nu, d, f)
        self.continuity_loss = ContinuityLoss()

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
        y = self.decoder.forward(seg_input)

        return FoamData(y, self.predicted_labels, x.domain)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, eps=1e-6)
        scheduler = ExponentialLR(optimizer, 0.9995)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]


class PipnManufacturedPorousPp(PorousPinnBase):
    """
    PIPN++ using the manufactured solutions equations without feature scaling.
    """

    def __init__(self,
                 nu: float,
                 d: float,
                 f: float,
                 fe_local_layers: list[int],
                 fe_global_layers: list[list[int]],
                 fe_global_radius: list[float],
                 fe_global_fraction: list[float],
                 seg_layers: list[int],
                 activation=Tanh):
        """
        :param nu: Kinematic viscosity.
        :param d: Darcy coefficient.
        :param f: Forchheimer coefficient.
        :param fe_local_layers: List of sizes for each layer in the first shared MLP.
        :param fe_global_layers: List of sizes for the geometry encoding layers. Must contain a list for each Set Abstraction layer.
        :param fe_global_radius: List of radii to use in SetAbstraction layers. Set N-1 elements to add a Global Set Abstraction layer at the end.
        :param fe_global_fraction: List of fractions to use in the SetAbstraction layers.
        :param seg_layers: List of sizes for each layer in the final shared MLP.
        :param activation: Activation function.
        """
        super().__init__(seg_layers[-1], None, None)
        self.save_hyperparameters()
        self.feature_extract = PointNetFeatureExtractPp(fe_local_layers,
                                                        fe_global_layers,
                                                        fe_global_fraction,
                                                        fe_global_radius,
                                                        activation)
        self.decoder = MLP(seg_layers, None, activation, False)

        self.momentum_loss = MomentumLossManufactured(nu, d, f)
        self.continuity_loss = ContinuityLoss()

    def forward(self, autograd_points: Tensor, x: FoamData) -> FoamData:
        """
           :param autograd_points: Internal points to use for autograd gradients computations. Must be passed through the model.
           :param x: Input features.
           :return: The predicted values.
        """
        geom_features = torch.cat([x['boundary']['boundaryId'], x['boundary']['C']], dim=-1)
        local_features, global_feature = self.feature_extract(geom_features,
                                                              x['boundary']['C'],
                                                              autograd_points)

        exp_global = global_feature.repeat(1, local_features.shape[-2], 1)
        seg_input = torch.cat([local_features, exp_global], dim=-1)
        y = self.decoder.forward(seg_input)

        return FoamData(y, self.predicted_labels, x.domain)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, eps=1e-6)
        scheduler = ExponentialLR(optimizer, 0.9995)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
