import torch
from torch import Tensor
from torch.nn import Tanh
from torch.optim.lr_scheduler import ExponentialLR

from dataset.foam_data import FoamData
from models.losses import ContinuityLoss, MomentumLossManufactured
from models.model_base import PorousPinnBase
from models.modules import PointNetFeatureExtract, PointNetFeatureExtractPp, MLP


class PipnManufactured(PorousPinnBase):
    def __init__(self, nu, d, f, fe_local_layers, fe_global_layers, seg_layers, activation=Tanh):
        super().__init__(seg_layers[-1], False, None)
        self.save_hyperparameters()

        self.feature_extract = PointNetFeatureExtract(fe_local_layers, fe_global_layers)
        self.decoder = MLP(seg_layers, None, activation, False)

        self.momentum_loss = MomentumLossManufactured(nu, d, f)
        self.continuity_loss = ContinuityLoss()

    def forward(self, autograd_points: Tensor, x: FoamData) -> FoamData:
        local_features, global_feature = self.feature_extract(x['cellToRegion'], autograd_points)

        exp_global = global_feature.repeat(1, local_features.shape[-2], 1)
        seg_input = torch.cat([local_features, exp_global], dim=-1)
        y = self.decoder.forward(seg_input)

        return FoamData(y, self.predicted_labels, x.domain)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, eps=1e-6)
        scheduler = ExponentialLR(optimizer, 0.9995)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]


class PipnManufacturedPorousPp(PorousPinnBase):
    def __init__(self, nu, d, f, fe_local_layers, fe_global_layers, fe_global_radius, fe_global_fraction, seg_layers,
                 activation=Tanh):
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
