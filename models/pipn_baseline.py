import torch
from torch import Tensor
from torch.optim.lr_scheduler import ExponentialLR
from dataset.foam_data import FoamData
from models.losses import ContinuityLoss, MomentumLossManufactured
from models.model_base import PorousPinnBase
from models.modules import PipnEncoder, PipnDecoder


class PipnPorous(PorousPinnBase):
    def __init__(self, nu, d, f, in_dim, out_features):
        super().__init__(out_features, nu, False)
        self.save_hyperparameters()
        self.encoder = PipnEncoder(in_dim, 64, 1024, [64], [64, 128])
        self.decoder = PipnDecoder(out_features, 64, 1024, [512, 256, 128])
        self.d = d
        self.f = f

        self.momentum_loss = MomentumLossManufactured(self.nu, self.d, self.f)
        self.continuity_loss = ContinuityLoss()

    def forward(self, autograd_points: Tensor, x: FoamData) -> FoamData:
        local_features, global_feature = self.encoder.forward(autograd_points, x['cellToRegion'])
        exp_global = global_feature.repeat(1, local_features.shape[-2], 1)
        y = self.decoder.forward(local_features, exp_global)
        return FoamData(y, self.predicted_labels, x.domain)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, eps=1e-6)
        scheduler = ExponentialLR(optimizer, 0.9995)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
