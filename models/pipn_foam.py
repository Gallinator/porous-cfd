import torch
from torch import Tensor
from torch.optim.lr_scheduler import ExponentialLR
from dataset.foam_data import FoamData
from dataset.foam_dataset import StandardScaler
from models.losses import ContinuityLossStandardized, MomentumLossFixed
from models.model_base import PorousPinnBase
from models.modules import PipnEncoder, PipnDecoder


class PipnFoam(PorousPinnBase):
    def __init__(self, nu, d, f, in_dim, out_features, scalers: dict[str, StandardScaler], loss_scaler=None):
        super().__init__(out_features, nu, True, loss_scaler)
        self.save_hyperparameters()
        self.encoder = PipnEncoder(in_dim, [64, 64], [96, 128, 1024])
        self.decoder = PipnDecoder(64, 1024,
                                   [512, 256, 128, out_features],
                                   [0.05, 0.05, 0, 0])
        self.d = d
        self.f = f

        self.u_scaler = scalers['U']
        self.p_scaler = scalers['p']
        self.points_scaler = scalers['C']

        self.momentum_loss = MomentumLossFixed(self.nu, self.d, self.f, self.u_scaler, self.points_scaler,
                                               self.p_scaler)
        self.continuity_loss = ContinuityLossStandardized(self.u_scaler, self.points_scaler)

    def to(self, *args, **kwargs):
        super().to(*args, *kwargs)
        self.u_scaler.to(*args, *kwargs).to(torch.float)
        self.p_scaler.to(*args, *kwargs).to(torch.float)
        self.points_scaler.to(*args, *kwargs).to(torch.float)
        return self

    def postprocess_out(self, u, p) -> tuple[Tensor, Tensor]:
        return self.u_scaler.inverse_transform(u), self.p_scaler.inverse_transform(p)

    def forward(self, all_points_grad: Tensor, x: FoamData) -> FoamData:
        local_features, global_feature = self.encoder.forward(all_points_grad, x['cellToRegion'])
        exp_global = global_feature.repeat(1, local_features.shape[-2], 1)
        y = self.decoder.forward(local_features, exp_global)
        return FoamData(y, self.predicted_labels, x.domain)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = ExponentialLR(optimizer, 0.999)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
