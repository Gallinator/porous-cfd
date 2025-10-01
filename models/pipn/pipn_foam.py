import torch
from torch import Tensor
from torch.nn import Tanh, Mish
from torch.optim.lr_scheduler import ExponentialLR
from torch_geometric.utils import unbatch

from dataset.foam_data import FoamData
from dataset.foam_dataset import StandardScaler
from models.losses import ContinuityLossStandardized, MomentumLossFixed
from models.model_base import PorousPinnBase
from models.modules import FeaturePropagationSeq, SetAbstractionSeq, \
    get_batch, SetAbstractionMrgSeq, MLP, PointNetFeatureExtractPp, PointNetFeatureExtract


# self.encoder = PipnEncoder(in_dim, [64, 64], [96, 256, 512, 1024])
# self.decoder = PipnDecoder(64, 1024,
#                            [128, 64, out_features],
#                            [0.05, 0, 0])

class PipnFoamBase(PorousPinnBase):
    def __init__(self, nu, d, f, out_features, scalers: dict[str, StandardScaler], loss_scaler=None):
        super().__init__(out_features, True, loss_scaler)
        self.save_hyperparameters()

        self.u_scaler = scalers['U']
        self.p_scaler = scalers['p']
        self.points_scaler = scalers['C']

        self.momentum_loss = MomentumLossFixed(nu, d, f, self.u_scaler, self.points_scaler, self.p_scaler)
        self.continuity_loss = ContinuityLossStandardized(self.u_scaler, self.points_scaler)

    def to(self, *args, **kwargs):
        super().to(*args, *kwargs)
        self.u_scaler.to(*args, *kwargs).to(torch.float)
        self.p_scaler.to(*args, *kwargs).to(torch.float)
        self.points_scaler.to(*args, *kwargs).to(torch.float)
        return self

    def postprocess_out(self, u, p) -> tuple[Tensor, Tensor]:
        return self.u_scaler.inverse_transform(u), self.p_scaler.inverse_transform(p)


class PipnFoam(PipnFoamBase):
    def __init__(self, nu, d, f, fe_local_layers, fe_global_layers, seg_layers, seg_dropout,
                 scalers: dict[str, StandardScaler],
                 loss_scaler=None,
                 activation=Tanh):
        super().__init__(nu, d, f, seg_layers[-1], scalers, loss_scaler)
        self.feature_extract = PointNetFeatureExtract(fe_local_layers, fe_global_layers, activation)
        self.decoder = MLP(seg_layers, seg_dropout, activation, False)

    def forward(self, autograd_points: Tensor, x: FoamData) -> FoamData:
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
    def __init__(self, nu, d, f, fe_local_layers, fe_global_layers, fe_radius, fe_fraction, seg_layers, seg_dropout,
                 scalers: dict[str, StandardScaler],
                 loss_scaler=None,
                 activation=Mish):
        super().__init__(nu, d, f, seg_layers[-1], scalers, loss_scaler)

        self.feature_extract = PointNetFeatureExtractPp(fe_local_layers, fe_global_layers, fe_fraction, fe_radius,
                                                        activation)
        self.decoder = MLP(seg_layers, seg_dropout, activation, False)

    def forward(self, autograd_points: Tensor, x: FoamData) -> FoamData:
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
    def __init__(self, n_dims, nu, d, f, fe_local_layers, seg_layers, seg_dropout,
                 scalers: dict[str, StandardScaler],
                 loss_scaler=None,
                 activation=Mish):
        super().__init__(nu, d, f, seg_layers[-1], scalers, loss_scaler)
        self.global_fe = SetAbstractionMrgSeq(fe_local_layers[0] - 1, n_dims, activation)
        self.local_fe = MLP(fe_local_layers, activation=activation)

        self.decoder = MLP(seg_layers, seg_dropout, activation, False)

    def forward(self, autograd_points: Tensor, x: FoamData) -> FoamData:
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
    def __init__(self, nu, d, f, enc_layers, enc_radius, enc_fraction, dec_layers, dec_k, last_dec_dropout,
                 scalers: dict[str, StandardScaler], loss_scaler, activation=Mish):
        super().__init__(nu, d, f, dec_layers[-1][-1], scalers, loss_scaler)
        self.encoder = SetAbstractionSeq(enc_fraction, enc_radius, enc_layers, activation)
        self.decoder = FeaturePropagationSeq(dec_layers, dec_k, last_dec_dropout, activation)

    def forward(self, all_points_grad: Tensor, in_data: FoamData) -> FoamData:
        batch = get_batch(all_points_grad)
        pos = all_points_grad.flatten(0, 1)
        x = torch.cat([in_data['sdf'], in_data['boundaryId']], dim=-1)
        x = x.flatten(0, 1)
        x = torch.cat([x, pos], dim=-1)

        out, skips = self.encoder(x, pos, batch)
        y, _, batch = self.decoder(*out, *skips)
        y = torch.stack(unbatch(y, batch))
        return FoamData(y, self.predicted_labels, in_data.domain)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = ExponentialLR(optimizer, 0.999)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
