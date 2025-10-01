import torch
from torch import Tensor
from torch.nn import Mish
from torch.optim.lr_scheduler import ExponentialLR
from torch_geometric.utils import unbatch

from dataset.foam_dataset import StandardScaler, FoamData, Normalizer
from models.modules import Branch, SetAbstractionSeq, get_batch, FeaturePropagationNeuralOperatorSeq
from models.pi_gano.base import PiGanoBase


class PiGanoPpFull(PiGanoBase):
    def __init__(self,
                 nu,
                 out_features,
                 branch_layers,
                 enc_layers,
                 enc_radius,
                 enc_fraction,
                 dec_layers,
                 dec_k,
                 fp_dropout,
                 scalers: dict[str, StandardScaler | Normalizer],
                 variable_boundaries: dict[str, list],
                 loss_scaler=None,
                 activation=Mish):
        super().__init__(nu, out_features, scalers, loss_scaler, variable_boundaries)

        self.branch = Branch(branch_layers, activation)
        self.encoder = SetAbstractionSeq(enc_fraction, enc_radius, enc_layers, True, activation)
        self.decoder = FeaturePropagationNeuralOperatorSeq(dec_layers, dec_k, branch_layers[-1], fp_dropout, activation)

    def forward(self, autograd_points: Tensor, x: FoamData) -> FoamData:
        # Prepare inputs
        param_features = self.get_parameters(x)
        par_embedding = self.branch.forward(param_features.clone())

        sa_input = torch.cat([x['sdf'], x['boundaryId'], autograd_points], dim=-1)
        sa_input = sa_input.reshape(-1, sa_input.shape[-1])
        sa_pos = autograd_points.reshape(-1, autograd_points.shape[-1])
        batch = get_batch(autograd_points)

        sa_out, sa_skips = self.encoder(sa_input, sa_pos, batch)
        y, _, batch = self.decoder(par_embedding, *sa_out, *sa_skips)

        y = torch.stack(unbatch(y, batch))
        return FoamData(y, self.predicted_labels, x.domain)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = ExponentialLR(optimizer, 0.999)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
