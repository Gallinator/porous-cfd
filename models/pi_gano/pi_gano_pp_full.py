import torch
from torch import Tensor
from torch.nn import SiLU, Module
from torch.optim.lr_scheduler import ExponentialLR
from torch_geometric.utils import unbatch

from dataset.foam_dataset import StandardScaler, FoamData, Normalizer
from models.losses import LossScaler
from models.modules import Branch, SetAbstractionSeq, get_batch, FeaturePropagationNeuralOperatorSeq
from models.pi_gano.base import PiGanoBase


class PiGanoPpFull(PiGanoBase):
    """
    Implementation of the experimental PI-GANO++ with Feature Propagation Neural Operators.
    """

    def __init__(self,
                 nu: float,
                 out_features: int,
                 branch_layers: list[int],
                 enc_layers: list[list[int]],
                 enc_radius: list[int],
                 enc_fraction: list[float],
                 dec_layers: list[list[int]],
                 dec_k: list[int],
                 fp_dropout: list[list[float]],
                 scalers: dict[str, StandardScaler | Normalizer],
                 variable_boundaries: dict[str, list],
                 loss_scaler: LossScaler = None,
                 activation: type[Module] = SiLU):
        """
        :param nu: Kinematic viscosity.
        :param out_features: Number of output features.
        :param branch_layers: List of branch layers sizes.
        :param enc_layers: List of layers to use for the encoder network. Must contain a list for each Set Abstraction to add.
        :param enc_radius: List of radii to use in SetAbstraction layers. Set N-1 elements to add a Global Set Abstraction layer at the end.
        :param enc_fraction: List of fractions to use in the SetAbstraction layers.
        :param dec_layers: List of layers to use for the decoder network. Must contain a list for each Feature Propagation to add.
        :param dec_k: List of k values for the decoder Feature Propagation layers.
        :param fp_dropout: List of dropout probabilities for the Feature Propagation layers. Must contain a list for each layer.
        :param scalers: Dictionary of output features scalers.
        :param variable_boundaries: Dictionary containing the variable boundaries subdomains and features.
        :param loss_scaler: Scaler applied to the losses.
        :param activation: Activation function.
        """
        super().__init__(nu, out_features, scalers, loss_scaler, variable_boundaries)

        self.branch = Branch(branch_layers, activation)
        self.encoder = SetAbstractionSeq(enc_fraction, enc_radius, enc_layers, True, activation)
        self.decoder = FeaturePropagationNeuralOperatorSeq(dec_layers, dec_k, branch_layers[-1], fp_dropout, activation)

    def forward(self, autograd_points: Tensor, x: FoamData) -> FoamData:
        """
        :param autograd_points: Internal points to use for autograd gradients computations. Must be passed through the model.
        :param x: Input features.
        :return: The predicted values.
        """
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
