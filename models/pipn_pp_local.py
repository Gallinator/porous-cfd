import torch
from torch import nn, Tensor
from torch_geometric.nn import MLP, knn_interpolate
from torch_geometric.utils import unbatch

from foam_dataset import StandardScaler
from models.pipn_pp import PipnPp, SetAbstraction, GlobalSetAbstraction


class LocalAggregator(torch.nn.Module):
    def __init__(self, k: int, mlp):
        super().__init__()
        self.k = k
        self.mlp = mlp

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.mlp(x)
        return x, pos_skip, batch_skip


class EncoderPpLocal(nn.Module):
    def __init__(self):
        super().__init__()
        self.local_conv = SetAbstraction(0.6, 0.2, MLP([2 + 7, 64], act=nn.Tanh(), norm=None))
        self.local_feature = LocalAggregator(8, MLP([64 + 7, 64], act=nn.Tanh(), norm=None))
        self.conv1 = SetAbstraction(0.5, 0.5, MLP([7 + 2, 64], act=nn.Tanh(), norm=None))
        self.conv2 = SetAbstraction(0.25, 0.8, MLP([64 + 2, 128], act=nn.Tanh(), norm=None))
        self.conv3 = GlobalSetAbstraction(MLP([128 + 2, 1024], act=nn.Tanh(), norm=None))

    def forward(self, x: Tensor, zones_ids: Tensor, boundary_id: Tensor) -> tuple[Tensor, Tensor]:
        batch = torch.concatenate([torch.tensor([i] * x.shape[-2]) for i in range(len(x))]).to(device=x.device,
                                                                                               dtype=torch.int64)
        local_in = torch.concatenate([x, zones_ids, boundary_id], dim=2)
        flat_x, flat_local_in = torch.concatenate([*x]), torch.concatenate([*local_in])

        local_out = self.local_conv(flat_local_in, flat_x, batch)
        local_y, _, local_batch = self.local_feature(*local_out, flat_local_in, flat_x, batch)
        local_features = torch.stack(unbatch(local_y, local_batch))

        global_in = local_in
        global_out = self.conv1(torch.concatenate([*global_in]), flat_x, batch)
        global_out = self.conv2(*global_out)
        global_y, _, batch = self.conv3(*global_out)
        global_feature = torch.stack(unbatch(global_y, batch))
        return local_features, global_feature


class PipnPpLocal(PipnPp):
    def __init__(self, n_internal: int, n_boundary: int, scalers: dict[str, StandardScaler]):
        super().__init__(n_internal, n_boundary, scalers)
        self.encoder = EncoderPpLocal()
