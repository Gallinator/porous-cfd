from typing import Optional

from torch.nn import Dropout, Linear, Tanh
import torch
from torch import nn, Tensor
from torch_cluster import fps, radius
from torch_geometric.nn import PointNetConv, global_max_pool, knn_interpolate
from torch_geometric.utils import unbatch
import torch_geometric.nn as gnn


class MLP(nn.Sequential):
    def __init__(self, in_features, layers: list, activation, dropout=None, last_activation=True):
        super().__init__()

        if dropout is not None and len(layers) != len(dropout):
            raise AssertionError(
                f'Mismatching number of layers ({len(layers) + 1}) and dropout ({len(dropout)}).')

        n_in = in_features

        for i, l in enumerate(layers):
            self.add_module(f'Linear {i}', Linear(n_in, l))
            if i < len(layers) - 1 or last_activation:
                self.add_module(f'Activation {i}', activation())
            if dropout is not None and dropout[i] > 0:
                self.add_module(f'Dropout {i}', Dropout(dropout[i]))
            n_in = l


class PipnEncoder(nn.Module):
    def __init__(self, in_features, local_layers, global_layers, activation=Tanh):
        super().__init__()
        self.local_feature = MLP(in_features, local_layers, activation)
        self.global_feature = MLP(local_layers[-1] + 1, global_layers, activation)

    def forward(self, x: Tensor, zones_ids: Tensor) -> tuple[Tensor, Tensor]:
        local_features = self.local_feature(x)
        global_feature = self.global_feature(torch.concatenate([local_features, zones_ids], dim=2))
        global_feature = torch.max(global_feature, dim=1, keepdim=True)[0]
        return local_features, global_feature


class PipnDecoder(nn.Module):
    def __init__(self, local_features, global_features, layers, dropout=None, activation=Tanh):
        super().__init__()
        self.decoder = MLP(local_features + global_features, layers, activation, dropout, last_activation=False)

    def forward(self, local_features: Tensor, global_feature: Tensor) -> Tensor:
        x = torch.concatenate([local_features, global_feature], 2)
        return self.decoder(x)


class Branch(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.linear = MLP(in_channels, hidden_channels, activation=Tanh)

    def forward(self, param_features: Tensor):
        """
        :param ceof_points: Coordinates of darcy boundary points(B, M, D)
        :param d: Darcy coefficients (B, M, D)
        :param f: Forchheimer coefficients (B, M, D)
        :param inlet_points: Coordinates of inlet boundary points(B, N, D)
        :param inlet_u: inlet velocity along x (B, N, K)
        :return: Parameter embedding (B, 1, H)
        """
        y = self.linear(param_features)
        return torch.max(y, dim=1, keepdim=True)[0]


class GeometryEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.linear = MLP(in_channels, hidden_channels, activation=Tanh)

    def forward(self, points: Tensor, zones_ids: Tensor) -> Tensor:
        """
        :param points: Coordinates (B, N, 2)
        :param zones_ids: Porous zone index (B, M, 1)
        :return: Embedding (B, 1, 64)
        """
        x = torch.cat([points, zones_ids], dim=-1)
        y = self.linear(x)
        return torch.max(y, dim=1, keepdim=True)[0]


class NeuralOperator(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.Tanh()
        )
        if dropout > 0:
            self.linear.append(nn.Dropout(dropout))

    def forward(self, x: Tensor, par_embedding: Tensor):
        return self.linear(x) * par_embedding


class NeuralOperatorSequential(nn.Sequential):
    def __init__(self, n_operators, n_features, dropout):
        super().__init__()
        for i in range(n_operators):
            self.add_module(f'Operator {i}', NeuralOperator(n_features, n_features, dropout[i]))

    def forward(self, input: Tensor, par_embedding: Tensor):
        for m in self:
            input = m(input, par_embedding)
        return input


class PointConvNext(PointNetConv):
    def __init__(self, r, **kwargs):
        super().__init__(**kwargs)
        self.r = r

    def message(self, x_j: Optional[Tensor], pos_i: Tensor, pos_j: Tensor) -> Tensor:
        msg = pos_j - pos_i / self.r
        if x_j is not None:
            msg = torch.cat([x_j, msg], dim=1)
        if self.local_nn is not None:
            msg = self.local_nn(msg)
        return msg


class SetAbstraction(torch.nn.Module):
    def __init__(self, ratio: float, r: float, mlp):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(mlp)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=128)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv((x, x[idx]), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class FeaturePropagation(torch.nn.Module):
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


class GlobalSetAbstraction(torch.nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.nn = mlp

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 2))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class GlobalEncoderPp(nn.Module):
    def __init__(self, fraction, radius, conv_mlp):
        super().__init__()
        convs = []
        for i, (f, r, l) in enumerate(zip(fraction, radius, conv_mlp[:-1])):
            convs.append((SetAbstraction(f, r, gnn.MLP(l, act=nn.Tanh(), norm=None)),
                          'x, pos, batch -> x, pos, batch'))

        convs.append((GlobalSetAbstraction(gnn.MLP(conv_mlp[-1], act=nn.Tanh(), norm=None)),
                      'x, pos, batch -> x, pos, batch'))

        self.set_abstractions = gnn.Sequential('x, pos, batch', convs)

    def get_batch(self, x: Tensor):
        batch = torch.arange(0, len(x)).unsqueeze(-1).repeat(1, x.shape[-2])
        return torch.cat([*batch]).to(device=x.device, dtype=torch.int64)

    def forward(self, x: Tensor, pos: Tensor):
        batch = self.get_batch(x)
        y, _, batch = self.set_abstractions(torch.concatenate([*x]), torch.concatenate([*pos]), batch)
        return torch.stack(unbatch(y, batch))


class GlobalEncoderPpMsg(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch_1 = gnn.Sequential('x, pos, batch', [
            (SetAbstraction(0.5, 0.6, MLP(2 + 1 + 2, [64, 128], Tanh)), 'x, pos, batch -> x, pos, batch'),
            (SetAbstraction(0.125, 0.8, MLP(128 + 2, [256], Tanh)), 'x, pos, batch -> x, pos, batch'),
        ])
        self.branch_2 = SetAbstraction(0.5, 0.6, MLP(2 + 1 + 2, [64, 128, 256], Tanh))
        self.branch_3 = GlobalSetAbstraction(MLP(2 + 1 + 2, [128, 256, 512], Tanh))
        self.branch_4 = GlobalSetAbstraction(MLP(256 + 2, [512], Tanh))

    def get_batch(self, x: Tensor):
        batch = torch.arange(0, len(x)).unsqueeze(-1).repeat(1, x.shape[-2])
        return torch.cat([*batch]).to(device=x.device, dtype=torch.int64)

    def forward(self, x: Tensor, pos: Tensor):
        b_0 = self.get_batch(x)
        x_0, pos_0 = x.flatten(start_dim=0, end_dim=1), pos.flatten(start_dim=0, end_dim=1)
        x_1, pos_1, b_1 = self.branch_1(x_0, pos_0, b_0)
        x_2, pos_2, b_2 = self.branch_2(x_0, pos_0, b_0)

        in_4 = torch.cat([x_1, x_2]), torch.cat([pos_1, pos_2]), torch.cat([b_1, b_2])

        x_3, pos_3, b_3 = self.branch_3(x_0, pos_0, b_0)
        x_4, pos_4, b_4 = self.branch_4(*in_4)

        x_3, x_4 = torch.stack(unbatch(x_3, b_3)), torch.stack(unbatch(x_4, b_4))

        return torch.cat([x_3, x_4], dim=-1)


class EncoderPp(nn.Module):
    def __init__(self, in_features, local_layers, fraction, radius, conv_mlp):
        super().__init__()
        self.local_feature = MLP(in_features, local_layers, Tanh)

        self.global_encoder = GlobalEncoderPp(fraction, radius, conv_mlp)

    def forward(self, pos: Tensor, zones_ids: Tensor) -> tuple[Tensor, Tensor]:
        local_features = self.local_feature(pos)
        global_in = torch.concatenate([local_features, zones_ids], dim=2)
        return local_features, self.global_encoder(global_in, pos)


class GeometryEncoderPp(nn.Module):
    def __init__(self, fraction, radius, conv_mlp):
        super().__init__()
        self.set_abstraction = GlobalEncoderPp(fraction, radius, conv_mlp)

    def forward(self, pos: Tensor, zones_ids: Tensor) -> Tensor:
        """
        :param pos: Coordinates (B, N, D)
        :param zones_ids: Porous zone index (B, M, 1)
        :return: Embedding (B, 1, K)
        """
        y = self.set_abstraction(pos, zones_ids)
        return torch.max(y, dim=1, keepdim=True)[0]
