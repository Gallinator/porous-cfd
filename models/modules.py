from torch.nn import Dropout, Linear, Tanh
import torch
from torch import nn, Tensor
from torch_cluster import fps, radius
from torch_geometric.nn import PointNetConv, global_max_pool
from torch_geometric.utils import unbatch
import torch_geometric.nn as gnn
import torchvision.ops as vnn

class MLP(nn.Sequential):
    def __init__(self, in_features, out_features, layers: list, activation, dropout: list | None, last_activation=True):
        super().__init__()

        if dropout is not None and len(layers) + 1 != len(dropout):
            raise AssertionError(
                f'Mismatching number of layers ({len(layers) + 1}) and dropout ({len(dropout)}).')

        n_in = in_features

        for i, l in enumerate(layers):
            self.add_module(f'Linear {i}', Linear(n_in, l))
            self.add_module(f'Activation {i}', activation())
            if dropout is not None and dropout[i] > 0:
                self.add_module(f'Dropout {i}', Dropout(dropout[i]))
            n_in = l

        # Out layers
        self.add_module(f'Linear out', Linear(n_in, out_features))
        if last_activation:
            self.add_module(f'Activation out', activation())
        if dropout is not None and dropout[-1] > 0:
            self.add_module('Dropout out', Dropout(dropout[-1]))


class PipnEncoder(nn.Module):
    def __init__(self, in_features, local_features, global_features, local_layers, global_layers, activation=Tanh):
        super().__init__()
        self.local_feature = MLP(in_features, local_features, local_layers, activation, None)
        self.global_feature = MLP(local_features + 1, global_features, global_layers, activation, None)

    def forward(self, x: Tensor, zones_ids: Tensor) -> tuple[Tensor, Tensor]:
        local_features = self.local_feature(x)
        global_feature = self.global_feature(torch.concatenate([local_features, zones_ids], dim=2))
        global_feature = torch.max(global_feature, dim=1, keepdim=True)[0]
        return local_features, global_feature


class PipnDecoder(nn.Module):
    def __init__(self, n_pde, local_features, global_features, layers, dropout=None, activation=Tanh):
        super().__init__()
        self.decoder = MLP(local_features + global_features, n_pde, layers, activation, dropout, last_activation=False)

    def forward(self, local_features: Tensor, global_feature: Tensor) -> Tensor:
        x = torch.concatenate([local_features, global_feature], 2)
        return self.decoder(x)


class Branch(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.linear = vnn.MLP(in_channels, hidden_channels, activation_layer=nn.Tanh)

    def forward(self, ceof_points: Tensor, d: Tensor, f: Tensor, inlet_points: Tensor, inlet_u: Tensor):
        """
        :param ceof_points: Coordinates of darcy boundary points(B, M, D)
        :param d: Darcy coefficients (B, M, D)
        :param f: Forchheimer coefficients (B, M, D)
        :param inlet_points: Coordinates of inlet boundary points(B, N, D)
        :param inlet_u: inlet velocity along x (B, N, K)
        :return: Parameter embedding (B, 1, H)
        """
        coefs_dim = d.shape[-1]
        points = torch.cat([ceof_points, inlet_points], dim=-2)
        par_dim = coefs_dim + coefs_dim + inlet_u.shape[-1]
        x = torch.zeros((points.shape[0], points.shape[1], par_dim), device=points.device)
        x[..., 0:ceof_points.shape[-2], 0:coefs_dim] = d
        x[..., 0:ceof_points.shape[-2], coefs_dim:coefs_dim * 2] = f
        x[..., ceof_points.shape[-2]:, coefs_dim * 2:] = inlet_u
        x = torch.cat([points, x], dim=-1)
        y = self.linear(x)
        return torch.max(y, dim=1, keepdim=True)[0]


class GeometryEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.linear = vnn.MLP(in_channels, hidden_channels, activation_layer=nn.Tanh)

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


class EncoderPp(nn.Module):
    def __init__(self):
        super().__init__()
        self.local_feature = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )

        self.conv1 = SetAbstraction(0.5, 0.5, gnn.MLP([65 + 2, 64], act=nn.Tanh(), norm=None))
        self.conv2 = SetAbstraction(0.25, 1.0, gnn.MLP([64 + 2, 128], act=nn.Tanh(), norm=None))
        self.conv3 = GlobalSetAbstraction(gnn.MLP([128 + 2, 1024], act=nn.Tanh(), norm=None))

    def forward(self, x: Tensor, zones_ids: Tensor) -> tuple[Tensor, Tensor]:
        local_features = self.local_feature(x)
        global_in = torch.concatenate([local_features, zones_ids], dim=2)
        batch = torch.concatenate([torch.tensor([i] * x.shape[-2]) for i in range(len(x))]).to(device=x.device,
                                                                                               dtype=torch.int64)
        out = self.conv1(torch.concatenate([*global_in]), torch.concatenate([*x]), batch)
        out = self.conv2(*out)
        y, _, batch = self.conv3(*out)
        global_feature = torch.stack(unbatch(y, batch))
        return local_features, global_feature
