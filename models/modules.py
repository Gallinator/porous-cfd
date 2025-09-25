from collections import OrderedDict
from typing import Optional

from torch.nn import Dropout, Linear, Tanh
import torch
from torch import nn, Tensor
from torch_cluster import fps, radius
from torch_geometric.nn import PointNetConv, global_max_pool, knn_interpolate
from torch_geometric.utils import unbatch
import torch_geometric.nn as gnn


def get_batch(x: Tensor):
    batch = torch.arange(0, len(x)).unsqueeze(-1).repeat(1, x.shape[-2])
    return torch.cat([*batch]).to(device=x.device, dtype=torch.int64)


class MLP(nn.Sequential):
    def __init__(self, layers: list, dropout=None, activation=Tanh, last_activation=True):
        super().__init__()

        if dropout is not None and len(layers) - 1 != len(dropout):
            raise AssertionError(
                f'Mismatching number of layers ({len(layers)}) and dropout ({len(dropout)}).')

        n_in = layers[0]
        for i, l in enumerate(layers[1:]):
            self.add_module(f'Linear {i}', Linear(n_in, l))
            if i < len(layers) - 2 or last_activation:
                self.add_module(f'Activation {i}', activation())
            if dropout is not None and dropout[i] > 0:
                self.add_module(f'Dropout {i}', Dropout(dropout[i]))
            n_in = l


class PointNetFeatureExtract(nn.Module):
    def __init__(self, local_layers, global_layers, activation=Tanh):
        super().__init__()
        self.local_feature = MLP(local_layers, activation=activation)
        self.global_feature = MLP(global_layers, activation=activation)

    def forward(self, x: Tensor, pos: Tensor) -> tuple[Tensor, Tensor]:
        local_features = self.local_feature(pos)

        global_in = torch.concatenate([local_features, x], dim=-1)
        global_feature = self.global_feature(global_in)
        global_feature = torch.max(global_feature, dim=1, keepdim=True)[0]
        return local_features, global_feature


class BatchedDecorator(nn.Module):
    """
    This module allows to convert from an input of dimension (B,M,N) to an input of (M,N) with a batch tensor used by PyTorch Geometric.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: Tensor, pos: Tensor) -> Tensor:
        batch = get_batch(x)
        x, pos = torch.cat([*x]), torch.cat([*pos])
        x, _, batch = self.module(x, pos, batch)
        return torch.stack(unbatch(x, batch))


class PointNetFeatureExtractPp(nn.Module):
    def __init__(self, local_layers, global_layers, global_fraction, global_radius, activation=Tanh):
        super().__init__()
        self.local_feature = MLP(local_layers, activation=activation)
        sa_layers = SetAbstractionSeq(global_fraction, global_radius, global_layers, return_skip=False)
        self.global_feature = BatchedDecorator(sa_layers)

    def forward(self, x: Tensor, pos: Tensor) -> tuple[Tensor, Tensor]:
        local_features = self.local_feature(pos)
        global_in = torch.concatenate([pos, x], dim=-1)
        global_feature = self.global_feature(global_in, pos)
        global_feature = torch.max(global_feature, dim=1, keepdim=True)[0]
        return local_features, global_feature


class GeometryEncoderPp(nn.Module):
    def __init__(self, fraction, radius, conv_mlp):
        super().__init__()
        sa_layers = SetAbstractionSeq(fraction, radius, conv_mlp, return_skip=False)
        self.set_abstraction = BatchedDecorator(sa_layers)

    def forward(self, x: Tensor, pos: Tensor) -> Tensor:
        """
        :param pos: Coordinates (B, N, D)
        :param zones_ids: Porous zone index (B, M, 1)
        :return: Embedding (B, 1, K)
        """
        in_data = torch.cat([x, pos], dim=-1)
        y = self.set_abstraction(in_data, pos)
        return torch.max(y, dim=1, keepdim=True)[0]


class Branch(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.linear = MLP(hidden_channels, activation=Tanh)

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
    def __init__(self, hidden_channels):
        super().__init__()
        self.linear = MLP(hidden_channels, activation=Tanh)

    def forward(self, x: Tensor, pos: Tensor) -> Tensor:
        """
        :param pos: Coordinates (B, N, D)
        :param zones_ids: Porous zone index (B, M, 1)
        :return: Embedding (B, 1, K)
        """
        in_data = torch.cat([x, pos], dim=-1)
        y = self.linear(in_data)
        return torch.max(y, dim=1, keepdim=True)[0]


class NeuralOperator(nn.Module):
    def __init__(self, out_channels, dropout):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(out_channels, out_channels),
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
            self.add_module(f'Operator {i}', NeuralOperator(n_features, dropout[i]))

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
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64)
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


class FeaturePropagationNeuralOperator(nn.Module):
    def __init__(self, k: int, mlp, par_size):
        super().__init__()
        self.k = k
        self.mlp = mlp
        self.par_reduce_mlp = nn.Sequential(nn.Linear(par_size, mlp.channel_list[-1]), mlp.act)

    def forward(self, par_embedding, x, pos, batch, x_skip, pos_skip, batch_skip):
        batch_size = par_embedding.shape[0]

        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.mlp(x)

        # Unbatch parameters
        n_repeats = x.shape[0] // batch_size
        layer_par = par_embedding.repeat(1, n_repeats, 1)
        layer_par = layer_par.flatten(start_dim=0, end_dim=1)
        x = x * self.par_reduce_mlp(layer_par)

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


class SetAbstractionMrgSeq(nn.Module):
    def __init__(self, in_features, activation):
        super().__init__()
        self.branch_1 = gnn.Sequential('x, pos, batch', [
            (SetAbstraction(0.5, 0.6,
                            gnn.MLP([in_features + 1 + in_features, 64, 128], act=activation, norm=None, plain_last=False)),
             'x, pos, batch -> x, pos, batch'),
            (SetAbstraction(0.125, 0.8,
                            gnn.MLP([128 + in_features, 256], act=activation, norm=None, plain_last=False)),
             'x, pos, batch -> x, pos, batch'),
        ])
        self.branch_2 = SetAbstraction(0.5, 0.6,
                                       gnn.MLP([in_features + 1 + in_features, 64, 128, 256], act=activation, norm=None,
                                               plain_last=False))
        self.branch_3 = GlobalSetAbstraction(
            gnn.MLP([in_features + 1 + in_features, 128, 256, 512], act=activation, norm=None, plain_last=False))
        self.branch_4 = GlobalSetAbstraction(
            gnn.MLP([256 + in_features, 512], act=activation, norm=None, plain_last=False))

    def forward(self, x: Tensor, pos: Tensor):
        b_0 = get_batch(x)
        x_0, pos_0 = x.flatten(start_dim=0, end_dim=1), pos.flatten(start_dim=0, end_dim=1)
        x_1, pos_1, b_1 = self.branch_1(x_0, pos_0, b_0)
        x_2, pos_2, b_2 = self.branch_2(x_0, pos_0, b_0)

        in_4 = torch.cat([x_1, x_2]), torch.cat([pos_1, pos_2]), torch.cat([b_1, b_2])

        x_3, pos_3, b_3 = self.branch_3(x_0, pos_0, b_0)
        x_4, pos_4, b_4 = self.branch_4(*in_4)

        x_3, x_4 = torch.stack(unbatch(x_3, b_3)), torch.stack(unbatch(x_4, b_4))

        return torch.cat([x_3, x_4], dim=-1)


class SetAbstractionSeq(nn.Module):
    def __init__(self, fraction, radius, conv_mlp, return_skip=True):
        super().__init__()
        layers = OrderedDict()
        for i, (f, r, l) in enumerate(zip(fraction, radius, conv_mlp)):
            layers[f'Sa-{i}'] = SetAbstraction(f, r, gnn.MLP(l, act=nn.Tanh(), norm=None, plain_last=False))
        if len(conv_mlp) > len(radius):
            layers['Global-Sa'] = GlobalSetAbstraction(
                gnn.MLP(conv_mlp[-1], act=nn.Tanh(), norm=None, plain_last=False))

        self.layers = nn.Sequential(layers)
        self.return_skip = return_skip

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> tuple:
        input = x, pos, batch
        skips = [input]
        for sa in self.layers:
            input = sa(*input)
            skips.append(input)
        return (input, skips[:-1]) if self.return_skip else input


class FeaturePropagationSeq(nn.Module):
    def __init__(self, fp_layers, k, dropout=None, activation=nn.Tanh()):
        super().__init__()

        dropout = [0.] * len(fp_layers) if dropout is None else dropout
        layers = OrderedDict()
        for i, (l, k, d) in enumerate(zip(fp_layers, k, dropout)):
            is_last = i == len(fp_layers) - 1
            layers[f'Fp-{i}'] = FeaturePropagation(k, gnn.MLP(l, act=activation, norm=None,
                                                              dropout=d,
                                                              plain_last=is_last))

        self.layers = nn.Sequential(layers)

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor, *skips) -> tuple:
        out = x, pos, batch
        for l, s in zip(self.layers, skips[::-1]):
            out = l(*out, *s)
        return out


class FeaturePropagationNeuralOperatorSeq(nn.Module):
    def __init__(self, fp_layers, k, par_size, dropout=None, activation=nn.Tanh()):
        super().__init__()
        dropout = [0.] * len(fp_layers) if dropout is None else dropout
        layers = OrderedDict()
        for i, (l, k, d) in enumerate(zip(fp_layers, k, dropout)):
            is_last = i == len(fp_layers) - 1
            layers[f'Fp-{i}'] = FeaturePropagationNeuralOperator(k, gnn.MLP(l, act=activation, norm=None,
                                                                            dropout=d,
                                                                            plain_last=is_last),
                                                                 par_size)
        self.layers = nn.Sequential(layers)

    def forward(self, par_embedding: Tensor, x: Tensor, pos: Tensor, batch: Tensor, *skips) -> tuple:
        """
        :param par_embedding: parameter embedding (B,N,M)
        :param x: feature vectors (B*N,K)
        :param pos: positions vector (B*N,D)
        :param batch: batch indices (B*N)
        :param skips: skip outputs from SetAbstractionSeq
        :return:
        """
        out = x, pos, batch
        for l, s in zip(self.layers, skips[::-1]):
            out = l(par_embedding, *out, *s)
        return out
