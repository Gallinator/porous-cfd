from collections import OrderedDict
from typing import Optional

from torch.nn import Dropout, Linear, Tanh, Module
import torch
from torch import nn, Tensor
from torch_cluster import fps, radius
from torch_geometric.nn import PointNetConv, global_max_pool, knn_interpolate
from torch_geometric.utils import unbatch
import torch_geometric.nn as gnn


def get_batch(x: Tensor) -> Tensor:
    """
    Utility function to convert batched inputs to PyTorch Geometric compatible batch vector.
    :param x: Batched data of shape (B,N,*).
    :return: A vector (N*B) whose entries represent the batch of each sample in x.
    """
    batch = torch.arange(0, len(x)).unsqueeze(-1).repeat(1, x.shape[-2])
    return torch.cat([*batch]).to(device=x.device, dtype=torch.int64)


class MLP(nn.Sequential):
    """
    Implements an MLP to standardize differences between the torch vision and PyTorch Geometric implementations.

    Supports custom activation function, per layer dropout and plain last layer.
    """

    def __init__(self, layers: list[int],
                 dropout: list[float] = None,
                 activation: type[Module] = Tanh,
                 last_activation=True):
        """
        :param layers: List of layers of shape (N), layers will be added using Linear modules.
        :param dropout: A list of values setting the dropout of each layer. Must be of length N-1.
        :param activation: The activation function type.
        :param last_activation: Set to true to add activation after the last layer.
        """
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
    """
    Extracts the features from an input point cloud. It corresponds to the first two shared layers and the max pooling of a PointNet.
    """

    def __init__(self, local_layers: list[int], global_layers: list[int], activation: type[Module] = Tanh):
        """
        :param local_layers: The first stack of shared MLP layers.
        :param global_layers: The stack of layers before the max pooling module.
        :param activation: The activation function.
        """
        super().__init__()
        self.local_feature = MLP(local_layers, activation=activation)
        self.global_feature = MLP(global_layers, activation=activation)

    def forward(self, x: Tensor, pos: Tensor) -> tuple[Tensor, Tensor]:
        """
        :param x: The input feature Tensor (B,N,F).
        :param pos: The input position tensor (B,N,D).
        :return: The local features (B,N,L) and global embedding (B,N,E).
        """
        local_features = self.local_feature(pos)

        global_in = torch.concatenate([local_features, x], dim=-1)
        global_feature = self.global_feature(global_in)
        global_feature = torch.max(global_feature, dim=1, keepdim=True)[0]
        return local_features, global_feature


class BatchedDecorator(nn.Module):
    """
    This module allows to convert from an input of dimension (B,M,N) to an input of (M,N) used by PyTorch Geometric.
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
    """
    Similar to PointNetFeatureExtract but replaces the second stack of MLP layers SetAbstraction layers.
    Used in the PointNet++ models.
    """

    def __init__(self,
                 local_layers: list[int],
                 global_layers: list[list[int]],
                 global_fraction: list[float],
                 global_radius: list[float],
                 activation: type[Module] = Tanh,
                 max_neighbors=64):
        """
        :param local_layers: List if layers to use in the first MLP stack.
        :param global_layers: List of SetAbstraction layers to use for the global embedding extraction of shape (N,*).
         Must contain a list of layers for each SetAbstraction layer.
        :param global_fraction: Fraction of points to sample in each SetAbstraction layer.
        :param global_radius: Radius of each SetAbstraction layer. Set N-1 values to add a GlobalSetAbstraction layer at the end.
        :param activation: The activation function.
        :param max_neighbors: Limit the maximum number of neighbors to aggregate information from.
        """
        super().__init__()
        self.local_feature = MLP(local_layers, activation=activation)
        sa_layers = SetAbstractionSeq(global_fraction, global_radius, global_layers, return_skip=False,
                                      activation=activation, max_neighbors=max_neighbors)
        self.global_feature = BatchedDecorator(sa_layers)

    def forward(self, geom_features: Tensor, geom_pos: Tensor, global_pos: Tensor) -> tuple[Tensor, Tensor]:
        """
        :param geom_features: Features to use in the geometry embedding extraction, shape (B,M,F).
        :param geom_pos: Points coordinates to use in the geometry embedding extraction, shape (B,M,D).
        :param global_pos: Points coordinates to use in the first MLP stack, shape (B,N,D).
        :return: The local features (B,N,L) and global embedding (B,N,E).
        """
        local_features = self.local_feature(global_pos)
        global_feature = self.global_feature(geom_features, geom_pos)

        return local_features, global_feature


class GeometryEncoderPp(nn.Module):
    """
    Geometry encoder used in the PI-GANO++.
    """

    def __init__(self, fraction: list[float],
                 radius: list[float],
                 conv_mlp: list[list[int]],
                 activation: type[Module] = Tanh,
                 max_neighbors=64):
        """
        See PointNetFeatureExtractPp.__init__().
        """
        super().__init__()
        sa_layers = SetAbstractionSeq(fraction, radius, conv_mlp,
                                      return_skip=False,
                                      activation=activation,
                                      max_neighbors=max_neighbors)
        self.set_abstraction = BatchedDecorator(sa_layers)

    def forward(self, x: Tensor, pos: Tensor) -> Tensor:
        """
        :param pos: Coordinates (B, M, D).
        :param x: Features tensor (B, M, F).
        :return: Embedding (B, 1, K).
        """
        return self.set_abstraction(x, pos)


class Branch(nn.Module):
    """
    Branch subnetwork used in the PI-GANO based models.
    """

    def __init__(self, hidden_channels: list[int], activation: type[Module] = Tanh):
        """
        :param hidden_channels: List of input and output layer sizes.
        :param activation: Activation function.
        """
        super().__init__()
        self.linear = MLP(hidden_channels, activation=activation)

    def forward(self, param_features: Tensor):
        """
        :param param_features: Branch network input features of shape (B, M, D).
        :return: Parameter embedding (B, 1, H).
        """
        y = self.linear(param_features)
        return torch.max(y, dim=1, keepdim=True)[0]


class GeometryEncoder(nn.Module):
    """
    Geometry encoder used in the PI-GANO model.
    """

    def __init__(self, hidden_channels: list[int], activation=Tanh):
        """
        :param hidden_channels: List of input and output layer sizes.
        :param activation: Activation function.
        """
        super().__init__()
        self.linear = MLP(hidden_channels, activation=activation)

    def forward(self, x: Tensor, pos: Tensor) -> Tensor:
        """
        :param pos: Coordinates (B, M, D).
        :param x: Features tensor (B, M, F).
        :return: Embedding (B, 1, K).
        """
        in_data = torch.cat([x, pos], dim=-1)
        y = self.linear(in_data)
        return torch.max(y, dim=1, keepdim=True)[0]


class NeuralOperator(nn.Module):
    """
    Implements the neural operator module used in PI-GANO models.

    A dropout layer is optionally applied before the multiplication with the branch output.
    """

    def __init__(self, out_channels: int, dropout: float, activation: type[Module] = Tanh):
        """
        :param out_channels: The size of the operator.
        :param dropout: The dropout probability.
        :param activation: The activation function.
        """
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(out_channels, out_channels),
        )
        if activation is not None:
            self.linear.append(activation())
        if dropout > 0:
            self.linear.append(nn.Dropout(dropout))

    def forward(self, x: Tensor, par_embedding: Tensor):
        """
        :param x: The layer input of shape (B,N,P).
        :param par_embedding: The branch output vector of shape (B,1,P).
        :return: The output of the neural operator.
        """
        return self.linear(x) * par_embedding


class NeuralOperatorSequential(nn.Sequential):
    """
    Sequential version of Neural Operators. Supports per layer dropout.
    """

    def __init__(self,
                 n_operators: int,
                 n_features: int,
                 dropout: list[float],
                 activation: type[Module] = Tanh,
                 last_activation=True):
        """
        :param n_operators: Number of Neura Operators.
        :param n_features: Size of the input features.
        :param dropout: List of dropout probabilities.
        :param activation: Activation function.
        :param last_activation: Pass True to add an activation function after the last layer.
        """
        super().__init__()
        for i in range(n_operators):
            activation = None if i == n_operators - 1 and not last_activation else activation
            self.add_module(f'Operator {i}', NeuralOperator(n_features, dropout[i], activation))

    def forward(self, input: Tensor, par_embedding: Tensor):
        for m in self:
            input = m(input, par_embedding)
        return input


class PointConvNext(PointNetConv):
    """
    Implementation of PointNetConv with radius normalization. For further details see PointNetConv.
    """

    def __init__(self, r: float, **kwargs):
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
    """
    Set abstraction layer, adapted from PyTorch Geometric examples. The MLP is applied before the aggregation.
    """

    def __init__(self, ratio: float, r: float, mlp: gnn.MLP, max_neighbors=64):
        """
        :param ratio: Ratio of points to select as centroids.
        :param r: Aggregation radius.
        :param mlp: MLP to use for aggregation.
        :param max_neighbors: Limit the neighbors to select on each centroid.
        """
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConvNext(r, local_nn=mlp)
        self.max_neighbors = max_neighbors

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor):
        """
        :param x: Point features of shape (N,F).
        :param pos: Point coordinates of shape (N,D).
        :param batch: Batch vector of shape (N).
        :return: x, pos and batch for the selected centroids.
        """
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=self.max_neighbors)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv((x, x[idx]), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class FeaturePropagation(torch.nn.Module):
    """
    Feature Propagation module, adapted from PyTorch Geometric examples.
    """

    def __init__(self, k: int, mlp: gnn.MLP):
        """
        :param k: The number of neighbors to select.
        :param mlp: The MLP to use for the propagation.
        """
        super().__init__()
        self.k = k
        self.mlp = mlp

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        """
        :param x: Points features from previous layers.
        :param pos: Points position from previous layers.
        :param batch: Points batch from previous layers.
        :param x_skip: Points features from previous skip connection.
        :param pos_skip: Points positions from skip connection.
        :param batch_skip: Points batch from skip connection.
        :return: The propagated x, pos and batch.
        """
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.mlp(x)
        return x, pos_skip, batch_skip


class FeaturePropagationNeuralOperator(nn.Module):
    """
    Implements experimental Neural Operator Feature Propagation.
    The output of the Feature Propagation is multiplied with a dimensionally-reduced branch vector.
    """

    def __init__(self, k: int, mlp: gnn.MLP, par_size: int):
        """
        :param k: The number of neighbors to select.
        :param mlp: The MLP to use for the propagation.
        :param par_size: Size of the branch output vector.
        """
        super().__init__()
        self.k = k
        self.mlp = mlp
        self.par_reduce_mlp = nn.Sequential(nn.Linear(par_size, mlp.channel_list[-1]), mlp.act)

    def forward(self, par_embedding, x, pos, batch, x_skip, pos_skip, batch_skip):
        """
        :param par_embedding: Output of the branch network of shape (B,1,P).
        :param x: Points features from previous layers.
        :param pos: Points position from previous layers.
        :param batch: Points batch from previous layers.
        :param x_skip: Points features from previous skip connection.
        :param pos_skip: Points positions from skip connection.
        :param batch_skip: Points batch from skip connection.
        :return: The propagated x, pos and batch.
        """
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
    """
    Global Set Abstraction layers, adapted from PyTorch Geometric examples.
    """

    def __init__(self, mlp: gnn.MLP):
        super().__init__()
        self.nn = mlp

    def forward(self, x, pos, batch):
        """
        :param x: Point features of shape (N,F).
        :param pos: Point coordinates of shape (N,D).
        :param batch: Batch vector of shape (N).
        :return: x, pos and batch for the selected centroids.
        """
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), pos.size(-1)), requires_grad=pos.requires_grad)
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class SetAbstractionMrgSeq(nn.Module):
    """
    Implements an MRG Set Abstraction branch as per the original paper.
    """

    def __init__(self, in_features: int,
                 n_dims: int,
                 activation: type[Module] = Tanh,
                 max_neighbors=64):
        """
        :param in_features: Number of input features.
        :param n_dims: Coordinates dimensionality.
        :param activation: Activation function.
        :param max_neighbors: Maximum neighbors used for aggregation.
        """
        super().__init__()
        self.branch_1 = gnn.Sequential('x, pos, batch', [
            (SetAbstraction(0.5, 0.5,
                            gnn.MLP([in_features + n_dims, 64, 128], act=activation(), norm=None,
                                    plain_last=False),
                            max_neighbors=max_neighbors),
             'x, pos, batch -> x, pos, batch'),
            (SetAbstraction(0.125, 1,
                            gnn.MLP([128 + n_dims, 256], act=activation(), norm=None, plain_last=False),
                            max_neighbors=max_neighbors),
             'x, pos, batch -> x, pos, batch'),
        ])
        self.branch_2 = SetAbstraction(0.5, 0.5,
                                       gnn.MLP([in_features + n_dims, 64, 128, 256], act=activation(), norm=None,
                                               plain_last=False),
                                       max_neighbors=max_neighbors)
        self.branch_3 = GlobalSetAbstraction(
            gnn.MLP([in_features + n_dims, 128, 256, 512], act=activation(), norm=None, plain_last=False))
        self.branch_4 = GlobalSetAbstraction(
            gnn.MLP([256 + n_dims, 512], act=activation(), norm=None, plain_last=False))

    def forward(self, x: Tensor, pos: Tensor):
        """
        :param x: Point features of shape (B,N,F).
        :param pos: Point coordinates of shape (B,N,D).
        :return: Output of shape (B,M,O)
        """
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
    """
    Sequential version of SetAbstraction modules. Allows to return the output of each layer in the sequence.
    """

    def __init__(self,
                 fraction: list[float],
                 radius: list[float],
                 conv_mlp: list[list[float]],
                 return_skip=True,
                 activation: type[Module] = Tanh,
                 max_neighbors=64):
        """
        :param fraction: Ratio of points to select as centroids.
        :param radius: Aggregation radius. Set N-1 elements to add a Global Set Abstraction layer at the end.
        :param conv_mlp: List of layers sizes (N).
        :param max_neighbors: Limit the neighbors to select on each centroid.
        :param return_skip: Set to True to return the output of each layer.
        :param activation: Activation function.
        """
        super().__init__()
        layers = OrderedDict()
        for i, (f, r, l) in enumerate(zip(fraction, radius, conv_mlp)):
            layers[f'Sa-{i}'] = SetAbstraction(f, r, gnn.MLP(l, act=activation(), norm=None, plain_last=False),
                                               max_neighbors=max_neighbors)
        if len(conv_mlp) > len(radius):
            layers['Global-Sa'] = GlobalSetAbstraction(
                gnn.MLP(conv_mlp[-1], act=activation(), norm=None, plain_last=False))

        self.layers = nn.Sequential(layers)
        self.return_skip = return_skip

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> tuple:
        """
        :param x: Point features of shape (N,F).
        :param pos: Point coordinates of shape (N,D).
        :param batch: Batch vector of shape (N).
        :return: x, pos and output or a list of all outputs of the layers.
        """
        input = x, pos, batch
        skips = [input]
        for sa in self.layers:
            input = sa(*input)
            skips.append(input)
        return (input, skips[:-1]) if self.return_skip else input


class FeaturePropagationSeq(nn.Module):
    """
    Sequential version of the Feature Propagation.
    """

    def __init__(self,
                 fp_layers: list[list[int]],
                 k: list[int],
                 dropout: list[list[float]] = None,
                 activation: type[Module] = Tanh):
        """
        :param fp_layers: List of layers to use for MLPs. Must contain a list for each Feature Propagation layer.
        :param k: List of values of k to use in each layer.
        :param dropout: List of dropout values.  Must contain a list for each Feature Propagation layer.
        :param activation: Activation function.
        """
        super().__init__()

        dropout = [0.] * len(fp_layers) if dropout is None else dropout
        layers = OrderedDict()
        for i, (l, k, d) in enumerate(zip(fp_layers, k, dropout)):
            is_last = i == len(fp_layers) - 1
            layers[f'Fp-{i}'] = FeaturePropagation(k, gnn.MLP(l, act=activation(), norm=None,
                                                              dropout=d,
                                                              plain_last=is_last))

        self.layers = nn.Sequential(layers)

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor, *skips: tuple[Tensor]) -> tuple:
        """
        :param x: Point features of shape (N,F).
        :param pos: Point coordinates of shape (N,D).
        :param batch: Batch vector of shape (N).
        :param skips: List of x, batch, pos skip outputs.
        :return: x, pos and output or a list of all outputs of the layers.
        """
        out = x, pos, batch
        for l, s in zip(self.layers, skips[::-1]):
            out = l(*out, *s)
        return out


class FeaturePropagationNeuralOperatorSeq(nn.Module):
    """
    Sequential version of Feature Propagation Neural Operator.
    """

    def __init__(self,
                 fp_layers: list[list[int]],
                 k: list[int],
                 par_size: int,
                 dropout: list[list[float]] = None,
                 activation: type[Module] = Tanh):
        """
        :param fp_layers: List of layers to use for MLPs. Must contain a list for each Feature Propagation layer.
        :param k: List of values of k to use in each layer.
        :param dropout: List of dropout values.  Must contain a list for each Feature Propagation layer.
        :param activation: Activation function.
        :param par_size: Size of each Neural Operator layer.
        """
        super().__init__()
        dropout = [0.] * len(fp_layers) if dropout is None else dropout
        layers = OrderedDict()
        for i, (l, k, d) in enumerate(zip(fp_layers, k, dropout)):
            is_last = i == len(fp_layers) - 1
            layers[f'Fp-{i}'] = FeaturePropagationNeuralOperator(k,
                                                                 gnn.MLP(l, act=activation(), norm=None,
                                                                         dropout=d,
                                                                         plain_last=is_last),
                                                                 par_size)
        self.layers = nn.Sequential(layers)

    def forward(self, par_embedding: Tensor, x: Tensor, pos: Tensor, batch: Tensor, *skips: tuple[Tensor]) -> tuple:
        """
        :param par_embedding: parameter embedding (B,1,M).
        :param x: feature vectors (N,K).
        :param pos: positions vector (N,D).
        :param batch: batch indices (N).
        :param skips: skip outputs from SetAbstractionSeq.
        :return:
        """
        out = x, pos, batch
        for l, s in zip(self.layers, skips[::-1]):
            out = l(par_embedding, *out, *s)
        return out
