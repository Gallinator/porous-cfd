from torchvision.ops import MLP
import torch
from torch import nn, Tensor
from torch_cluster import fps, radius
from torch_geometric.nn import PointNetConv, global_max_pool, MLP
from torch_geometric.utils import unbatch


class Branch(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.linear = MLP(in_channels, hidden_channels, activation_layer=nn.Tanh)

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
        self.linear = MLP(in_channels, hidden_channels, activation_layer=nn.Tanh)

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
    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.Tanh()
        )
        if dropout:
            self.linear.append(nn.Dropout(0.15))

    def forward(self, x: Tensor, par_embedding: Tensor):
        return self.linear(x) * par_embedding


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
        self.conv1 = SetAbstraction(0.5, 0.5, MLP([65 + 2, 64], act=nn.Tanh(), norm=None))
        self.conv2 = SetAbstraction(0.25, 1.0, MLP([64 + 2, 128], act=nn.Tanh(), norm=None))
        self.conv3 = GlobalSetAbstraction(MLP([128 + 2, 1024], act=nn.Tanh(), norm=None))

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
