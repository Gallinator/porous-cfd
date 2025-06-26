from pathlib import Path
import numpy as np
import torch
import torch_geometric
from torch import tensor, Tensor
from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import RadiusGraph

from data_parser import parse_meta, parse_boundary, parse_internal_mesh


class StandardScaler:
    def __init__(self, std, mean):
        super().__init__()
        self.std = std
        self.mean = mean

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return self.std * data + self.mean

    def __getitem__(self, item):
        return StandardScaler(self.std[item], self.mean[item])

    def to_torch(self, device=None):
        return StandardScaler(torch.tensor(self.std, device=device), torch.tensor(self.mean, device=device))


class PdeData:
    def __init__(self, data: Tensor | np.ndarray):
        self.data = data

    @property
    def u(self) -> Tensor | np.ndarray:
        return self.data[..., 0:2]

    @property
    def ux(self) -> Tensor | np.ndarray:
        return self.data[..., 0:1]

    @property
    def uy(self) -> Tensor | np.ndarray:
        return self.data[..., 1:2]

    @property
    def p(self) -> Tensor | np.ndarray:
        return self.data[..., 2:3]

    def numpy(self):
        return PdeData(self.data.numpy(force=True))


class FoamData(torch_geometric.data.Data):
    def __init__(self, pos=None, x=None, y=None, obs_index=None):
        super().__init__(pos=pos, x=x, y=y)
        self.obs_index = obs_index

    @property
    def zones_ids(self) -> Tensor | np.ndarray:
        return self.x

    @property
    def pde(self):
        return PdeData(self.y)

    @property
    def obs_ux(self) -> Tensor | np.ndarray:
        return self.pde.ux[self.obs_samples, :]

    @property
    def obs_uy(self) -> Tensor | np.ndarray:
        return self.pde.uy[self.obs_samples, :]

    @property
    def obs_p(self) -> Tensor | np.ndarray:
        return self.pde.p[self.obs_samples, :]


class FoamDataset(InMemoryDataset):
    def __init__(self, data_dir: str, n_internal: int, n_boundary: int, n_obs: int, meta_dir=None):
        self.n_boundary = n_boundary
        self.n_internal = n_internal
        self.n_obs = n_obs
        self.samples = [d for d in Path(data_dir).iterdir() if d.is_dir()]
        self.meta = parse_meta(data_dir if meta_dir is None else meta_dir)
        self.check_sample_size()
        self.standard_scaler = StandardScaler(
            np.array(self.meta['Std']['Points'] + self.meta['Std']['U'] + [self.meta['Std']['p']]),
            np.array(self.meta['Mean']['Points'] + self.meta['Mean']['U'] + [self.meta['Mean']['p']]),
        )

        self.data = [self.load_case(case) for case in track(self.samples, description='Loading data into memory')]

    def check_sample_size(self):
        data_min_points = self.meta['Min points']['Internal']
        if self.n_internal > data_min_points:
            raise ValueError(f'Cannot sample {self.n_internal} points from {data_min_points} points!')
        data_min_points = self.meta['Min points']['Boundary']
        if self.n_boundary > data_min_points:
            raise ValueError(f'Cannot sample {self.n_boundary} points from {data_min_points} points!')

    def reorder_data(self, data: np.ndarray) -> np.ndarray:
        return np.concatenate((data[:, 0:2], data[:, 4:7], data[:, 8:9], data[:, 2:4], data[:, 7:8]), axis=1)

    def load_case(self, case_dir):
        b_data = parse_boundary(case_dir, ['U'], ['p'])
        b_samples = np.random.choice(len(b_data), replace=False, size=self.n_boundary)
        b_data = b_data[b_samples]

        i_data = (parse_internal_mesh(case_dir, 'U', 'p'))
        i_samples = np.random.choice(len(i_data), replace=False, size=self.n_internal)
        i_data = i_data[i_samples]

        data = np.concatenate((i_data, b_data))

        obs_samples = np.random.choice(len(i_data), replace=False, size=self.n_obs)

        # Do not standardize zones indices
        data[:, 0:-1] = self.standard_scaler.transform(data[:, 0:-1])

        return (tensor(data[..., 0:2], dtype=torch.float),
                tensor(data[..., -1:], dtype=torch.float),
                tensor(data[..., 2:5], dtype=torch.float),
                tensor(obs_samples, dtype=torch.int64))

