from pathlib import Path
import numpy as np
import torch
from rich.progress import track
from torch import tensor, Tensor
from torch.utils.data import Dataset
from data_parser import parse_meta, parse_boundary, parse_internal_mesh


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


class FoamData:
    def __init__(self, batch: tuple | list):
        self.data, self.obs_samples = batch
        self.points = self.data[..., 0:2]
        self.pde = PdeData(self.data[..., 2:5])

    @property
    def zones_ids(self) -> Tensor | np.ndarray:
        return self.data[..., 5:6]

    @property
    def obs_ux(self) -> Tensor | np.ndarray:
        return self.pde.ux.gather(1, self.obs_samples)

    @property
    def obs_uy(self) -> Tensor | np.ndarray:
        return self.pde.uy.gather(1, self.obs_samples)

    @property
    def obs_p(self) -> Tensor | np.ndarray:
        return self.pde.p.gather(1, self.obs_samples)

    def numpy(self):
        return FoamData([self.data.numpy(force=True), self.obs_samples.numpy(force=True)])


class FoamDataset(Dataset):
    def __init__(self, data_dir: str, n_internal: int, n_boundary: int, n_obs: int, meta=None):
        self.n_boundary = n_boundary
        self.n_internal = n_internal
        self.n_obs = n_obs
        self.samples = [d for d in Path(data_dir).iterdir() if d.is_dir()]
        self.meta = parse_meta(data_dir) if meta is None else meta
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

    def __len__(self):
        return len(self.samples)

    def load_case(self, case_dir):
        b_points, b_u, b_p, b_zones_ids = parse_boundary(case_dir, ['U'], ['p'])
        b_samples = np.random.choice(len(b_points), replace=False, size=self.n_boundary)
        b_points, b_u, b_p, b_zones_ids = b_points[b_samples], b_u[b_samples], b_p[b_samples], b_zones_ids[b_samples]

        i_points, i_u, i_p, i_zones_ids = (parse_internal_mesh(case_dir, 'U', 'p'))
        i_samples = np.random.choice(len(i_points), replace=False, size=self.n_internal)
        i_points, i_u, i_p, i_zones_ids = i_points[i_samples], i_u[i_samples], i_p[i_samples], i_zones_ids[i_samples]

        points = np.concatenate((i_points, b_points))
        u = np.concatenate((i_u, b_u))
        p = np.concatenate((i_p, b_p))
        zones_ids = np.concatenate((i_zones_ids, b_zones_ids))

        data = np.concatenate((points, u, p, zones_ids), axis=1)
        obs_samples = np.random.choice(len(i_points), replace=False, size=self.n_obs)

        # Do not standardize zones indices
        data[:, 0:-1] = self.standard_scaler.transform(data[:, 0:-1])

        return (tensor(data, dtype=torch.float),
                tensor(obs_samples, dtype=torch.int64).unsqueeze(dim=1))

    def __getitem__(self, item) -> tuple[Tensor, Tensor]:
        return self.data[item]


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
