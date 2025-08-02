from pathlib import Path
import numpy as np
import torch
from rich.progress import track
from torch import tensor, Tensor
from torch.utils.data import Dataset
from data_parser import parse_meta, parse_boundary, parse_internal_mesh
from visualization import plot_fields


class PdeData:
    def __init__(self, data: Tensor | np.ndarray):
        self.data = data

    @property
    def u(self) -> Tensor | np.ndarray:
        return self.data[..., 0:3]

    @property
    def ux(self) -> Tensor | np.ndarray:
        return self.data[..., 0:1]

    @property
    def uy(self) -> Tensor | np.ndarray:
        return self.data[..., 1:2]

    @property
    def uz(self) -> Tensor | np.ndarray:
        return self.data[..., 2:3]

    @property
    def p(self) -> Tensor | np.ndarray:
        return self.data[..., 3:4]

    def numpy(self):
        return PdeData(self.data.numpy(force=True))


class FoamData:
    def __init__(self, batch: tuple | list):
        self.data, self.obs_samples = batch
        self.points = self.data[..., 0:3]
        self.pde = PdeData(self.data[..., 3:7])

    @property
    def zones_ids(self) -> Tensor | np.ndarray:
        return self.data[..., 7:8]

    @property
    def obs_ux(self) -> Tensor | np.ndarray:
        return self.pde.ux.gather(1, self.obs_samples)

    @property
    def obs_uy(self) -> Tensor | np.ndarray:
        return self.pde.uy.gather(1, self.obs_samples)

    @property
    def obs_uz(self) -> Tensor | np.ndarray:
        return self.pde.uz.gather(1, self.obs_samples)

    @property
    def obs_p(self) -> Tensor | np.ndarray:
        return self.pde.p.gather(1, self.obs_samples)

    @property
    def mom_x(self):
        return self.data[..., 7:8]

    @property
    def mom_y(self):
        return self.data[..., 8:9]

    @property
    def mom_z(self):
        return self.data[..., 9:10]

    @property
    def div(self):
        return self.data[..., -1:]

    def numpy(self):
        return FoamData([self.data.numpy(force=True), self.obs_samples.numpy(force=True)])


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

    def to(self, *args, **kwargs):
        return StandardScaler(torch.tensor(self.std).to(*args, *kwargs),
                              torch.tensor(self.mean).to(*args, *kwargs))


class FoamDataset(Dataset):
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

    def __len__(self):
        return len(self.samples)

    def reorder_data(self, data: np.ndarray) -> np.ndarray:
        return np.concatenate((data[:, 0:3], data[:, 6:10], data[:, 11:12], data[:, 3:6], data[:, 10:11]), axis=1)

    def load_case(self, case_dir):
        b_data = parse_boundary(case_dir, ['momentError', 'U'], ['p', 'div(phi)'])
        b_samples = np.random.choice(len(b_data), replace=False, size=self.n_boundary)
        b_data = b_data[b_samples]

        i_data = (parse_internal_mesh(case_dir, 'momentError', 'U', 'p', 'div(phi)'))
        i_samples = np.random.choice(len(i_data), replace=False, size=self.n_internal)
        i_data = i_data[i_samples]

        data = np.concatenate((i_data, b_data))
        data = self.reorder_data(data)

        obs_samples = np.random.choice(len(i_data), replace=False, size=self.n_obs)

        # Do not standardize zones indices
        data[:, 0:-5] = self.standard_scaler.transform(data[:, 3:-5])

        return (tensor(data, dtype=torch.float),
                tensor(obs_samples, dtype=torch.int64).unsqueeze(dim=1))

    def __getitem__(self, item) -> tuple[Tensor, Tensor]:
        return self.data[item]
