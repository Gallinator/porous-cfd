from pathlib import Path
import numpy as np
import torch
from numpy.random import default_rng
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
        self.data = batch
        self.points = self.data[..., 0:2]
        self.pde = PdeData(self.data[..., 2:5])

    @property
    def zones_ids(self) -> Tensor | np.ndarray:
        return self.data[..., 7:8]

    @property
    def fx(self) -> Tensor | np.ndarray:
        return self.data[..., 5:6]

    @property
    def fy(self) -> Tensor | np.ndarray:
        return self.data[..., 6:7]

    def numpy(self):
        return FoamData([self.data.numpy(force=True)])


class FoamDataset(Dataset):
    def __init__(self, data_dir: str, n_internal: int, n_boundary: int, meta_dir=None, rng=default_rng()):
        self.n_boundary = n_boundary
        self.n_internal = n_internal
        self.rng = rng
        self.samples = [d for d in Path(data_dir).iterdir() if d.is_dir()]
        self.meta = parse_meta(data_dir if meta_dir is None else meta_dir)
        self.check_sample_size()

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

    def create_manufactured_solutions(self, points: np.array, porous: np.array) -> tuple[np.array, np.array, np.array]:
        points_x, points_y = points[:, 0:1], points[:, 1:2]

        # Create manufactures data
        u_x = np.sin(points_y) * np.cos(points_x)
        u_y = -np.sin(points_x) * np.cos(points_y)
        u = np.concatenate([u_x, u_y], axis=1)

        p = -1 / 4 * (np.cos(2 * points_x) + np.cos(2 * points_y))

        f_x = 2 * 0.01 * np.cos(points_x) * np.sin(points_y)
        f_y = -2 * 0.01 * np.sin(points_x) * np.cos(points_y)

        f_x += 0.01 * 100 * u_x * porous
        f_y += 0.01 * 100 * u_y * porous

        f = np.concatenate([f_x, f_y], axis=1)

        return u, p, f

    def load_case(self, case_dir):
        b_data = parse_boundary(case_dir, [], [])
        b_samples = self.rng.choice(len(b_data), replace=False, size=self.n_boundary)
        b_data = b_data[b_samples]

        i_data = (parse_internal_mesh(case_dir, ))
        i_samples = self.rng.choice(len(i_data), replace=False, size=self.n_internal)
        i_data = i_data[i_samples]

        data = np.concatenate((i_data, b_data))

        points = data[:, 0:2]
        zones_ids = data[..., 7:8]

        u, p, f = self.create_manufactured_solutions(points, zones_ids)
        data = np.concatenate([points, u, p, f, zones_ids], axis=-1)

        return tensor(data, dtype=torch.float)

    def __getitem__(self, item) -> Tensor:
        return self.data[item]
