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
    def ux(self) -> Tensor:
        return self.data[..., 0:1]

    @property
    def uy(self) -> Tensor:
        return self.data[..., 1:2]

    @property
    def p(self) -> Tensor:
        return self.data[..., 2:3]

    def numpy(self):
        return PdeData(self.data.numpy(force=True))


class FoamData:
    def __init__(self, batch: Tensor | np.ndarray):
        self.data = batch
        self.points = batch[..., 0:2]
        self.pde = PdeData(batch[..., 2:5])

    def numpy(self):
        return FoamData(self.data.numpy(force=True))


class FoamDataset(Dataset):
    def __init__(self, data_dir: str, n_internal: int, n_boundary: int):
        self.n_boundary = n_boundary
        self.n_internal = n_internal
        self.samples = [d for d in Path(data_dir).iterdir() if d.is_dir()]
        self.meta = parse_meta(data_dir)
        self.check_sample_size()

        self.data = [self.load_case(case) for case in track(self.samples, description='Loading data into memory')]

    def check_sample_size(self):
        data_min_points = self.meta['Internal']['Min points']
        if self.n_internal > data_min_points:
            raise ValueError(f'Cannot sample {self.n_internal} points from {data_min_points} points!')
        data_min_points = self.meta['Boundary']['Min points']
        if self.n_boundary > data_min_points:
            raise ValueError(f'Cannot sample {self.n_boundary} points from {data_min_points} points!')

    def __len__(self):
        return len(self.samples)

    def create_manufactured_solutions(self, points: np.array) -> tuple[np.array, np.array, np.array]:
        points_x, points_y = points[:, 0:1], points[:, 1:2]

        # Create manufactures data
        u_x = np.sin(points_y) * np.cos(points_x)
        u_y = -np.sin(points_x) * np.cos(points_y)
        u = np.concatenate([u_x, u_y], axis=1)

        p = -1 / 4 * (np.cos(2 * points_x) + np.cos(2 * points_y))

        f_x = 2 * 0.01 * np.cos(points_x) * np.sin(points_y)
        f_y = -2 * 0.01 * np.sin(points_x) * np.cos(points_y)
        f = np.concatenate([f_x, f_y], axis=1)

        return u, p, f

    def load_case(self, case_dir):
        b_points, _, _ = parse_boundary(case_dir)
        b_samples = np.random.choice(len(b_points), replace=False, size=self.n_boundary)
        b_points = b_points[b_samples]

        i_points = parse_internal_mesh(case_dir)[0]
        i_samples = np.random.choice(len(i_points), replace=False, size=self.n_internal)
        i_points = i_points[i_samples]

        points = np.concatenate((i_points, b_points))
        u, p, f = self.create_manufactured_solutions(points)

        return (tensor(points, dtype=torch.float),
                tensor(u, dtype=torch.float),
                tensor(p, dtype=torch.float),
                tensor(f, dtype=torch.float))

    def __getitem__(self, item) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.data[item]
