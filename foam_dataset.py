from pathlib import Path
import numpy as np
import torch
from rich.progress import track
from torch import tensor, Tensor
from torch.utils.data import Dataset
from data_parser import parse_meta, parse_boundary, parse_internal_mesh


class PredictedDataBatch:
    def __init__(self, u: Tensor | np.ndarray, p: Tensor | np.ndarray):
        self.u = u
        self.p = p

    @property
    def ux(self) -> Tensor:
        return self.u[:, :, 0:1]

    @property
    def uy(self) -> Tensor:
        return self.u[:, :, 1:2]

    def numpy(self):
        return PredictedDataBatch(self.u.numpy(force=True), self.p.numpy(force=True))


class FoamDataBatch:
    def __init__(self, batch: tuple | list):
        self.points, self.u, self.p, self.f, self.porous_zone = batch

    @property
    def ux(self) -> Tensor:
        return self.u[:, :, 0:1]

    @property
    def uy(self) -> Tensor:
        return self.u[:, :, 1:2]

    @property
    def fx(self) -> Tensor:
        return self.f[:, :, 0:1]

    @property
    def fy(self) -> Tensor:
        return self.f[:, :, 1:2]

    def numpy(self):
        return FoamDataBatch(
            (self.points.numpy(force=True),
             self.u.numpy(force=True),
             self.p.numpy(force=True),
             self.f.numpy(force=True),
             self.porous_zone.numpy(force=True))
        )


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
        b_points, _, _, b_porous = parse_boundary(case_dir)
        b_samples = np.random.choice(len(b_points), replace=False, size=self.n_boundary)
        b_points, b_porous = b_points[b_samples], b_porous[b_samples]

        i_points, i_porous = parse_internal_mesh(case_dir)
        i_samples = np.random.choice(len(i_points), replace=False, size=self.n_internal)
        i_points, i_porous = i_points[i_samples], i_porous[i_samples]

        points = np.concatenate((i_points, b_points))
        porous = np.concatenate((i_porous, b_porous))

        u, p, f = self.create_manufactured_solutions(points, porous)

        return (tensor(points, dtype=torch.float),
                tensor(u, dtype=torch.float),
                tensor(p, dtype=torch.float),
                tensor(f, dtype=torch.float),
                tensor(porous, dtype=torch.float))

    def __getitem__(self, item) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
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
        return StandardScaler(self.std[item], self.std[item])
