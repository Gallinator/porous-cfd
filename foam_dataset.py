import json
from pathlib import Path

import numpy as np
import torch
from rich.progress import track
from torch import tensor, Tensor
from torch.utils.data import Dataset

from data_parser import parse_meta, parse_boundary, parse_internal_mesh
from visualization import plot_fields


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

    def load_case(self, case_dir):
        b_points, b_u, b_p = parse_boundary(case_dir)
        b_samples = np.random.choice(len(b_points), replace=False, size=self.n_boundary)
        b_points, b_u, b_p = b_points[b_samples], b_u[b_samples], b_p[b_samples]

        i_points, i_u, i_p = parse_internal_mesh(case_dir, "U", "p")
        i_samples = np.random.choice(len(i_points), replace=False, size=self.n_internal)
        i_points, i_u, i_p = i_points[i_samples], i_u[i_samples], i_p[i_samples]

        points = np.concatenate((i_points, b_points))
        u = np.concatenate((i_u, b_u))
        p = np.concatenate((i_p, b_p))

        return tensor(points, dtype=torch.float), tensor(u, dtype=torch.float), tensor(p, dtype=torch.float)

    def __getitem__(self, item) -> tuple[Tensor, Tensor, Tensor]:
        return self.data[item]
