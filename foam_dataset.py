from pathlib import Path
import numpy as np
import torch
import torch_geometric
from torch import tensor, Tensor
from torch_geometric.data import InMemoryDataset

from data_parser import parse_meta, parse_boundary, parse_internal_mesh


def get_domain_dict(n_internal, n_boundary):
    boundary_subdomain_size = int(n_boundary / 5)
    inlet_start = n_internal
    interface_start = inlet_start + boundary_subdomain_size
    outlet_start = interface_start + boundary_subdomain_size
    walls_start = outlet_start + boundary_subdomain_size
    return {
        'internal': slice(None, n_internal),
        'boundary': slice(n_internal, None),
        'inlet': slice(inlet_start, interface_start),
        'interface': slice(interface_start, outlet_start),
        'outlet': slice(outlet_start, walls_start),
        'walls': slice(walls_start, None)
    }


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


class Normalizer:
    def __init__(self, min, max):
        super().__init__()
        self.min = min
        self.max = max
        self.range = max - min

    def transform(self, data):
        return (data - self.min) / self.range

    def inverse_transform(self, data):
        return self.min + self.range * data

    def __getitem__(self, item):
        return Normalizer(self.min[item], self.max[item])

    def to_torch(self, device=None):
        return StandardScaler(torch.tensor(self.min, device=device), torch.tensor(self.max, device=device))


class DomainData(Data):
    def __init__(self, pos=None, x=None, y=None, residuals=None, domain_dict=None, batch=None):
        super().__init__(pos=pos, x=x, y=y)
        self.residuals = residuals
        self.domain_dict = domain_dict
        self.batch = batch

    @property
    def pde(self):
        return PdeData(self.y, self.batch if hasattr(self, 'batch') else None,
                       self.domain_dict if hasattr(self, 'domain_dict') else None)

    @property
    def zones_ids(self):
        return self.x[..., 0:1]

    @property
    def d(self):
        return self.x[..., 1:3]

    @property
    def inlet_ux(self):
        return self.x[..., 3:4]

    @property
    def mom_x(self):
        return self.residuals[..., 0:1]

    @property
    def mom_y(self):
        return self.residuals[..., 1:2]

    @property
    def div(self):
        return self.residuals[..., 2:]

    def slice(self, item):
        data = torch.cat([self.pos, self.x, self.y, self.residuals], dim=-1)
        if self.batch is not None:
            data = torch.cat([data, self.batch.unsqueeze(1)], dim=-1)
            batched_data = torch.stack(unbatch(data, self.batch))
            sliced_data = batched_data[..., self.domain_dict[item], :]
            data = torch.cat([*sliced_data])
            sliced_batch = data[..., -1]
        else:
            data = data[..., self.domain_dict[item], :]
            sliced_batch = self.batch

        return DomainData(data[..., 0:2],  # pos
                          data[..., 2:6],  # zones
                          data[..., 6:9],  # y-pde
                          data[..., 9:],  # residuals,
                          self.domain_dict,
                          sliced_batch.to(dtype=torch.int64))


class PdeData:
    def __init__(self, data: Tensor | np.ndarray, batch=None, domain_dict=None):
        self.data = data
        self.domain_dict = domain_dict
        self.batch = batch

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

    def slice(self, item):
        if self.batch is not None:
            data = torch.cat([self.data, self.batch.unsqueeze(1)], dim=-1)
            batched_data = torch.stack(unbatch(data, self.batch))
            sliced_data = batched_data[..., self.domain_dict[item], :]
            data = torch.cat([*sliced_data])
            sliced_batch = data[..., -1]
        else:
            data = self.data[..., self.domain_dict[item], :]
            sliced_batch = self.batch

        return PdeData(data[..., :-1], sliced_batch.to(dtype=torch.int64), self.domain_dict)

    def numpy(self):
        return PdeData(self.data.numpy(force=True))


class FoamData(DomainData):
    def __init__(self, pos=None, x=None, y=None, residuals=None, obs_index=None, domain_dict: dict = None):
        super().__init__(pos=pos, x=x, y=y, residuals=residuals, domain_dict=domain_dict)
        self.obs_index = obs_index

    @property
    def obs(self):
        return DomainData(self.pos[self.obs_index, :],
                          self.x[self.obs_index, :],
                          self.y[self.obs_index, :],
                          self.residuals[self.obs_index, :],
                          self.domain_dict,
                          self.batch)


class FoamDataset(InMemoryDataset):
    def __init__(self, data_dir: str, n_internal: int, n_boundary: int, n_obs: int, meta_dir=None):
        self.n_boundary = n_boundary
        self.n_internal = n_internal
        self.n_obs = n_obs

        self.meta = parse_meta(f'{data_dir}/raw' if meta_dir is None else meta_dir)
        self.check_sample_size()
        self.standard_scaler = StandardScaler(
            np.array(self.meta['Std']['Points'] + self.meta['Std']['U'] + [self.meta['Std']['p']]),
            np.array(self.meta['Mean']['Points'] + self.meta['Mean']['U'] + [self.meta['Mean']['p']]),
        )
        self.d_normalizer = Normalizer(np.zeros(2), np.array(self.meta['Darcy']['Max']))

        super().__init__(data_dir)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def check_sample_size(self):
        data_min_points = self.meta['Min points']['Internal']
        if self.n_internal > data_min_points:
            raise ValueError(f'Cannot sample {self.n_internal} points from {data_min_points} points!')
        data_min_points = self.meta['Min points']['Boundary']
        if self.n_boundary > data_min_points:
            raise ValueError(f'Cannot sample {self.n_boundary} points from {data_min_points} points!')

    def reorder_data(self, data: np.ndarray) -> np.ndarray:
        points, moment, pde, div = data[..., 0:2], data[..., 2:4], data[..., 4:7], data[..., 7:8]
        zones = data[..., 8:]
        return np.concatenate([points, pde, zones, moment, div], axis=-1)

    def load_case(self, case_dir):
        b_data = parse_boundary(case_dir, ['momentError', 'U'], ['p', 'div(phi)'])

        b_data = np.concatenate(list(b_data.values()))
        b_samples = np.random.choice(len(b_data), replace=False, size=self.n_boundary)
        b_data = b_data[b_samples]

        i_data = (parse_internal_mesh(case_dir, 'momentError', 'U', 'p', 'div(phi)'))
        i_samples = np.random.choice(len(i_data), replace=False, size=self.n_internal)
        i_data = i_data[i_samples]

        data = np.concatenate((i_data, b_data))

        obs_samples = np.random.choice(len(i_data), replace=False, size=self.n_obs)

        data = self.reorder_data(data)

        # Do not standardize zones indices and residuals
        data[:, 0:-4] = self.standard_scaler.transform(data[:, 0:-4])
        data = tensor(data, dtype=torch.float)

        return (data[..., 0:2],  # pos
                data[..., 5:6],  # zones
                data[..., 2:5],  # y-pde
                data[..., -3:],  # residuals
                tensor(obs_samples, dtype=torch.int64))

    def process(self):
        data = []
        for case in Path(self.raw_dir).iterdir():
            if not case.is_dir():
                continue
            pos, x, y, residuals, obs_ids = self.load_case(case)
            data.append(FoamData(pos, x, y, obs_ids, residuals))

        self.save(data, self.processed_paths[0])
