import json
from pathlib import Path
import numpy as np
import torch
from torch_geometric.utils import unbatch
from torch import tensor, Tensor
from torch_geometric.data import InMemoryDataset, Data
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
        self.standard_scaler = StandardScaler(
            np.array(self.meta['Std']['Points'] + self.meta['Std']['U'] + [self.meta['Std']['p']]),
            np.array(self.meta['Mean']['Points'] + self.meta['Mean']['U'] + [self.meta['Mean']['p']]),
        )
        self.d_normalizer = Normalizer(np.zeros(2), np.array(self.meta['Coefs']['d']['Max']))
        self.f_normalizer = Normalizer(np.zeros(2), np.array(self.meta['Coefs']['f']['Max']))

        with open(Path(data_dir).parent / 'min_points.json') as f:  self.min_points = json.load(f)
        self.min_boundary = sum(list(self.min_points.values())[1:])

        self.domain_dict = self.get_domain_map()

        super().__init__(data_dir)

        self.check_sample_size()
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def get_domain_map(self):
        min_tot = sum([self.min_points[k] for k in self.min_points.keys() if k != 'internal'])
        domain_map = {'internal': slice(None, self.n_internal), 'boundary': slice(self.n_internal, None)}

        exclude_keys = domain_map.keys()
        prev_start = self.n_internal
        for k, m in self.min_points.items():
            if k in exclude_keys: continue
            next_start = prev_start + int(self.n_boundary * m / min_tot)
            domain_map[k] = slice(prev_start, next_start)
            prev_start = next_start
        return domain_map

    def check_sample_size(self):
        min_points = self.min_points['internal']
        if self.n_internal > min_points:
            raise ValueError(f'Cannot sample {self.n_internal} points from {min_points} points!')
        if self.n_boundary > self.min_boundary:
            raise ValueError(f'Cannot sample {self.n_boundary} points from {self.min_boundary} points!')

    def extract_inlet_conditions(self, boundary_data: dict[str:np.ndarray]) -> np.ndarray:
        inlet_data = []
        for key in boundary_data.keys():
            if key == 'inlet':
                inlet_ux = boundary_data[key][..., 4:5]
                inlet_ux = self.standard_scaler[2:3].transform(inlet_ux)
                inlet_data.append(inlet_ux)
            else:
                inlet_data.append(np.zeros((len(boundary_data[key]), 1)))
        return np.concatenate(inlet_data)

    def get_boundaries_samples(self, boundary_dict: dict):
        samples = []
        cur_start = 0
        tot = sum([self.min_points[k] for k in boundary_dict.keys()])

        for k, b in boundary_dict.items():
            n = self.n_boundary * self.min_points[k] / tot
            index = np.random.choice(len(b), replace=False, size=int(n)) + cur_start
            samples.extend(index.tolist())
            cur_start += len(b)
        return samples

    def reorder_data(self, data: np.ndarray) -> np.ndarray:
        points, pde, zones, d = data[..., 0:2], data[..., 4:7], data[..., 8:9], data[..., 9:11]
        moment, div = data[..., 2:4], data[..., 7:8]
        inlet = data[..., 11:12]
        return np.concatenate([points, pde, zones, d, inlet, moment, div], axis=1)

    def load_case(self, case_dir):
        b_dict = parse_boundary(case_dir, ['momentError', 'U'], ['p', 'div(phi)'])
        inlet_data = self.extract_inlet_conditions(b_dict)

        b_data = np.concatenate(list(b_dict.values()))
        b_data = np.concatenate([b_data, inlet_data], axis=-1)

        b_samples = self.get_boundaries_samples(b_dict)
        b_data = b_data[b_samples]

        i_data = parse_internal_mesh(case_dir, 'momentError', 'U', 'p', 'div(phi)')
        i_samples = np.random.choice(len(i_data), replace=False, size=self.n_internal)
        i_data = i_data[i_samples]
        i_data = np.concatenate([i_data, np.zeros((len(i_data), 1))], axis=-1)

        data = np.concatenate((i_data, b_data))

        obs_samples = np.random.choice(len(i_data), replace=False, size=self.n_obs)

        data = self.reorder_data(data)

        # Do not standardize zones indices and residuals
        data[:, 0:-7] = self.standard_scaler.transform(data[:, 0:-7])
        data[:, 6:8] = self.d_normalizer.transform(data[:, 6:8])

        data = tensor(data, dtype=torch.float)

        return (data[..., 0:2],  # pos
                data[..., 5:9],  # zones-d-inlet
                data[..., 2:5],  # y-pde
                data[..., 9:],  # residuals
                tensor(obs_samples, dtype=torch.int64))

    def process(self):
        data = []
        for case in Path(self.raw_dir).iterdir():
            if not case.is_dir():
                continue
            case_data = FoamData(*self.load_case(case))
            data.append(case_data)

        self.save(data, self.processed_paths[0])
