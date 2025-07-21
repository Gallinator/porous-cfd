from pathlib import Path
import numpy as np
import torch
from rich.progress import track
from torch import tensor, Tensor
from torch.utils.data import Dataset
from data_parser import parse_meta, parse_boundary, parse_internal_mesh


class DomainData:
    def __init__(self, data: Tensor):
        self.data = data
        self.points = self.data[..., 0:2]
        self.pde = PdeData(self.data[..., 2:5])

    @property
    def zones_ids(self) -> Tensor | np.ndarray:
        return self.data[..., 5:6]

    @property
    def d(self) -> Tensor | np.ndarray:
        return self.data[..., 6:8]

    @property
    def inlet_ux(self):
        return self.data[..., 8:9]

    @property
    def mom_x(self):
        return self.data[..., 9:10]

    @property
    def mom_y(self):
        return self.data[..., 10:11]

    @property
    def div(self):
        return self.data[..., 11:12]


class PdeData:
    def __init__(self, data: Tensor | np.ndarray, domain_dict=None):
        self.data = data
        self.domain_dict = domain_dict

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

    def __getitem__(self, item):
        if self.domain_dict is None:
            raise NotImplementedError('Subdomain indexing is not available')
        return PdeData(self.data[..., self.domain_dict[item], :])

    def numpy(self):
        return PdeData(self.data.numpy(force=True))


class FoamData(DomainData):
    def __init__(self, batch: list, domain_dict=None):
        super().__init__(batch[0])
        self.obs_samples = batch[1]
        self.domain_dict = domain_dict

    @property
    def obs(self):
        return DomainData(torch.gather(self.data, 1, self.obs_samples))

    def __getitem__(self, item):
        if self.domain_dict is None:
            raise NotImplementedError('Subdomain indexing is not available')
        return DomainData(self.data[..., self.domain_dict[item], :])

    def numpy(self):
        return FoamData([self.data.numpy(force=True), self.obs_samples.numpy(force=True)], self.domain_dict)


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
        self.d_normalizer = Normalizer(np.zeros(2), np.array(self.meta['Darcy']['Max']))

        with open(Path(data_dir).parent / 'min_points.json') as f:  self.min_points = json.load(f)
        self.domain_dict = self.get_domain_map()

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

    def get_domain_map(self):
        boundary_subdomain_size = int(self.n_boundary / 5)
        inlet_start = self.n_internal
        interface_start = inlet_start + boundary_subdomain_size
        outlet_start = interface_start + boundary_subdomain_size
        walls_start = outlet_start + boundary_subdomain_size
        return {
            'internal': slice(None, self.n_internal),
            'boundary': slice(self.n_internal, None),
            'inlet': slice(inlet_start, interface_start),
            'interface': slice(interface_start, outlet_start),
            'outlet': slice(outlet_start, walls_start),
            'walls': slice(walls_start, None)
        }

    def reorder_data(self, data: np.ndarray) -> np.ndarray:
        points, pde, zones, d = data[..., 0:2], data[..., 4:7], data[..., 8:9], data[..., 9:11]
        moment, div = data[..., 2:4], data[..., 7:8]
        inlet = data[..., 11:12]
        return np.concatenate([points, pde, zones, d, inlet, moment, div], axis=1)

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

    def extend_gather_indices(self, index, n_features: int) -> torch.Tensor:
        return tensor(index, dtype=torch.int64).unsqueeze(dim=1).repeat(1, n_features)

    def get_boundaries_samples(self, boundary_dict: dict):
        samples = []
        cur_start = 0
        if self.n_boundary % 5 != 0:
            print(
                f'Warning: attempting to sample {self.n_boundary} boundary points but it is only possible to sample {self.n_boundary - self.n_boundary % 5} points')
        for k, b in boundary_dict.items():
            n = self.n_boundary * (2 if k == 'walls' else 1) / 5
            index = np.random.choice(len(b), replace=False, size=int(n)) + cur_start
            samples.extend(index.tolist())
            cur_start += len(b)
        return samples

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
        data = self.reorder_data(data)

        obs_samples = np.random.choice(len(i_data), replace=False, size=self.n_obs)
        obs_samples = self.extend_gather_indices(obs_samples, data.shape[-1])

        # Do not standardize zones indices
        data[:, 0:-7] = self.standard_scaler.transform(data[:, 0:-7])
        data[:, 6:8] = self.d_normalizer.transform(data[:, 6:8])

        return tensor(data, dtype=torch.float), obs_samples

    def __getitem__(self, item) -> tuple[Tensor, Tensor]:
        return self.data[item]
