import itertools
import json
from pathlib import Path
import numpy as np
import pandas
import torch
from pandas import DataFrame
from rich.progress import track
from torch import tensor
from torch.utils.data import Dataset
from data_parser import parse_meta, parse_boundary_fields, parse_internal_fields
from dataset.foam_data import FoamData


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


def collate_fn(samples: list[FoamData]) -> FoamData:
    batch_data = torch.stack([s.data for s in samples])
    subdomains = samples[0].domain.keys()
    domain = {subd: torch.stack([s.domain[subd] for s in samples]) for subd in subdomains}
    return FoamData(batch_data, samples[0].labels, domain)


class FoamDataset(Dataset):
    def __init__(self, data_dir, fields, n_internal, n_boundary, n_obs, rng,
                 variable_boundaries=None, normalize_fields=None, meta_dir=None):
        self.n_boundary = n_boundary
        self.n_internal = n_internal
        self.n_obs = n_obs
        self.rng = rng
        self.fields = fields
        self.variable_boundaries = variable_boundaries
        self.dims = ['x', 'y', 'z']
        self.normalize_fields = normalize_fields

        self.samples = [d for d in Path(data_dir).iterdir() if d.is_dir()]

        if normalize_fields is not None:
            self.meta = parse_meta(data_dir if meta_dir is None else meta_dir)
            stats = self.meta['Stats']
            self.normalizers = {}
            for field in normalize_fields['Standardize']:
                field_stats = stats[field]
                self.normalizers[field] = StandardScaler(np.array(field_stats['Std']), np.array(field_stats['Mean']))
            for field in normalize_fields['Scale']:
                field_stats = stats[field]
                self.normalizers[field] = Normalizer(np.array(field_stats['Min']), np.array(field_stats['Max']))

        with open(Path(data_dir).parent / 'min_points.json') as f:
            self.min_points = json.load(f)
        self.min_boundary = sum([self.min_points[k] for k in self.min_points.keys() if k != 'internal'])

        self.check_sample_size()
        self.data = [self.load_case(case) for case in track(self.samples, description='Loading data into memory')]

    def check_sample_size(self):
        min_points = self.min_points['internal']
        if self.n_internal > min_points:
            raise ValueError(f'Cannot sample {self.n_internal} points from {min_points} points!')
        if self.n_boundary > self.min_boundary:
            raise ValueError(f'Cannot sample {self.n_boundary} points from {self.min_boundary} points!')

    def __len__(self):
        return len(self.samples)

    def sample_boundary(self, boundary_fields: DataFrame) -> DataFrame:
        """
        Samples the boundary points. Return internal_fields to avoid sampling.
        :param boundary_fields:
        :return: The samples boundary_fields
        """
        boundary_names = boundary_fields.index.unique()
        tot = sum([self.min_points[bound] for bound in boundary_names])
        sampled_df = []

        n_sampled, cur_index = 0, 0
        for i, bound in enumerate(boundary_names):
            sample_size = int(self.n_boundary * self.min_points[bound] / tot)
            bound_points = len(boundary_fields.loc[bound])

            if i == len(boundary_names) - 1:
                sample_size += self.n_boundary - (sample_size + n_sampled)

            samples = self.rng.choice(bound_points, replace=False, size=sample_size) + cur_index
            sampled_df.append(boundary_fields.iloc[samples])
            n_sampled += sample_size
            cur_index += bound_points

        return pandas.concat(sampled_df)

    def sample_internal(self, internal_fields: DataFrame) -> DataFrame:
        """
        Samples the internal points. Return internal_fields to avoid sampling.
        :param internal_fields:
        :return: The samples internal_fields
        """
        samples = self.rng.choice(len(internal_fields), replace=False, size=self.n_internal)
        return internal_fields.iloc[samples]

    def sample_obs(self, boundary_fields, internal_fields) -> np.ndarray:
        """
        Samples the observations points.
        This function must return a vector whose values are the indices of the observation points.
        It is assumed that internal fields and boundary fields to be concatenated in that order.
        :param boundary_fields:
        :param internal_fields:
        :return: Vector of indices corresponding to observation points.
        """
        return self.rng.choice(len(internal_fields), replace=False, size=self.n_obs)

    def decompose_multidim_label(self, label, size) -> list[str]:
        """
        Extracts labelled dimensions for a multidimensional label.
        Currently supports only x,y,z.
        :param label: the label name
        :param size: the number of components of a label
        :return: the labels for each component of label
        """
        return [label + self.dims[i] for i in range(size)]

    def get_labels(self, domain_fields: DataFrame) -> dict:
        """
        Create labels required by FoamDataset. Fields with more than one dimension are split.
        :param domain_fields:
        :return: the labels
        """
        labels = {}
        sub_labels = {}
        multi_index = domain_fields.columns.to_frame()

        for f in domain_fields.columns.get_level_values(0).unique():
            dims = multi_index.loc[f][1]
            if dims[0] == '':
                labels[f] = None
            else:
                dim = [f'{f}{d}' for d in dims]
                sub_labels[f] = dim
                labels.update(zip(dim, [None] * len(dim)))
        labels.update(sub_labels)
        return labels

    def get_variable_boundaries(self, boundary_fields: DataFrame) -> DataFrame:
        """
        Creates a dataframe of variable boundary conditions.
        NANs are added to unaffected subdomains.
        :param boundary_fields:
        :return:
        """
        result_df = DataFrame(index=boundary_fields.index)
        columns_df = boundary_fields.columns.to_frame()

        for var_field, var_bound in self.variable_boundaries.items():
            new_field_name = f'{var_field}-{var_bound}'
            if var_field in columns_df.index.levels[0]:
                labels = itertools.product([new_field_name], columns_df.loc[var_field][1])
                result_df.loc[var_bound, labels] = boundary_fields.loc[var_bound, var_field].to_numpy()
            else:
                f, dim = var_field[:-1], var_field[-1]
                new_label = [(new_field_name, '')]
                result_df.loc[var_bound, new_label] = boundary_fields.loc[var_bound, [(f, dim)]].to_numpy()

        return result_df.fillna(0)

    def get_domain(self, boundary_fields: DataFrame, internal_fields: DataFrame) -> dict[str, np.ndarray]:
        """
        Creates a domain dictionary whose keys are subdomain names and values are lists of indices.
        It is assumed internal and boundary fields to be concatenated in that order.
        :param boundary_fields:
        :param internal_fields:
        :return:
        """
        n_internal = len(internal_fields)
        domain = {'internal': np.arange(n_internal),
                  'boundary': np.arange(len(boundary_fields)) + n_internal}

        for b in boundary_fields.index.unique():
            b_range = boundary_fields.index.get_loc(b)
            domain[b] = np.arange(b_range.start, b_range.stop) + n_internal
        return domain

    def normalize(self, fields: DataFrame):
        """
        Scales or standardize fields using the normalizers passed to the constructor.
        The fields are normalized in-place.
        :param fields: Fields to normalize
        :return:
        """
        for f, norm in self.normalizers.items():
            fields[f] = norm.transform(fields[f].to_numpy())

    def load_case(self, case_dir) -> FoamData:
        boundary_fields = parse_boundary_fields(case_dir, *self.fields)
        internal_fields = parse_internal_fields(case_dir, *self.fields)

        # Normalize
        if self.normalize_fields is not None:
            self.normalize(internal_fields)
            self.normalize(boundary_fields)

        # Sampling
        boundary_fields = self.sample_boundary(boundary_fields)
        internal_fields = self.sample_internal(internal_fields)

        # Add variable boundary conditions
        if self.variable_boundaries is not None:
            variable_fields = self.get_variable_boundaries(boundary_fields)
            boundary_fields = pandas.concat([boundary_fields, variable_fields], axis=1)

        domain_data = pandas.concat([internal_fields, boundary_fields]).fillna(0)

        domain = self.get_domain(boundary_fields, internal_fields)
        labels = self.get_labels(domain_data)

        domain['obs'] = self.sample_obs(boundary_fields, internal_fields)
        domain = {d: torch.tensor(s, dtype=torch.int64) for d, s in domain.items()}

        return FoamData(tensor(domain_data.to_numpy(), dtype=torch.float), labels, domain)

    def __getitem__(self, item) -> FoamData:
        return self.data[item]
