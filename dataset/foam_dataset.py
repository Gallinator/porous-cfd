import itertools
import json
from pathlib import Path
import numpy as np
import pandas
import torch
from pandas import DataFrame
from rich.progress import track
from scipy.spatial.distance import cdist
from sklearn.preprocessing import OneHotEncoder
from torch import tensor
from torch.utils.data import Dataset
from dataset.data_parser import parse_meta, parse_boundary_fields, parse_internal_fields
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
        self.std = torch.tensor(self.std).to(*args, *kwargs)
        self.mean = torch.tensor(self.mean).to(*args, *kwargs)
        return self


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

    def to(self, *args, **kwargs):
        self.min = torch.tensor(self.min).to(*args, **kwargs)
        self.max = torch.tensor(self.max).to(*args, **kwargs)
        self.range = torch.tensor(self.range).to(*args, **kwargs)
        return self


def collate_fn(samples: list[FoamData]) -> FoamData:
    batch_data = torch.stack([s.data for s in samples])
    subdomains = samples[0].domain.keys()
    domain = {subd: torch.stack([s.domain[subd] for s in samples]) for subd in subdomains}
    return FoamData(batch_data, samples[0].labels, domain)


class FoamDataset(Dataset):
    def __init__(self, data_dir, n_internal, n_boundary, n_obs, rng, meta_dir=None, extra_fields=[],
                 regions_weights: dict[str, float] = None):
        self.n_boundary = n_boundary
        self.n_internal = n_internal
        self.n_obs = n_obs
        self.rng = rng
        self.regions_weights = regions_weights
        self.data_dir = data_dir

        with open(Path(data_dir) / 'data_config.json') as f:
            data_cfg = json.load(f)
            self.fields = data_cfg['Fields'] + extra_fields
            self.variable_boundaries = data_cfg['Variable boundaries']
            self.dim_labels = data_cfg['Dims']
            self.normalize_fields = data_cfg['Normalize fields']

        self.samples = sorted([d for d in Path(data_dir).iterdir() if d.is_dir()])
        self.n_dims = len(self.dim_labels)

        if self.normalize_fields is not None:
            self.meta = parse_meta(data_dir if meta_dir is None else meta_dir)
            stats = self.meta['Stats']
            self.normalizers = {}
            for field in self.normalize_fields['Standardize']:
                field_stats = stats[field]
                self.normalizers[field] = StandardScaler(np.array(field_stats['Std']),
                                                         np.array(field_stats['Mean']))
            for field in self.normalize_fields['Scale']:
                field_stats = stats[field]
                self.normalizers[field] = Normalizer(np.array(field_stats['Min']),
                                                     np.array(field_stats['Max']))

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

    def get_weights(self, boundary_names):
        weights = np.ones((len(boundary_names)))
        if self.regions_weights:
            for i, b in enumerate(boundary_names):
                if b in self.regions_weights:
                    weights[i] = self.regions_weights[b]
        return weights

    def get_stratified_sampling_n(self, boundary_names, total_sample_size):
        n_min = np.array([self.min_points[bound] for bound in boundary_names]).astype(np.int64)
        n_mean = np.array([self.meta['Points'][bound]['Mean'] for bound in boundary_names]).astype(np.int64)
        n_total = np.sum(n_mean)

        weights = self.get_weights(boundary_names)
        fractions = n_mean / n_total * weights
        # Renormalize to [0,1]
        fractions = fractions / np.sum(fractions)
        target_n = (fractions * total_sample_size).astype(np.int64)

        # Rebalance
        exceeding_samples = np.maximum(target_n - n_min, np.zeros_like(target_n))
        n_free = np.count_nonzero(exceeding_samples <= 0)
        # Take into account int truncation
        total_to_redist = np.sum(exceeding_samples) + total_sample_size - np.sum(target_n)
        sort_ids = np.argsort(n_min)
        for id in sort_ids:
            # Check if exceeded
            if exceeding_samples[id] > 0:
                continue
            added_samples = min(n_min[id], total_to_redist // n_free)
            target_n[id] += added_samples
            n_free -= 1
            total_to_redist -= added_samples
        # Replace exceeded samples
        target_n[exceeding_samples > 0] = n_min[exceeding_samples > 0]

        exceeding_samples = np.maximum(target_n - n_min, np.zeros_like(target_n))
        if np.sum(exceeding_samples) != 0:
            n_exceeding = zip(boundary_names.values[exceeding_samples > 0], exceeding_samples[exceeding_samples > 0])
            raise RuntimeError(f'Unable to satisfy sampling constraints. '
                               f'The following samples exceed the minimum:\n{list(n_exceeding)}')

        return target_n

    def sample_boundary(self, boundary_fields: DataFrame) -> DataFrame:
        """
        Samples the boundary points. Return internal_fields to avoid sampling.
        :param boundary_fields:
        :return: The samples boundary_fields
        """
        boundary_names = boundary_fields.index.unique()
        target_n_samples = self.get_stratified_sampling_n(boundary_names, self.n_boundary)

        sampled_df = []
        for i, bound in enumerate(boundary_names):
            bound_points = len(boundary_fields.loc[bound])
            samples = self.rng.choice(bound_points, replace=False, size=target_n_samples[i])
            sampled_df.append(boundary_fields.loc[bound].iloc[samples])

        return pandas.concat(sampled_df)

    def sample_internal(self, internal_fields: DataFrame) -> DataFrame:
        """
        Samples the internal points. Return internal_fields to avoid sampling.
        :param internal_fields:
        :return: The samples internal_fields
        """

        boundary_names = ['fluid', 'porous']
        target_n_samples = self.get_stratified_sampling_n(boundary_names, self.n_internal)

        # Create a temp dataframe with internal and porous labels
        temp_index = np.empty((len(internal_fields), 1), dtype='U8')
        temp_index[internal_fields['cellToRegion'].values > 0] = 'porous'
        temp_index[internal_fields['cellToRegion'].values == 0] = 'internal'
        temp_df = internal_fields.copy()
        temp_df.index = temp_index.flatten().tolist()
        boundary_names[0] = 'internal'

        sampled_df = []
        for i, bound in enumerate(boundary_names):
            bound_points = len(temp_df.loc[bound])
            samples = self.rng.choice(bound_points, replace=False, size=target_n_samples[i])
            sampled_df.append(temp_df.loc[bound].iloc[samples])

        sampled_df = pandas.concat(sampled_df)
        sampled_df.index = ['internal'] * len(sampled_df)
        return sampled_df

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
        return [label + self.dim_labels[i] for i in range(size)]

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
            if dims.iloc[0] == '':
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
        porous_cells = internal_fields['cellToRegion']
        porous_ids = porous_cells.reset_index().index[porous_cells > 0]
        domain = {'internal': np.arange(n_internal),
                  'boundary': np.arange(len(boundary_fields)) + n_internal,
                  'porous': porous_ids}

        for b in boundary_fields.index.unique():
            b_range = boundary_fields.index.get_loc(b)
            if b_range == 0:
                continue
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

    def add_sdf(self, internal_fields, boundary_fields):
        all_points = np.concatenate([internal_fields['C'].values, boundary_fields['C'].values])
        tgt_points = boundary_fields['C'].values

        if 'C' in self.normalizers:
            c_scaler = self.normalizers['C']
            all_points = c_scaler.inverse_transform(all_points)
            tgt_points = c_scaler.inverse_transform(tgt_points)

        sdf = cdist(all_points, tgt_points)
        sdf = np.min(sdf, axis=-1)
        sdf = sdf / np.max(sdf)

        internal_sign = (0.5 - internal_fields['cellToRegion'].values.flatten()) * 2
        internal_fields['sdf'] = sdf[:len(internal_fields)] * internal_sign
        # Boundary points are positive
        boundary_fields['sdf'] = sdf[len(internal_fields):]

    def add_boundary_id(self, internal_fields, boundary_fields):
        unique_bc = boundary_fields.index.unique()
        boundary_multi_index = list(itertools.product(['boundaryId'], unique_bc))

        # Internal fields have zero Id
        internal_fields[boundary_multi_index] = np.zeros((len(internal_fields), len(unique_bc)))

        ohe = OneHotEncoder(sparse_output=False)
        ohe_values = ohe.fit_transform(np.vstack(boundary_fields.index.values))
        boundary_fields[boundary_multi_index] = ohe_values

    def add_features(self, internal_fields: DataFrame, boundary_fields):
        self.add_sdf(internal_fields, boundary_fields)
        self.add_boundary_id(internal_fields, boundary_fields)

    def load_case(self, case_dir) -> FoamData:
        boundary_fields = parse_boundary_fields(case_dir, *self.fields, max_dim=self.n_dims)
        internal_fields = parse_internal_fields(case_dir, *self.fields, max_dim=self.n_dims)

        # Normalize
        if self.normalize_fields is not None:
            self.normalize(internal_fields)
            self.normalize(boundary_fields)

        # Sampling
        boundary_fields = self.sample_boundary(boundary_fields).sort_index(axis=0)
        internal_fields = self.sample_internal(internal_fields).sort_index(axis=0)

        # Add variable boundary conditions
        if self.variable_boundaries is not None:
            variable_fields = self.get_variable_boundaries(boundary_fields)
            boundary_fields = pandas.concat([boundary_fields, variable_fields], axis=1)

        self.add_features(internal_fields, boundary_fields)

        domain_data = pandas.concat([internal_fields, boundary_fields]).fillna(0)

        domain = self.get_domain(boundary_fields, internal_fields)
        labels = self.get_labels(domain_data)

        domain['obs'] = self.sample_obs(boundary_fields, internal_fields)
        domain = {d: torch.tensor(s, dtype=torch.int64) for d, s in domain.items()}

        return FoamData(tensor(domain_data.to_numpy(), dtype=torch.float), labels, domain)

    def __getitem__(self, item) -> FoamData:
        return self.data[item]
