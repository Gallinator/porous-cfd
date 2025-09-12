import numpy as np
from pandas import DataFrame

from dataset.foam_dataset import FoamDataset


class FluidDataset(FoamDataset):
    def add_features(self, internal_fields: DataFrame, boundary_fields):
        # Here additional features can be added to the dataset
        # IMPORTANT: it is not possible to create new dataframe objects, the input must be modified by adding columns
        # Example adding a column of zeros:
        # internal_fields['useless'] = np.zeros((len(internal_fields)))
        # boundary_fields['useless'] = np.zeros((len(boundary_fields)))
        return NotImplemented

    def sample_obs(self, boundary_fields, internal_fields) -> np.ndarray:
        # Must return a numpy array of indices (rows) to use as observations (sensors)
        # The indices must assume that the internal and boundary data rows are concatenated with the internal at the top
        # Example of random sampling:
        # return self.rng.choice(len(internal_fields), replace=False, size=self.n_obs)
        return NotImplemented

    def sample_boundary(self, boundary_fields: DataFrame) -> DataFrame:
        # Return the data to be used as boundary points
        # If no sampling is required just return the boundary_fields
        # Example of random sampling:
        # samples = self.rng.choice(len(internal_fields), replace=False, size=self.n_internal)
        # return internal_fields.iloc[samples]
        return NotImplemented

    def sample_internal(self, internal_fields: DataFrame) -> DataFrame:
        # Return the data to be used as internal points
        # If no sampling is required just return the internal_fields
        # Example:
        # samples = self.rng.choice(len(internal_fields), replace=False, size=self.n_internal)
        # return internal_fields.iloc[samples]
        return NotImplemented
