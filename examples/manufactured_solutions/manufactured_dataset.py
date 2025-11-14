import numpy as np
from pandas import DataFrame
from dataset.foam_dataset import FoamDataset


class ManufacturedDataset(FoamDataset):
    """
    Manufactured solutions dataset.

    The velocity and pressure fields are created when loading each case.
    """

    def __init__(self,
                 data_dir: str,
                 n_internal: int,
                 n_boundary: int,
                 d: float,
                 f: float,
                 rng,
                 meta_dir=None,
                 extra_fields=[]):
        """
        :param data_dir: The base data directory. Must contain a folder for each OpenFOAM case
        :param n_internal: The number of internal points to sample.
        :param n_boundary: The number of boundary points to sample.
        :param rng: Random object used for sampling.
        :param meta_dir: Directory containing the meta.json file. Pass None to use the file inside data_dir.
        :param extra_fields: Additional fields to parse, by name.
        :param d: The Darcy coefficient.
        :param f: The Forchheimer coefficient.
        """
        self.d = d
        self.f = f
        super().__init__(data_dir, n_internal, n_boundary, 0, rng, meta_dir, extra_fields=extra_fields)

    def add_features(self, internal_fields: DataFrame, boundary_fields: DataFrame):
        """
        Adds the SDF, boundary id and forcing term features to internal_fields and boundary_fields in-place.

        The added features are the velocity, pressure and forcing terms for the continuity and momentum equations.
        """
        super().add_features(internal_fields, boundary_fields)
        self.add_manufactured_solutions(internal_fields)
        self.add_manufactured_solutions(boundary_fields)

    def add_manufactured_solutions(self, fields: DataFrame):
        points_x, points_y = fields['C', 'x'].to_numpy(), fields['C', 'y'].to_numpy()
        zones_ids = fields['cellToRegion'].to_numpy()

        # Create manufactures data
        u_x = np.sin(points_y) * np.cos(points_x)
        u_y = -np.sin(points_x) * np.cos(points_y)

        p = -1 / 4 * (np.cos(2 * points_x) + np.cos(2 * points_y))

        f_x = 2 * 0.01 * np.cos(points_x) * np.sin(points_y)
        f_y = -2 * 0.01 * np.sin(points_x) * np.cos(points_y)

        u_mag = np.sqrt(u_x ** 2 + u_y ** 2)
        f_x += (0.01 * self.d + 1 / 2 * self.f * u_mag) * u_x * zones_ids
        f_y += (0.01 * self.d + 1 / 2 * self.f * u_mag) * u_y * zones_ids

        fields['f', 'x'] = f_x.flatten()
        fields['f', 'y'] = f_y.flatten()
        fields['U', 'x'] = u_x.flatten()
        fields['U', 'y'] = u_y.flatten()
        fields['p', ''] = p.flatten()
