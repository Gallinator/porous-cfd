from pandas import DataFrame
from dataset.foam_dataset import FoamDataset


class VerticalDuctDataset(FoamDataset):
    """
    Custom dataset for the vertical duct experiment.

    Merges the inlet and top-inlet boundary id features into the inlet boundary id.
    """

    def add_features(self, internal_fields: DataFrame, boundary_fields):
        super().add_features(internal_fields, boundary_fields)

        internal_fields.drop('inlet-top', level=1, axis=1, inplace=True)

        inlet_id = boundary_fields[('boundaryId', 'inlet')].values
        inlet_top_id = boundary_fields[('boundaryId', 'inlet-top')].values
        inlet_id = inlet_id + inlet_top_id
        boundary_fields[('boundaryId', 'inlet')] = inlet_id
        boundary_fields.drop('inlet-top', level=1, axis=1, inplace=True)
