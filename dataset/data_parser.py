import itertools
import json
from pathlib import Path
import pandas
from pandas import DataFrame

def parse_boundary_fields(case_path: str, *fields, max_dim=3) -> DataFrame:
    """
    See parse_case_fields
    :param case_path:
    :param fields:
    :param max_dim:
    :return:
    """
    # Must return a DataFrame. The labels must be multi-level, with the top level having the variable name and the lower level having the dimension name.
    # If there is only one dimension (pressure) just use an empty string ''.
    # Each sample must have an index (column) with its name. This will be used later on.
    return NotImplemented


def add_multidim_field(fields_df: DataFrame, field_name, field_values, max_dim) -> DataFrame:
    """
    Adds a multidimensional field to the DataFrame, using multi level columns
    :param fields_df: the source DataFrame
    :param field_name: the name of the field to add
    :param field_values: the values of the field to add. Must be a 2D array
    :param max_dim: maximum dimension of the field. The exceeding dimensions will be dropped.
    :return: the source dataframe with the field added
    """
    dim_labels = ['x', 'y', 'z']
    cols = list(itertools.product([field_name], dim_labels[:max_dim]))
    cat_df = DataFrame(field_values[..., :max_dim], columns=cols)
    return pandas.concat([fields_df, cat_df], axis=1)


def parse_internal_fields(case_dir: str, *fields, max_dim=3) -> DataFrame:
    """
    Parses the internal fields
    :param case_dir: The case path
    :param fields: The fields to parse
    :param max_dim:Maximum dimension of each field. Used only on OpenFoam cases
    :return: a DataFrame with index set as 'internal'
    """
    # Must return a DataFrame. The labels must be multi-level, with the top level having the variable name and the lower level having the dimension name.
    # If there is only one dimension (pressure) just use an empty string ''.
    # Each sample must have an index (column) with the name internal. This will be used later on.
    return NotImplemented


def parse_case_fields(case_dir, *fields, max_dim=3) -> DataFrame:
    """
    Parses the fields of a case.
    :param case_dir: The case path
    :param fields: The fields to parse
    :param max_dim: Maximum dimension of each field. Used only on OpenFoam cases
    :return: a DataFrame indexed according to each subdomain
    """
    internal_fields = parse_internal_fields(case_dir, *fields, max_dim=max_dim)
    boundary_fields = parse_boundary_fields(case_dir, *fields, max_dim=max_dim)
    return pandas.concat([internal_fields, boundary_fields])


def parse_meta(data_dir: str) -> dict:
    with open(Path(data_dir, 'meta.json'), 'r') as f:
        return json.load(f)


def parse_elapsed_time(case_dir: str) -> int:
    with open(Path(case_dir, 'timing.txt'), 'r') as f:
        return int(f.readline())
