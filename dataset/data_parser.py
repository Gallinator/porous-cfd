import glob
import itertools
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas
from foamlib import FoamCase, FoamFile
from pandas import DataFrame, MultiIndex


def parse_post_process_field(path: str):
    """This is a temporary workaround as foamlib cannot read post processed fields"""

    def parse_content(content: str):
        if content[0] == '(':
            return [float(v) for v in content[content.find('(') + 1:content.rfind(')')].split()]
        else:
            return float(content)

    with open(path, 'r') as f:
        lines = f.readlines()
        if (m := re.match('(\d+){(.+)}', lines[0])) is not None:
            n = m.groups()[0]
            v = parse_content(m.groups()[1])
            return [v] * int(n)

        data = []
        for l in lines[3:-1]:
            data.append(parse_content(l))
    return data


def parse_boundary_patch(patch_dir, *fields, max_dim=3) -> DataFrame:
    """
    See parse internal_fields
    :param patch_dir:
    :param fields:
    :param max_dim:
    :return:
    """
    scalar_fields = {Path(p).name: p for p in list(glob.glob(f'{patch_dir}/scalarField/*'))}
    vector_fields = {Path(p).name: p for p in list(glob.glob(f'{patch_dir}/vectorField/*'))}
    avail_fields = scalar_fields | vector_fields
    fields_df = DataFrame()

    face_centres = FoamFile(f'{patch_dir}/faceCentres')[None]
    if 'C' in fields:
        fields_df = add_multidim_field(fields_df, 'C', face_centres, max_dim)

    for f in list(set(fields) - {'d', 'f', 'cellToRegion', 'C'}):
        parsed_values = parse_post_process_field(f"{avail_fields[f]}")
        parsed_values = make_column(parsed_values)
        dim = parsed_values.shape[-1]
        if dim > 1:
            fields_df = add_multidim_field(fields_df, f, parsed_values, max_dim)
        else:
            fields_df[f, ''] = parsed_values.flatten()

    if 'cellToRegion' in fields:
        fields_df['cellToRegion', ''] = np.zeros(len(face_centres))
    for coef in set(fields) & {'d', 'f'}:
        fields_df = add_multidim_field(fields_df, coef, np.zeros(face_centres.shape), max_dim)

    fields_df.columns = MultiIndex.from_tuples(fields_df.columns)
    return fields_df.reindex(columns=fields, level=0)


def parse_boundary_fields(case_path: str, *fields, max_dim=3) -> DataFrame:
    """
    See parse_case_fields
    :param case_path:
    :param fields:
    :param max_dim:
    :return:
    """
    last_step = int(FoamCase(case_path)[-1].time)
    postprocess_path = f"{case_path}/postProcessing"
    boundary_df = []

    for boundary_name in sorted(os.listdir(postprocess_path)):
        last_step_dir = f'{postprocess_path}/{boundary_name}/surface/{last_step}'
        patch_dir = list(os.listdir(f"{postprocess_path}/{boundary_name}/surface/{last_step}"))[0]
        parsed_fields = parse_boundary_patch(f'{last_step_dir}/{patch_dir}', *fields, max_dim=max_dim)
        parsed_fields.index = [boundary_name] * len(parsed_fields)
        boundary_df.append(parsed_fields)

    return pandas.concat(boundary_df)


def make_column(field) -> np.array:
    """
    Reshapes the input to a column array, if necessary.
    """
    f = np.array(field)
    if len(f.shape) == 1:
        return np.vstack(f)
    return f


def parse_coef(case_dir: str, coef: str):
    fv_options = FoamFile(f'{case_dir}/system/fvOptions')
    return fv_options['porousFilter']['explicitPorositySourceCoeffs'][coef]


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
    Parses the internal fields of an OpenFOAM case
    :param case_dir: The case path
    :param fields: The fields to parse
    :param max_dim:Maximum dimension of each field. If a field is larger than this value, the last component is dropped
    :return: a DataFrame with index set as 'internal'
    """
    case = FoamCase(case_dir)
    last_step = case[-1]
    fields_df = DataFrame()

    if 'C' in fields:
        fields_df = add_multidim_field(fields_df, 'C', last_step.cell_centers().internal_field, max_dim)

    cell_to_region = make_column(case[0]['cellToRegion'].internal_field)
    if 'cellToRegion' in fields:
        fields_df['cellToRegion', ''] = cell_to_region.flatten()

    for f in list(set(fields) - {'d', 'f', 'cellToRegion', 'C'}):
        parsed_field = last_step[f].internal_field
        parsed_field = make_column(parsed_field)
        dim = parsed_field.shape[-1]
        if dim > 1:
            fields_df = add_multidim_field(fields_df, f, parsed_field, max_dim)
        else:
            fields_df[f, ''] = parsed_field.flatten()

    for coef in set(fields) & {'d', 'f'}:
        fields_df = add_multidim_field(fields_df, coef, cell_to_region * parse_coef(case_dir, coef), max_dim)

    fields_df.columns = MultiIndex.from_tuples(fields_df.columns)
    fields_df.index = ['internal'] * len(fields_df)
    return fields_df.reindex(columns=fields, level=0)


def parse_case_fields(case_dir, *fields, max_dim=3) -> DataFrame:
    """
    Parses the fields of an OpenFoam case.
    :param case_dir: The case path
    :param fields: The fields to parse
    :param max_dim: Maximum dimension of each field. If a field is larger than this value, the last component is dropped
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
