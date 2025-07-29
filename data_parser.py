import json
import os
import re
from pathlib import Path

import numpy as np
from foamlib import FoamCase, FoamFile


def parse_post_process_fields(path: str):
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


def parse_boundary(case_path: str, vectors: list[str], scalars: list[str]) -> dict[str, np.ndarray]:
    last_step = int(FoamCase(case_path)[-1].time)
    boundaries_path = f"{case_path}/postProcessing"
    b_dict = {}

    for b in sorted(os.listdir(boundaries_path)):
        scalar_values, vector_values = [], []
        intermediate_dir = list(os.listdir(f"{boundaries_path}/{b}/surface/{last_step}"))[0]
        coords = FoamFile(f"{boundaries_path}/{b}/surface/{last_step}/{intermediate_dir}/faceCentres")[None]

        for s in scalars:
            values = parse_post_process_fields(
                f"{boundaries_path}/{b}/surface/{last_step}/{intermediate_dir}/scalarField/{s}")
            values = make_column(values)
            scalar_values.append(values)

        for v in vectors:
            values = parse_post_process_fields(
                f"{boundaries_path}/{b}/surface/{last_step}/{intermediate_dir}/vectorField/{v}")
            vector_values.append(values)

        b_dict[b] = np.concatenate([coords, *vector_values, *scalar_values, np.zeros((len(coords), 7))], axis=-1)

    return b_dict


def make_column(field) -> np.array:
    f = np.array(field)
    if len(f.shape) == 1:
        return np.vstack(f)
    return f


def parse_coef(case_dir: str, coef: str):
    fv_options = FoamFile(f'{case_dir}/system/fvOptions')
    return fv_options['porousFilter']['explicitPorositySourceCoeffs'][coef][0:2]


def parse_internal_mesh(case_path: str, *fields) -> np.ndarray:
    case = FoamCase(case_path)
    last_step = case[-1]
    domain_points = last_step.cell_centers().internal_field
    fields_values = []
    for f in fields:
        parsed_field = last_step[f].internal_field
        parsed_field = make_column(parsed_field)
        fields_values.append(parsed_field)

    porous_points = case[0]['cellToRegion'].internal_field.reshape((-1, 1))

    d = parse_coef(case_path, 'd')
    f = parse_coef(case_path, 'f')

    return np.concatenate([domain_points, *fields_values, porous_points, d * porous_points, f * porous_points], axis=1)


def parse_meta(data_dir: str) -> dict:
    with open(Path(data_dir, 'meta.json'), 'r') as f:
        return json.load(f)


def parse_elapsed_time(case_dir: str) -> int:
    with open(Path(case_dir, 'timing.txt'), 'r') as f:
        return int(f.readline())
