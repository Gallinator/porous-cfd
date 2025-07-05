import json
import os
from pathlib import Path

import numpy as np
from foamlib import FoamCase, FoamFile


def parse_scalar_field(path: str):
    """This is a temporary workaround as foamlib cannot read boundary scalar fields"""
    centers = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for l in lines[3:-1]:
            centers.append(float(l))
    return centers


def parse_boundary(case_path: str, vectors: list[str], scalars: list[str]) -> np.ndarray:
    last_step = int(FoamCase(case_path)[-1].time)
    boundaries_path = f"{case_path}/postProcessing"
    faces = []
    scalar_values, vector_values = {s: [] for s in scalars}, {v: [] for v in vectors}
    for b in os.listdir(boundaries_path):
        intermediate_dir = list(os.listdir(f"{boundaries_path}/{b}/surface/{last_step}"))[0]
        coords = FoamFile(f"{boundaries_path}/{b}/surface/{last_step}/{intermediate_dir}/faceCentres")[None]
        coords = make_at_most_2d(coords)
        faces.extend(coords)

        for s in scalars:
            if s == 'div(phi)':
                values = parse_scalar_field(
                    f"{boundaries_path}/{b}/surface/{last_step}/{intermediate_dir}/scalarField/{s}")
            else:
                values = FoamFile(f"{boundaries_path}/{b}/surface/{last_step}/{intermediate_dir}/scalarField/{s}")[None]
            values = make_column(values)
            scalar_values[s].extend(values)
            np.array(scalar_values[s])
        for v in vectors:
            values = FoamFile(f"{boundaries_path}/{b}/surface/{last_step}/{intermediate_dir}/vectorField/{v}")[None]
            values = make_at_most_2d(values)
            vector_values[v].extend(values)
    vector_values = [np.array(vector_values[v]) for v in vectors]
    scalar_values = [np.array(scalar_values[s]) for s in scalars]
    return np.concatenate([np.array(faces)] + vector_values + scalar_values + [np.zeros_like(scalar_values[0])], axis=1)


def make_at_most_2d(field) -> np.array:
    f = np.array(field)
    if len(f.shape) > 1 and f.shape[-1] > 2:
        return f[:, :2]
    return f


def make_column(field) -> np.array:
    f = np.array(field)
    if len(f.shape) == 1:
        return np.vstack(f)
    return f


def parse_d(case_dir: str):
    fv_options = FoamFile(f'{case_dir}/system/fvOptions')
    return fv_options['porousFilter']['explicitPorositySourceCoeffs']['d']


def parse_internal_mesh(case_path: str, *fields) -> np.ndarray:
    case = FoamCase(case_path)
    last_step = case[-1]
    domain_points = last_step.cell_centers().internal_field
    domain_points = make_at_most_2d(domain_points)
    fields_values = []
    for f in fields:
        parsed_field = last_step[f].internal_field
        parsed_field = make_column(parsed_field)
        parsed_field = make_at_most_2d(parsed_field)
        fields_values.append(parsed_field)

    porous_points = case[0]['cellToRegion'].internal_field.reshape((-1, 1))

    d = parse_d(case_path)
    d = np.array([d]).repeat(len(porous_points), axis=0)
    d = make_at_most_2d(d)

    return np.concatenate([domain_points, *fields_values, porous_points, d * porous_points], axis=1)


def parse_meta(data_dir: str) -> dict:
    with open(Path(data_dir, 'meta.json'), 'r') as f:
        return json.load(f)


def parse_elapsed_time(case_dir: str) -> int:
    with open(Path(case_dir, 'timing.txt'), 'r') as f:
        return int(f.readline())
