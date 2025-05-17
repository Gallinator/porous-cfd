import json
import os
from pathlib import Path

import numpy as np
from foamlib import FoamCase, FoamFile


def parse_boundary(case_path: str):
    last_step = int(FoamCase(case_path)[-1].time)
    boundaries_path = f"{case_path}/postProcessing"
    faces = []
    u = []
    p = []

    for s in os.listdir(boundaries_path):
        coords = parse_face_centers(f"{boundaries_path}/{s}/surface/{last_step}/patch_{s}/faceCentres")
        coords = make_at_most_2d(coords)
        u_values = FoamFile(f"{boundaries_path}/{s}/surface/{last_step}/patch_{s}/vectorField/U")[None]
        u_values = make_at_most_2d(u_values)
        p_values = FoamFile(f"{boundaries_path}/{s}/surface/{last_step}/patch_{s}/scalarField/p")[None]
        p_values = make_column(p_values)
        faces.extend(coords)
        u.extend(u_values)
        p.extend(p_values)

    return np.array(faces), np.array(u), np.array(p)


def parse_face_centers(path: str):
    """This is a temporary workaround as foamlib cannot read all cell centers"""
    centers = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for l in lines[3:-1]:
            content = l[1:-2]
            content = content.split(' ')
            centers.append(list(map(float, content)))
    return centers


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


def parse_internal_mesh(case_path: str, *fields):
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

    return [domain_points] + fields_values


def parse_meta(data_dir: str) -> dict:
    with open(Path(data_dir, 'meta.json'), 'r') as f:
        return json.load(f)
