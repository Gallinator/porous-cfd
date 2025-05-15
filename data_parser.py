import os
import numpy as np
from foamlib import FoamCase, FoamFile


def parse_boundary(case_path: str):
    last_step = int(FoamCase(case_path)[-1].time)
    boundaries_path = f"{case_path}/postProcessing"
    faces = []
    u = []
    p = []
    for s in os.listdir(boundaries_path):
        coords = FoamFile(f"{boundaries_path}/{s}/surface/{last_step}/patch_{s}/faceCentres")[None]
        u_values = FoamFile(f"{boundaries_path}/{s}/surface/{last_step}/patch_{s}/vectorField/U")[None]
        p_values = FoamFile(f"{boundaries_path}/{s}/surface/{last_step}/patch_{s}/scalarField/p")[None]
        faces.extend(coords)
        u.extend(u_values)
        p.extend(p_values)
    return np.array(faces), np.array(u), np.array(p)


def reshape_fields(fields: list[np.ndarray], size: int):
    for i, f in enumerate(fields):
        if len(f.shape) == 1:
            fields[i] = np.atleast_2d(f).T


def parse_internal_mesh(case_path: str, *fields):
    case = FoamCase(case_path)
    last_step = case[-1]
    domain_points = [last_step.cell_centers().internal_field]
    field_value = []
    for f in fields:
        parsed_field = last_step[f].internal_field
        field_value.append(parsed_field)
    reshape_fields(field_value, len(domain_points))
    return domain_points + field_value
