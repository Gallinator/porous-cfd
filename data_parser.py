import os
import Ofpp
import numpy as np

def parse_boundary(case_path: str):
    last_step = get_last_step(case_path)
    boundaries_path = f"{case_path}/postProcessing"
    coords = []
    u = []
    p = []
    for s in os.listdir(boundaries_path):
        u_coords, u_values = parse_surface(f"{boundaries_path}/{s}/surface/{last_step}/U_patch_{s}.raw", 2)
        p_coords, p_values = parse_surface(f"{boundaries_path}/{s}/surface/{last_step}/p_patch_{s}.raw", 1)
        coords += u_coords
        u += u_values
        p += p_values
    return np.array(coords), np.array(u), np.array(p)


def parse_surface(surface_path: str, field_components: int):
    with open(surface_path) as f:
        lines = f.readlines()
        coords = []
        field = []
        for row in lines[2:]:
            content = row.split(' ')
            coords.append([float(content[0]), float(content[1])])
            field.append([float(content[3 + i]) for i in range(field_components)])
    return coords, field


def get_last_step(case_path: str) -> str:
    steps = os.listdir(case_path)
    steps = [s for s in steps if s.isdigit()]
    steps.sort()
    return steps[-1]


def parse_internal_mesh(case_path: str, fields: list[str]):
    domain_mesh = Ofpp.FoamMesh(case_path)
    domain_mesh.read_cell_centres(case_path)
    domain_points = (domain_mesh.cell_centres())
    last_step = get_last_step(case_path)
    field_value = ()
    for f in fields:
        parsed_field = Ofpp.parse_internal_field(f"{case_path}/{last_step}/{f}")
        field_value += parsed_field
    return domain_points + field_value

