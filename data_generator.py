import glob
import json
import math
import os
import pathlib
import shutil
import subprocess
import mathutils
import bpy
import numpy as np
from tqdm import tqdm
from bpy import ops
from data_parser import parse_boundary, parse_internal_mesh

OPENFOAM_COMMAND = "/usr/lib/openfoam/openfoam2412/etc/openfoam"


def generate_transformed_meshes(meshes_path: str, dest_dir: str):
    pathlib.Path(dest_dir).mkdir(parents=True, exist_ok=True)

    with open(f'{meshes_path}/transforms.json', 'r') as f:
        ops.ed.undo_push()
        ops.object.select_all(action='SELECT')
        ops.object.delete()
        for mesh, transforms in json.load(f).items():
            ops.wm.obj_import(filepath=f'{meshes_path}/{mesh}',
                              forward_axis='Y',
                              up_axis='Z')
            for t in transforms:
                for r in t["rotation"]:
                    ops.object.select_all(action='SELECT')
                    ops.object.duplicate(linked=False)
                    obj = bpy.context.selected_objects[0]

                    scale = t["scale"]
                    obj.scale = mathutils.Vector((scale[0], scale[1], 1.0))

                    obj.rotation_euler = mathutils.Euler((0.0, 0.0, math.radians(-r)))

                    ops.wm.obj_export(filepath=f'{dest_dir}/s{scale[0]}-{scale[1]}_r{r}_{mesh}',
                                      forward_axis='Y',
                                      up_axis='Z',
                                      export_materials=False,
                                      export_selected_objects=True)
                    # Delete copy
                    ops.object.delete()
            # Delete original
            ops.object.select_all(action='SELECT')
            ops.object.delete()


def clean_dir(directory: str):
    for root, dirs, files in os.walk(directory):
        for f in files:
            os.remove(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))


def generate_openfoam_cases(meshes_dir: str, dest_dir: str):
    pathlib.Path(dest_dir).mkdir(parents=True, exist_ok=True)

    meshes = glob.glob(f"{meshes_dir}/*.obj")
    for m in meshes:
        case_path = f"{dest_dir}/{pathlib.Path(m).stem}"
        shutil.copytree('assets/openfoam-case-template', case_path)
        shutil.copyfile(m, f"{case_path}/constant/triSurface/mesh.obj")


def generate_data(cases_path: str):
    for case in tqdm(glob.glob(f"{cases_path}/*"), desc="Running cases"):
        process = subprocess.Popen(OPENFOAM_COMMAND, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL, text=True)
        process.communicate(f"{case}/Run")
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f'Failed to run {case}')


def generate_meta():
    boundary_num_points, internal_num_points = [], []
    for case in tqdm(glob.glob('data/*'), desc='Generating metadata'):
        b_n, i_n = parse_case_num_points(case)
        boundary_num_points.append(b_n)
        internal_num_points.append(i_n)

    boundary_meta = {"Min points": int(np.min(boundary_num_points))}
    internal_meta = {"Min points": int(np.min(internal_num_points))}
    meta_dict = {"Internal": internal_meta, "Boundary": boundary_meta}
    with open('data/meta.json', 'w') as meta:
        meta.write(json.dumps(meta_dict, indent=4))


def parse_case_num_points(case_path: str):
    boundary_c, boundary_u, boundary_p = parse_boundary(case_path)
    internal_c, internal_u, internal_p = parse_internal_mesh(case_path, "U", "p")

    boundary_n = len(boundary_c)
    internal_n = len(internal_c)
    return boundary_n, internal_n


clean_dir('data')
clean_dir('assets/generated-meshes')
generate_transformed_meshes()
generate_openfoam_cases("assets/openfoam-case-template")
generate_data()
generate_meta()
generate_transformed_meshes('assets/meshes/train', 'assets/generated-meshes/train')
generate_openfoam_cases('assets/generated-meshes/train', 'data/train')
