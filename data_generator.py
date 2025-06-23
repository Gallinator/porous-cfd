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
from rich.progress import track
from bpy import ops
from welford import Welford

from data_parser import parse_boundary, parse_internal_mesh

OPENFOAM_COMMAND = "/usr/lib/openfoam/openfoam2412/etc/openfoam"


def generate_transformed_meshes(meshes_dir: str, dest_dir: str):
    pathlib.Path(dest_dir).mkdir(parents=True, exist_ok=True)

    with open(f'{meshes_dir}/transforms.json', 'r') as f:
        ops.ed.undo_push()
        ops.object.select_all(action='SELECT')
        ops.object.delete()
        for mesh, transforms in json.load(f).items():
            ops.wm.obj_import(filepath=f'{meshes_dir}/{mesh}',
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
        shutil.copyfile(m, f"{case_path}/snappyHexMesh/constant/triSurface/mesh.obj")


def generate_data(cases_dir: str):
    for case in track(glob.glob(f"{cases_dir}/*"), description="Generating geometries"):
        process = subprocess.Popen(OPENFOAM_COMMAND, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL, text=True)
        process.communicate(f"{case}/snappyHexMesh/Run")
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f'Failed to run {case}')

    for case in track(glob.glob(f"{cases_dir}/*"), description="Running cases"):
        process = subprocess.Popen(OPENFOAM_COMMAND, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL, text=True)
        process.communicate(f"{case}/simpleFoam/Run")
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f'Failed to run {case}')
        clean_dir(f"{case}/snappyHexMesh")
        os.rmdir(f"{case}/snappyHexMesh")
        shutil.move(f"{case}/simpleFoam", 'tmp')
        os.rmdir(f'{case}')
        shutil.move("tmp", f'{case}')


def generate_meta(data_dir: str):
    boundary_num_points, internal_num_points, porous_num_points = [], [], []
    running_stats = Welford()

    for case in track(glob.glob(f'{data_dir}/*'), description='Generating metadata'):
        b_data = parse_boundary(case, ['U'], ['p'])
        i_data = parse_internal_mesh(case, "U", "p")
        n_porous = np.count_nonzero(b_data[..., -1] > 0) + np.count_nonzero(i_data[..., -1] > 0)

        boundary_num_points.append(len(b_data))
        internal_num_points.append(len(i_data))
        porous_num_points.append(n_porous)

        data = np.concatenate((i_data, b_data))

        running_stats.add_all(data[..., 0:5])

    min_points_meta = {"Boundary": int(np.min(boundary_num_points)),
                       "Internal": int(np.min(internal_num_points)),
                       'Porous': int(np.min(porous_num_points))}
    features_std, features_mean = np.sqrt(running_stats.var_p).tolist(), running_stats.mean.tolist()
    std_meta = {'Points': features_std[0:2], 'U': features_std[2:4], 'p': features_std[4]}
    mean_meta = {'Points': features_mean[0:2], 'U': features_mean[2:4], 'p': features_mean[4]}

    meta_dict = {"Min points": min_points_meta, 'Mean': mean_meta, 'Std': std_meta}

    with open(f'{data_dir}/meta.json', 'w') as meta:
        meta.write(json.dumps(meta_dict, indent=4))


clean_dir('data')
clean_dir('assets/generated-meshes')

for d in os.listdir('assets/meshes'):
    generate_transformed_meshes(f'assets/meshes/{d}', f'assets/generated-meshes/{d}')
    generate_openfoam_cases(f'assets/generated-meshes/{d}', f'data/{d}')
    generate_data(f'data/{d}')
    generate_meta(f'data/{d}')
