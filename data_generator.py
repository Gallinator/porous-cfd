import argparse
import glob
import itertools
import json
import math
import os
import pathlib
from random import Random
import re
import shutil
import subprocess
from argparse import ArgumentParser
import mathutils
import bpy
import numpy as np
from foamlib import FoamFile
from rich.progress import track
from bpy import ops
from welford import Welford

from data_parser import parse_boundary, parse_internal_mesh, parse_elapsed_time

OPENFOAM_COMMAND = ""


def import_mesh(mesh: str):
    ops.wm.obj_import(filepath=mesh, forward_axis='Y', up_axis='Z')


def parse_rotations(rotations_dict: dict):
    if not rotations_dict:
        return [0]
    start, stop, n = rotations_dict[0], rotations_dict[1], rotations_dict[2]
    return np.linspace(start, stop, n)


def parse_scale(scale_dict: dict) -> list:
    if [] in scale_dict.values():
        return [(1, 1)]

    def parse_values(data: list):
        return np.linspace(data[0], data[1], data[2])

    if 'xy' in scale_dict:
        scales = parse_values(scale_dict['xy'])
        return list(zip(scales, scales))

    scales_x = parse_values(scale_dict['x'])
    scales_y = parse_values(scale_dict['y'])
    return list(itertools.product(scales_x, scales_y))


def generate_transformed_meshes(meshes_dir: str, dest_dir: str):
    pathlib.Path(dest_dir).mkdir(parents=True, exist_ok=True)

    with open(f'{meshes_dir}/transforms.json', 'r') as f:
        ops.ed.undo_push()
        ops.object.select_all(action='SELECT')
        ops.object.delete()
        for mesh, transforms in json.load(f).items():
            import_mesh(f'{meshes_dir}/{mesh}')
            rotations = parse_rotations(transforms['rotation'])
            scales = parse_scale(transforms['scale'])
            for r, s in itertools.product(rotations, scales):
                ops.object.select_all(action='SELECT')
                ops.object.duplicate(linked=False)
                obj = bpy.context.selected_objects[0]
                obj.scale = mathutils.Vector((s[0], s[1], 1.0))

                obj.rotation_euler = mathutils.Euler((0.0, 0.0, math.radians(-r)))

                ops.wm.obj_export(filepath=f'{dest_dir}/s{s[0]}-{s[1]}_r{r}_{mesh}',
                                  forward_axis='Y',
                                  up_axis='Z',
                                  export_materials=False,
                                  export_selected_objects=True)
                # Delete copy
                ops.object.delete()
            # Delete original
            ops.object.select_all(action='SELECT')
            ops.object.delete()


def create_case_template_dirs():
    """
    Creates the missing directories in the case template because git does not track directories
    """
    pathlib.Path(f'assets/openfoam-case-template/snappyHexMesh/0').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'assets/openfoam-case-template/snappyHexMesh/constant/triSurface').mkdir(parents=True, exist_ok=True)


def clean_dir(directory: str):
    for root, dirs, files in os.walk(directory):
        for f in files:
            os.remove(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))


def get_location_inside(mesh: str):
    ops.object.select_all(action='SELECT')
    ops.object.delete()
    import_mesh(mesh)
    ops.object.select_all(action='SELECT')
    obj = bpy.context.object
    verts = [obj.matrix_world @ v.co for v in obj.data.vertices]
    verts = np.array(verts)
    center = np.sum(verts, axis=0) / len(verts)
    ops.object.delete()
    return center[0:2]


def get_location_outside():
    return -0.3 + 1e-6, -0.3 + 1e-6


def write_locations_in_mesh(case_path: str, loc_in_mesh):
    snappy_dict = FoamFile(f'{case_path}/system/snappyHexMeshDict')
    locations_in_mesh = snappy_dict['castellatedMeshControls']['locationInMesh']
    locations_in_mesh[0:2] = loc_in_mesh
    snappy_dict['castellatedMeshControls']['locationInMesh'] = locations_in_mesh
    snappy_dict['castellatedMeshControls']['refinementSurfaces']['mesh']['insidePoint'] = locations_in_mesh


def set_par_dict_coeffs(dict_path: str, n_proc: int):
    i, prev = 1, n_proc
    while True:
        proc_x = 2 ** i
        proc_y = n_proc / proc_x
        if proc_y % 2 != 0 or proc_y <= proc_x:
            proc_y = int(proc_y)
            break
        i += 1
    proc_x = max(proc_x, proc_y)
    proc_y = min(proc_x, proc_y)

    with open(dict_path) as f:
        lines = f.read()
        lines = re.sub('numberOfSubdomains\s+\d+;', f'numberOfSubdomains {n_proc};', lines)
        lines = re.sub('n\s+\(.+\)', f'n ({proc_x} {proc_y} 1);', lines)

    with open(dict_path, 'w') as f:
        f.write(lines)


def set_run_n_proc(run_path: str, n_proc: int):
    with open(run_path, 'r') as f:
        data = f.read()
        data = re.sub('\$n_proc', str(n_proc), data, re.MULTILINE)
    with open(run_path, 'w') as f:
        f.write(data)


def set_decompose_par(case_path: str, n_proc: int):
    if n_proc % 2 != 0:
        raise ValueError('n_proc must be an even number!')
    dict_path = f'{case_path}/system/decomposeParDict'
    set_par_dict_coeffs(dict_path, n_proc)
    set_par_dict_coeffs(dict_path, n_proc)
    set_run_n_proc(f'{case_path}/Run', n_proc)


def generate_openfoam_cases(meshes_dir: str, dest_dir: str, n_proc):
    pathlib.Path(dest_dir).mkdir(parents=True, exist_ok=True)

    meshes = glob.glob(f"{meshes_dir}/*.obj")
    for m in meshes:
        case_path = f"{dest_dir}/{pathlib.Path(m).stem}"
        shutil.copytree('assets/openfoam-case-template', case_path)
        shutil.copyfile(m, f"{case_path}/snappyHexMesh/constant/triSurface/mesh.obj")
        write_locations_in_mesh(f'{case_path}/snappyHexMesh', get_location_inside(m))

        set_decompose_par(f'{case_path}/snappyHexMesh', n_proc)
        set_decompose_par(f'{case_path}/simpleFoam', n_proc)


def raise_with_log_text(case_path, text):
    with open(f'{case_path}/log.txt') as log:
        raise RuntimeError(f'{text} {case_path}\n\n {log.read()}')


def generate_data(cases_dir: str):
    for case in track(glob.glob(f"{cases_dir}/*"), description="Generating geometries"):
        process = subprocess.Popen(OPENFOAM_COMMAND, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL, text=True)
        process.communicate(f"{case}/snappyHexMesh/Run")
        process.wait()
        if process.returncode != 0:
            raise_with_log_text(f'{case}/snappyHexMesh', 'Failed to generate mesh for case ')

    for case in track(glob.glob(f"{cases_dir}/*"), description="Running cases"):
        process = subprocess.Popen(OPENFOAM_COMMAND, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL, text=True)
        process.communicate(f"{case}/simpleFoam/Run")
        process.wait()
        if process.returncode != 0:
            raise_with_log_text(f'{case}/simpleFoam', 'Failed to run ')

        clean_dir(f"{case}/snappyHexMesh")
        os.rmdir(f"{case}/snappyHexMesh")
        shutil.move(f"{case}/simpleFoam", 'tmp')
        os.rmdir(f'{case}')
        shutil.move("tmp", f'{case}')


def generate_meta(data_dir: str):
    boundary_num_points, internal_num_points, porous_num_points = [], [], []
    running_stats = Welford()
    elapse_times = []

    for case in track(glob.glob(f'{data_dir}/*'), description='Generating metadata'):
        b_data = parse_boundary(f'{case}', ['U'], ['p'])
        i_data = parse_internal_mesh(f'{case}', "U", "p")
        n_porous = np.count_nonzero(b_data[..., -1] > 0) + np.count_nonzero(i_data[..., -1] > 0)

        boundary_num_points.append(len(b_data))
        internal_num_points.append(len(i_data))
        porous_num_points.append(n_porous)

        data = np.concatenate((i_data, b_data))

        running_stats.add_all(data[..., 0:5])

        elapse_times.append(parse_elapsed_time(case) / 1e6)

    min_points_meta = {"Boundary": int(np.min(boundary_num_points)),
                       "Internal": int(np.min(internal_num_points)),
                       'Porous': int(np.min(porous_num_points))}
    features_std, features_mean = np.sqrt(running_stats.var_p).tolist(), running_stats.mean.tolist()
    std_meta = {'Points': features_std[0:2], 'U': features_std[2:4], 'p': features_std[4]}
    mean_meta = {'Points': features_mean[0:2], 'U': features_mean[2:4], 'p': features_mean[4]}
    timing_meta = {'Total': sum(elapse_times), 'Average': np.mean(elapse_times)}

    meta_dict = {"Min points": min_points_meta, 'Mean': mean_meta, 'Std': std_meta, 'Timing': timing_meta}

    with open(f'{data_dir}/meta.json', 'w') as meta:
        meta.write(json.dumps(meta_dict, indent=4))


def generate_split(data_path: str, config_path: str, rng=Random()):
    if not os.path.exists(config_path):
        return
    with open(config_path) as f:
        splits = json.load(f)['splits']
    cases = list(os.listdir(f"{data_path}"))
    rng.shuffle(cases)
    n = len(cases)
    start = 0
    for s in splits:
        end = start + int(splits[s] * n)
        for case in cases[start:end]:
            shutil.move(f'{data_path}/{case}', f'{pathlib.Path(data_path).parent.parent}/{s}/raw/{case}')
        start = end

    # Move remaining cases to first split (usually train)
    for case in os.listdir(f'{data_path}'):
        first_split = list(splits.keys())[0]
        shutil.move(f'{data_path}/{case}', f'{pathlib.Path(data_path).parent.parent}/{first_split}/raw/{case}')


def build_arg_parser() -> ArgumentParser:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--openfoam-dir', type=str,
                            help='OpenFOAM installation directory')
    arg_parser.add_argument('--openfoam-procs', type=int,
                            help='the number of processors to use for OpenFoam simulations',
                            default=2)
    return arg_parser


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    OPENFOAM_COMMAND = f'{args.openfoam_dir}/etc/openfoam'

    create_case_template_dirs()
    clean_dir('data')
    clean_dir('assets/generated-meshes')

    rng = Random(8421)

    for d in os.listdir('assets/meshes'):
        generate_transformed_meshes(f'assets/meshes/{d}', f'assets/generated-meshes/{d}')
        generate_openfoam_cases(f'assets/generated-meshes/{d}',
                                f'data/{d}/raw',
                                args.openfoam_procs)
        generate_split(f'data/{d}/raw', f'assets/meshes/{d}/config.json', rng=rng)

    for d in os.listdir('data'):
        generate_data(f'data/{d}/raw')
        generate_meta(f'data/{d}/raw')
