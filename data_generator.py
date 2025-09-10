import argparse
import glob
import itertools
import json
import os
import pathlib
from random import Random
import re
import shutil
import subprocess
import sys
from argparse import ArgumentParser
import bpy
import numpy as np
from foamlib import FoamFile
from pandas import DataFrame
from rich.progress import track
from bpy import ops
from welford import Welford
from dataset.data_parser import parse_boundary_fields, parse_internal_fields, parse_elapsed_time

OPENFOAM_COMMAND = ""


class MinMaxTracker:
    def __init__(self):
        self.min, self.max = None, None

    def update(self, value: np.ndarray):
        min_val, max_val = np.min(value, axis=0), np.max(value, axis=0)
        self.min = min_val if self.min is None else np.min(np.stack([self.min, min_val]), axis=0)
        self.max = max_val if self.max is None else np.max(np.stack([self.max, max_val]), axis=0)


def import_mesh(mesh: str):
    ops.wm.obj_import(filepath=mesh, forward_axis='Y', up_axis='Z')


def get_random_in_range(l, h, rng):
    return l + rng.random() * (h - l)


def merge_trees(trees):
    ops.object.select_all(action='DESELECT')
    windbreak = trees[0]
    windbreak.select_set(True)
    for i, t in enumerate(trees[:-1]):
        modifier = windbreak.modifiers.new(name="Boolean", type='BOOLEAN')
        modifier.operation = 'UNION'
        modifier.object = trees[i + 1]
        bpy.context.view_layer.objects.active = windbreak
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    return windbreak


def create_windbreak(src_tree, n_trees, scales, rng):
    trees = []
    prev_obj = src_tree
    for n in range(n_trees):
        ops.object.select_all(action='DESELECT')
        src_tree.select_set(True)
        ops.object.duplicate(linked=False)
        obj = bpy.context.selected_objects[0]

        scale_xy = get_random_in_range(*scales['xy'], rng=rng)
        scale_z = get_random_in_range(*scales['z'], rng=rng)
        obj.scale = (scale_xy, scale_xy, scale_z)
        rot_z = get_random_in_range(0, 360, rng=rng)
        obj.rotation_euler = (*obj.rotation_euler[0:2], rot_z)
        bpy.ops.object.transform_apply(scale=False, location=False, rotation=True)
        y_size = obj.dimensions[1]
        prev_y_size = prev_obj.dimensions[1]
        if n > 0:
            obj.location[1] = prev_obj.location[1] + prev_y_size / 2 + y_size / 2 * 0.8
        trees.append(obj)
        prev_obj = obj
    return trees


def generate_transformed_meshes(meshes_dir: str, dest_dir: str, rng):
    pathlib.Path(dest_dir).mkdir(parents=True, exist_ok=True)

    with open(f'{meshes_dir}/transforms.json', 'r') as f:
        ops.ed.undo_push()
        ops.object.select_all(action='SELECT')
        ops.object.delete()
        for mesh, transforms in json.load(f).items():
            import_mesh(f'{meshes_dir}/{mesh}')
            n_trees = transforms['n_trees']
            scales = transforms['scale']
            src_obj = bpy.context.selected_objects[0]
            ops.object.select_all(action='DESELECT')

            for i in range(transforms['n_windbreaks']):

                trees = create_windbreak(src_obj, n_trees, scales, rng)

                windbreak = merge_trees(trees)

                bpy.ops.object.select_all(action='DESELECT')
                windbreak.select_set(True)

                modifier = windbreak.modifiers.new(name="Remesh", type='REMESH')
                modifier.voxel_size = 0.2
                bpy.context.view_layer.objects.active = windbreak
                bpy.ops.object.modifier_apply(modifier=modifier.name)

                bpy.context.view_layer.objects.active = windbreak
                bpy.ops.object.transform_apply()
                bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME')
                windbreak.location = [0, 0, windbreak.location[2]]

                ops.wm.obj_export(filepath=f'{dest_dir}/{i}_{mesh}',
                                  forward_axis='Y',
                                  up_axis='Z',
                                  export_materials=False,
                                  export_selected_objects=True)
                for t in trees:
                    t.select_set(True)
                # Delete copies
                ops.object.delete()
        # Delete original
        ops.object.select_all(action='SELECT')
        ops.object.delete()

        shutil.copytree(f'{meshes_dir}/houses', f'{dest_dir}/houses')


def create_case_template_dirs():
    """
    Creates the missing directories in the case template because git does not track directories
    """
    pathlib.Path(f'assets/openfoam-case-template/constant/triSurface').mkdir(parents=True, exist_ok=True)


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
    return center


def get_location_outside():
    return -0.3 + 1e-6, -0.3 + 1e-6


def write_locations_in_mesh(case_path: str, loc_in_mesh):
    snappy_dict = FoamFile(f'{case_path}/system/snappyHexMeshDict')
    snappy_dict['castellatedMeshControls']['locationInMesh'] = loc_in_mesh
    snappy_dict['castellatedMeshControls']['refinementSurfaces']['mesh']['insidePoint'] = loc_in_mesh


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


def write_coefs(fv_options_path: str, coefs: list, coef: str):
    with open(fv_options_path) as f:
        lines = f.read()
    lines = re.sub(f'{coef}\s+(.+);', f'{coef} ({coefs[0]} {coefs[1]} {coefs[2]});', lines)

    with open(fv_options_path, 'w') as f:
        f.write(lines)


def generate_openfoam_cases(meshes_dir: str, dest_dir: str, case_config_dir: str, n_proc, rng):
    pathlib.Path(dest_dir).mkdir(parents=True, exist_ok=True)

    with open(f'{case_config_dir}/config.json', 'r') as config:
        config = json.load(config)['cfd params']
        meshes = glob.glob(f"{meshes_dir}/*.obj")
        houses = glob.glob(f'{meshes_dir}/houses/*.obj')
        params = list(itertools.product(meshes, config['inlet']))
        for m, inlet_ux in params:
            mesh_name = re.match('.+_(.+obj)', m)[1]
            d = config['trees'][mesh_name]['d']
            f = config['trees'][mesh_name]['f']
            case_path = f"{dest_dir}/{pathlib.Path(m).stem}_d{d[0]}_{f[0]}_in{inlet_ux}"
            shutil.copytree('assets/openfoam-case-template', case_path)
            shutil.copyfile(m, f"{case_path}/constant/triSurface/mesh.obj")

            rand_house = houses[rng.randint(0, len(houses) - 1)]
            shutil.copyfile(rand_house, f"{case_path}/constant/triSurface/solid.obj")

            write_locations_in_mesh(f'{case_path}', get_location_inside(m))

            FoamFile(f'{case_path}/0/U')['internalField'] = [inlet_ux, 0, 0]

            fv_options = f'{case_path}/system/fvOptions'
            write_coefs(fv_options, d, 'd')
            write_coefs(fv_options, f, 'f')

            set_decompose_par(f'{case_path}', n_proc)


def raise_with_log_text(case_path, text):
    with open(f'{case_path}/log.txt') as log:
        raise RuntimeError(f'{text} {case_path}\n\n {log.read()}')


def generate_data(cases_dir: str):
    for case in track(glob.glob(f"{cases_dir}/*"), description="Running cases"):
        process = subprocess.Popen(OPENFOAM_COMMAND, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL, text=True)
        process.communicate(f"{case}/Run")
        process.wait()
        if process.returncode != 0:
            raise_with_log_text(f'{case}', 'Failed to run ')


def generate_meta(data_dir: str, *fields):
    min_points = {'internal': sys.maxsize}
    min_max_tracker = MinMaxTracker()
    stats_tracker = Welford()
    fields_columns = None
    elapse_times = []

    for case in track(glob.glob(f'{data_dir}/*/'), description='Generating metadata'):
        internal_fields = parse_internal_fields(case, *fields)
        boundary_fields = parse_boundary_fields(case, *fields)

        if fields_columns is None:
            fields_columns = internal_fields.columns

        min_points['internal'] = min(min_points['internal'], len(internal_fields))
        for bound in boundary_fields.index.unique():
            bound_size = len(boundary_fields.loc[bound])
            min_points[bound] = min(min_points[bound], bound_size) if bound in min_points.keys() else bound_size

        data = np.concatenate([internal_fields.to_numpy(), boundary_fields.to_numpy()])

        min_max_tracker.update(data)
        stats_tracker.add_all(data)
        elapse_times.append(parse_elapsed_time(case) / 1e6)

    min_df = DataFrame(np.expand_dims(min_max_tracker.min, 0), columns=fields_columns)
    max_df = DataFrame(np.expand_dims(min_max_tracker.max, 0), columns=fields_columns)
    mean_df = DataFrame(np.expand_dims(stats_tracker.mean, 0), columns=fields_columns)
    std_df = DataFrame(np.expand_dims(np.sqrt(stats_tracker.var_p), 0), columns=fields_columns)
    fields_meta = {}

    for f in fields_columns.get_level_values(0).unique():
        fields_meta[f] = {
            'Min': min_df[f].to_numpy().flatten().tolist(),
            'Max': max_df[f].to_numpy().flatten().tolist(),
            'Mean': mean_df[f].to_numpy().flatten().tolist(),
            'Std': std_df[f].to_numpy().flatten().tolist()
        }

    meta_dict = {"Min points": min_points,
                 'Stats': fields_meta}

    with open(f'{data_dir}/meta.json', 'w') as meta:
        meta.write(json.dumps(meta_dict, indent=4))


def generate_split(data_path: str, config_path: str, rng):
    if not os.path.exists(config_path):
        return
    with open(config_path) as f:
        config = json.load(f)
        if 'splits' not in config.keys():
            return
        splits = config['splits']
    cases = list(os.listdir(f"{data_path}"))
    rng.shuffle(cases)
    n = len(cases)
    start = 0
    for s in splits:
        end = start + int(splits[s] * n)
        for case in cases[start:end]:
            shutil.move(f'{data_path}/{case}', f'{pathlib.Path(data_path).parent}/{s}/{case}')
        start = end

    # Move remaining cases to first split (usually train)
    for case in os.listdir(f'{data_path}'):
        first_split = list(splits.keys())[0]
        shutil.move(f'{data_path}/{case}', f'{pathlib.Path(data_path).parent}/{first_split}/{case}')

    shutil.rmtree(pathlib.Path(data_path))


def generate_min_points(data_parent: str):
    dicts = []
    for split in glob.glob(f'{data_parent}/*'):
        with open(f'{split}/meta.json', 'r') as f:
            dicts.append(json.load(f)['Min points'])

    out = dict.fromkeys(dicts[0].keys(), sys.float_info.max)
    for d in dicts:
        out = {k: min(out[k], d[k]) for k in d.keys()}

    with open(f'{data_parent}/min_points.json', 'w') as f:
        f.write(json.dumps(out))


def build_arg_parser() -> ArgumentParser:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--openfoam-dir', type=str,
                            help='OpenFOAM installation directory')
    arg_parser.add_argument('--openfoam-procs', type=int,
                            help='the number of processors to use for OpenFoam simulations',
                            default=2)
    arg_parser.add_argument('--data-base-dir', type=str, default='data')
    return arg_parser


def clean_processor_data(data_dir):
    for case in glob.glob(f'{data_dir}/*/'):
        for proc in glob.glob(f'{case}/processor*/'):
            shutil.rmtree(proc)


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    OPENFOAM_COMMAND = f'{args.openfoam_dir}/etc/openfoam'
    data_root_dir = args.data_base_dir
    pathlib.Path(data_root_dir).mkdir(exist_ok=True, parents=True)

    create_case_template_dirs()
    clean_dir(data_root_dir)
    clean_dir('assets/generated-meshes')

    rng = Random(8421)

    for d in os.listdir('assets/meshes'):
        generate_transformed_meshes(f'assets/meshes/{d}', f'assets/generated-meshes/{d}', rng=rng)
        generate_openfoam_cases(f'assets/generated-meshes/{d}',
                                f'{data_root_dir}/{d}',
                                f'assets/meshes/{d}',
                                args.openfoam_procs, rng=rng)
        generate_split(f'{data_root_dir}/{d}', f'assets/meshes/{d}/config.json', rng=rng)

    for split in glob.glob(f'{data_root_dir}/*/'):
        generate_data(f'{data_root_dir}/{split}')
        generate_meta(f'{data_root_dir}/{split}', 'C', 'U', 'p', 'd', 'f')
        clean_processor_data(f'{data_root_dir}/{split}')
    generate_min_points(data_root_dir)
