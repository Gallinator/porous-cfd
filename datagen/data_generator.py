import argparse
import glob
import json
import os
import re
import shutil
import sys
from abc import abstractmethod
from argparse import ArgumentParser
from pathlib import Path
from random import Random
import bpy
import matplotlib
from bpy import ops
import numpy as np
from foamlib import FoamFile
from pandas import DataFrame
from rich.progress import track
from welford import Welford

from dataset.data_parser import parse_internal_fields, parse_boundary_fields, parse_elapsed_time
from visualization.common import plot_dataset_dist, plot_u_direction_change


def build_arg_parser() -> ArgumentParser:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--openfoam-dir', type=str,
                            help='OpenFOAM installation directory')
    arg_parser.add_argument('--openfoam-procs', type=int,
                            help='the number of processors to use for OpenFoam simulations',
                            default=2)
    arg_parser.add_argument('--data-root-dir', type=str, default='data')
    arg_parser.add_argument('--meta-only', action="store_true",
                            help='regenerate the meta files', default=False)
    return arg_parser


class MinMaxTracker:
    def __init__(self):
        self.min, self.max = None, None

    def update(self, value: np.ndarray):
        min_val, max_val = np.min(value, axis=0), np.max(value, axis=0)
        self.min = min_val if self.min is None else np.min(np.stack([self.min, min_val]), axis=0)
        self.max = max_val if self.max is None else np.max(np.stack([self.max, max_val]), axis=0)


class DataGeneratorBase:
    def __init__(self, src_dir, openfoam_bin, n_procs: int, keep_p=0.5, meta_only=False):
        self.openfoam_bin = openfoam_bin
        self.n_procs = n_procs
        self.src_dir = Path(src_dir)
        self.meshes_dir = self.src_dir / 'meshes'
        self.case_template_dir = self.src_dir / 'openfoam-case-template'
        self.drop_p = keep_p
        self.meta_only = meta_only

        self.data_config_path = self.src_dir / 'data_config.json'
        with open(self.data_config_path) as f:
            data_config = json.load(f)
            self.fields = data_config['Fields']
            self.dims = data_config['Dims']

        self.meshes_sets_paths = [Path(p) for p in glob.glob(str(self.src_dir / 'meshes/*/'))]
        self.generated_meshes_dir = self.src_dir / 'generated_meshes'

    def clean_dir(self, directory: str | Path):
        for root, dirs, files in os.walk(directory):
            for f in files:
                os.remove(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))

    def write_locations_in_mesh(self, case_path: str, loc_in_mesh):
        snappy_dict = FoamFile(f'{case_path}/system/snappyHexMeshDict')
        snappy_dict['castellatedMeshControls']['locationInMesh'] = loc_in_mesh
        snappy_dict['castellatedMeshControls']['refinementSurfaces']['mesh']['insidePoint'] = loc_in_mesh

    def set_par_dict_coeffs(self, dict_path: str):
        i, prev = 1, self.n_procs
        while True:
            proc_x = 2 ** i
            proc_y = self.n_procs / proc_x
            if proc_y % 2 != 0 or proc_y <= proc_x:
                proc_y = int(proc_y)
                break
            i += 1
        proc_x = max(proc_x, proc_y)
        proc_y = min(proc_x, proc_y)

        with open(dict_path) as f:
            lines = f.read()
            lines = re.sub('numberOfSubdomains\s+\d+;', f'numberOfSubdomains {self.n_procs};', lines)
            lines = re.sub('n\s+\(.+\)', f'n ({proc_x} {proc_y} 1);', lines)

        with open(dict_path, 'w') as f:
            f.write(lines)

    def set_run_n_proc(self, run_path: str):
        with open(run_path, 'r') as f:
            data = f.read()
            data = re.sub('\$n_proc', str(self.n_procs), data, re.MULTILINE)
        with open(run_path, 'w') as f:
            f.write(data)

    def set_decompose_par(self, case_path: str):
        if self.n_procs % 2 != 0:
            raise ValueError('n_proc must be an even number!')
        dict_path = f'{case_path}/system/decomposeParDict'
        self.set_par_dict_coeffs(dict_path)
        self.set_par_dict_coeffs(dict_path)
        self.set_run_n_proc(f'{case_path}/Run')

    def write_coefs(self, fv_options_path: str, coefs: list, coef: str):
        with open(fv_options_path) as f:
            lines = f.read()
        lines = re.sub(f'{coef}\s+(.+);', f'{coef} ({coefs[0]} {coefs[1]} {coefs[2]});', lines)

        with open(fv_options_path, 'w') as f:
            f.write(lines)

    @abstractmethod
    def create_case_template_dirs(self):
        """
        Creates the missing directories in the case template because git does not track directories
        """
        pass

    @abstractmethod
    def generate_transformed_meshes(self, meshes_dir, dest_dir, rng):
        pass

    @abstractmethod
    def generate_openfoam_cases(self, meshes_dir: Path, dest_dir: Path, case_config_dir, rng):
        pass

    def generate_split(self, data_path: Path, config_dir: Path, rng):
        config_path = config_dir / 'config.json'
        if not os.path.exists(config_path):
            return
        with open(config_path) as f:
            config = json.load(f)
            if 'splits' not in config.keys():
                return
            splits = dict(sorted(config['splits'].items()))
        cases = sorted(list(os.listdir(f"{data_path}")))
        rng.shuffle(cases)
        n = len(cases)
        start = 0
        for s in splits:
            end = start + int(splits[s] * n)
            for case in cases[start:end]:
                shutil.move(f'{data_path}/{case}', f'{Path(data_path).parent}/{s}/{case}')
            start = end

        # Move remaining cases to first split (usually train)
        for case in os.listdir(f'{data_path}'):
            first_split = list(splits.keys())[0]
            shutil.move(f'{data_path}/{case}', f'{Path(data_path).parent}/{first_split}/{case}')

        shutil.rmtree(Path(data_path))

    @abstractmethod
    def generate_data(self, split_dir: Path):
        pass

    def get_random_in_range(self, l, h, rng):
        return l + rng.random() * (h - l)

    def import_mesh(self, mesh: str):
        ops.wm.obj_import(filepath=mesh, forward_axis='Y', up_axis='Z')

    def raise_with_log_text(self, case_path, text):
        with open(f'{case_path}/log.txt') as log:
            raise RuntimeError(f'{text} {case_path}\n\n {log.read()}')

    def get_location_inside(self, mesh: str):
        ops.object.select_all(action='SELECT')
        ops.object.delete()
        self.import_mesh(mesh)
        ops.object.select_all(action='SELECT')
        obj = bpy.context.object
        verts = [obj.matrix_world @ v.co for v in obj.data.vertices]
        verts = np.array(verts)
        center = np.sum(verts, axis=0) / len(verts)
        ops.object.delete()
        return center

    def get_location_inside(self, mesh: str):
        ops.object.select_all(action='SELECT')
        ops.object.delete()
        self.import_mesh(mesh)
        ops.object.select_all(action='SELECT')
        obj = bpy.context.object
        verts = [obj.matrix_world @ v.co for v in obj.data.vertices]
        verts = np.array(verts)
        center = np.sum(verts, axis=0) / len(verts)
        ops.object.delete()
        return center

    def is_sane(self, case_path):
        with open(f'{case_path}/constant/polyMesh/cellZones', 'r') as f:
            lines = f.read()
            match = re.search('>.+\n(\d+)\n\(', lines, flags=re.MULTILINE)
            n_porous = int(match.groups()[0])
        with open(f'{case_path}/0/cellToRegion', 'r') as f:
            lines = f.read()
            match = re.search('>.+\n(\d+)\n\(', lines, flags=re.MULTILINE)
            n_total = int(match.groups()[0])
        return n_porous < n_total / 2

    def generate_meta(self, data_dir: str | Path, *fields, max_dim=3):
        fields_min_max_tracker, count_min_max_tracker = MinMaxTracker(), MinMaxTracker()
        fields_stats_tracker, count_stats_tracker = Welford(), Welford()
        fields_columns, boundary_names = None, None
        elapse_times = []

        for case in track(glob.glob(f'{data_dir}/*/'), description='Generating metadata'):
            internal_fields = parse_internal_fields(case, *fields, max_dim=max_dim)
            boundary_fields = parse_boundary_fields(case, *fields, max_dim=max_dim)

            if fields_columns is None:
                fields_columns = internal_fields.columns
            if boundary_names is None:
                boundary_names = sorted(boundary_fields.index.unique())

            data = np.concatenate([internal_fields.to_numpy(), boundary_fields.to_numpy()])

            fields_min_max_tracker.update(data)
            fields_stats_tracker.add_all(data)
            elapse_times.append(parse_elapsed_time(case) / 1e6)

            points_counts = [len(internal_fields),
                             np.count_nonzero(internal_fields['cellToRegion'] > 0),
                             np.count_nonzero(internal_fields['cellToRegion'] == 0)]
            points_counts.extend(boundary_fields.groupby(boundary_fields.index).count().iloc[:, -1].values)

            points_counts = np.array([points_counts])
            count_min_max_tracker.update(points_counts)
            count_stats_tracker.add_all(points_counts)

        min_df = DataFrame(np.expand_dims(fields_min_max_tracker.min, 0), columns=fields_columns)
        max_df = DataFrame(np.expand_dims(fields_min_max_tracker.max, 0), columns=fields_columns)
        mean_df = DataFrame(np.expand_dims(fields_stats_tracker.mean, 0), columns=fields_columns)
        std_df = DataFrame(np.expand_dims(np.sqrt(fields_stats_tracker.var_p), 0), columns=fields_columns)
        fields_meta = {}

        for f in fields_columns.get_level_values(0).unique():
            fields_meta[f] = {
                'Min': min_df[f].to_numpy().flatten().tolist(),
                'Max': max_df[f].to_numpy().flatten().tolist(),
                'Mean': mean_df[f].to_numpy().flatten().tolist(),
                'Std': std_df[f].to_numpy().flatten().tolist()
            }

        timing_meta = {'Total': sum(elapse_times), 'Average': np.mean(elapse_times)}

        counts_df = DataFrame([[*count_min_max_tracker.min],
                               [*count_min_max_tracker.max],
                               [*count_stats_tracker.mean],
                               [*np.sqrt(count_stats_tracker.var_p)]],
                              index=['Min', 'Max', 'Mean', 'Std'],
                              columns=['internal', 'porous', 'fluid', *boundary_names])
        points_meta = {}
        for b in counts_df.columns:
            points_meta[b] = {
                'Min': counts_df[b].loc['Min'],
                'Max': counts_df[b].loc['Max'],
                'Mean': counts_df[b].loc['Mean'],
                'Std': counts_df[b].loc['Std']
            }

        meta_dict = {
            'Points': points_meta,
            'Stats': fields_meta,
            'Timing': timing_meta}

        with open(f'{data_dir}/meta.json', 'w') as meta:
            meta.write(json.dumps(meta_dict, indent=4))

    def clean_processor_data(self, data_dir):
        for case in glob.glob(f'{data_dir}/*/'):
            for proc in glob.glob(f'{case}/processor*/'):
                shutil.rmtree(proc)

    def generate_min_points(self, data_parent: str | Path):
        dicts = []
        for split in glob.glob(f'{data_parent}/*/'):
            if Path(split).name == 'plots':
                continue
            with open(f'{split}/meta.json', 'r') as f:
                dicts.append(json.load(f)['Points'])

        out = dict.fromkeys(dicts[0].keys(), sys.float_info.max)
        for d in dicts:
            out = {k: int(min(out[k], d[k]['Min'])) for k in d.keys()}

        with open(f'{data_parent}/min_points.json', 'w') as f:
            f.write(json.dumps(out))

    def generate(self, dest_dir, seed=8421):
        rng = Random(seed)

        dest_dir = Path(dest_dir)
        dest_dir.mkdir(exist_ok=True, parents=True)

        plots_dir = Path(dest_dir) / 'plots'
        plots_dir.mkdir(exist_ok=True, parents=True)

        if not self.meta_only:
            self.create_case_template_dirs()
            self.clean_dir(dest_dir)
            self.clean_dir(self.generated_meshes_dir)

            for mesh_set_path in self.meshes_sets_paths:
                generate_set_dir = self.generated_meshes_dir / mesh_set_path.name
                self.generate_transformed_meshes(mesh_set_path, generate_set_dir, rng=rng)

                set_dest_dir = dest_dir / mesh_set_path.name
                self.generate_openfoam_cases(generate_set_dir, set_dest_dir, mesh_set_path, rng=rng)

                self.generate_split(set_dest_dir, mesh_set_path, rng=rng)

        default_backend = matplotlib.get_backend()
        matplotlib.use('Agg')

        for split in glob.glob(f'{dest_dir}/*/'):
            split_path = Path(split)
            if split_path.name == 'plots':
                continue

            if not self.meta_only:
                self.generate_data(split_path)

            self.generate_meta(split_path, *self.fields, max_dim=len(self.dims))
            self.clean_processor_data(split_path)

            shutil.copyfile(self.data_config_path, split_path / 'data_config.json')

            case_plots_dir = plots_dir / split_path.name
            case_plots_dir.mkdir(exist_ok=True, parents=True)
            plot_dataset_dist(split, case_plots_dir)
            plot_u_direction_change(split, case_plots_dir)

        matplotlib.use(default_backend)

        self.generate_min_points(dest_dir)
