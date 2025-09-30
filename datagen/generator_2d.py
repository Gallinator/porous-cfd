import glob
import itertools
import os
import shutil
import subprocess
from abc import abstractmethod
from pathlib import Path
import numpy as np
from rich.progress import track

from datagen.data_generator import DataGeneratorBase
from datagen.momentum_error import write_momentum_error


class Generator2DBase(DataGeneratorBase):
    def __init__(self, src_dir, openfoam_bin, n_procs: int, keep_p=0.5, meta_only=False):
        super().__init__(src_dir, openfoam_bin, n_procs, keep_p, meta_only)
        self.write_momentum = True

    def create_case_template_dirs(self):
        (self.case_template_dir / 'snappyHexMesh/0').mkdir(parents=True, exist_ok=True)
        (self.case_template_dir / 'snappyHexMesh/constant/triSurface').mkdir(parents=True,
                                                                             exist_ok=True)

    def parse_rotations(self, rotations_dict: dict):
        if not rotations_dict:
            return [0]
        start, stop, n = rotations_dict[0], rotations_dict[1], rotations_dict[2]
        return np.linspace(start, stop, n)

    def parse_scale(self, scale_dict: dict) -> list:
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

    def parse_angles(self, config: dict):
        if 'angle' in config.keys():
            return np.linspace(*config['angle'])
        else:
            return [0]

    @abstractmethod
    def generate_transformed_meshes(self, meshes_dir: Path, dest_dir: Path, rng):
        pass

    @abstractmethod
    def generate_openfoam_cases(self, meshes_dir, dest_dir, case_config_dir, rng):
        pass

    def generate_data(self, split_dir: Path):
        for case in track(glob.glob(f"{split_dir}/*"), description="Generating geometries"):
            process = subprocess.Popen(self.openfoam_bin, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                       stdout=subprocess.DEVNULL, text=True)
            process.communicate(f"{case}/snappyHexMesh/Run")
            process.wait()
            if process.returncode != 0:
                self.raise_with_log_text(f'{case}/snappyHexMesh', 'Failed to generate mesh for case ')

        for case in track(glob.glob(f"{split_dir}/*"), description="Running cases"):
            process = subprocess.Popen(self.openfoam_bin, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                       stdout=subprocess.DEVNULL, text=True)
            process.communicate(f"{case}/simpleFoam/Run")
            process.wait()
            if process.returncode != 0:
                self.raise_with_log_text(f'{case}/simpleFoam', 'Failed to run ')

            if self.write_momentum:
                write_momentum_error(f"{case}/simpleFoam")

            self.clean_dir(f"{case}/snappyHexMesh")
            os.rmdir(f"{case}/snappyHexMesh")
            shutil.move(f"{case}/simpleFoam", 'tmp')
            os.rmdir(f'{case}')
            shutil.move("tmp", f'{case}')
