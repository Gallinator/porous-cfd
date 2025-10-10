import glob
import subprocess
from abc import abstractmethod
from pathlib import Path
from warnings import warn
import mathutils
import numpy as np
from rich.progress import track
import bpy
from bpy import ops
from datagen.data_generator import DataGeneratorBase
from datagen.momentum_error import write_momentum_error


class Generator3DBase(DataGeneratorBase):
    def get_location_inside(self, mesh: str):
        ops.object.select_all(action='SELECT')
        ops.object.delete()
        self.import_mesh(mesh)
        ops.object.select_all(action='SELECT')
        obj = bpy.context.object

        s_x, s_y, s_z = obj.dimensions / 2
        x, y, z = np.meshgrid(np.linspace(-s_x, s_x, 20),
                              np.linspace(-s_y, s_y, 20),
                              np.linspace(-s_z, s_z, 20))
        grid = np.stack([x.flatten(), y.flatten(), z.flatten()]).T

        _, closest, normal, _ = zip(*[obj.closest_point_on_mesh(g) for g in grid])
        dir = np.array(closest) - grid
        dot = np.sum(np.array(normal) * dir, axis=-1)

        inside_mask = dot.flatten() > 0
        inside_grid = grid[inside_mask]
        dist = np.linalg.norm(dir[inside_mask], axis=-1)
        center = inside_grid[np.argmax(dist)]
        center = obj.matrix_world @ mathutils.Vector(center)
        ops.object.delete()

        return np.array(center)

    def create_case_template_dirs(self):
        (self.case_template_dir / 'constant/triSurface').mkdir(parents=True, exist_ok=True)

    def generate_data(self, split_dir: Path):
        for case in track(glob.glob(f"{split_dir}/*"), description="Running cases"):
            process = subprocess.Popen(self.openfoam_bin, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                       stdout=subprocess.DEVNULL, text=True)
            process.communicate(f"{case}/Run")
            process.wait()
            if process.returncode != 0:
                self.raise_with_log_text(f'{case}', 'Failed to run ')

            write_momentum_error(case)

            if not self.is_sane(case):
                warn(f'Case {case} is not sane, please check for errors!')

    @abstractmethod
    def generate_transformed_meshes(self, meshes_dir: Path, dest_dir: Path, rng):
        pass

    @abstractmethod
    def generate_openfoam_cases(self, meshes_dir, dest_dir, case_config_dir, rng):
        pass
