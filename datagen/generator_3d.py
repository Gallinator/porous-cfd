import glob
import shutil
import subprocess
from abc import abstractmethod
from pathlib import Path
from random import Random
from warnings import warn
import mathutils
import numpy as np
from rich.progress import track
import bpy
from bpy import ops
from datagen.data_generator import DataGeneratorBase
from datagen.momentum_error import write_momentum_error


class Generator3DBase(DataGeneratorBase):
    """
    Base class for 3D cases generation.
    """

    def get_location_inside(self, mesh_path: str):
        """
        Calculates a valid value for the location inside to be set in the snappyHexMesh dict. It works by laying a grid of uniform points over the mesh and selecting the point inside the mesh with the maximum distance from the surface.
        """

        ops.object.select_all(action='SELECT')
        ops.object.delete()
        self.import_mesh(mesh_path)
        ops.object.select_all(action='SELECT')
        obj = bpy.context.object
        verts = np.array([v.co for v in obj.data.vertices])

        min_b, max_b = np.min(verts, axis=0), np.max(verts, axis=0)

        x, y, z = np.meshgrid(np.linspace(min_b[0], max_b[0], 20),
                              np.linspace(min_b[1], max_b[1], 20),
                              np.linspace(min_b[2], max_b[2], 20))
        grid = np.stack([x.flatten(), y.flatten(), z.flatten()]).T

        _, closest, normal, _ = zip(*[obj.closest_point_on_mesh(g) for g in grid])

        dir = np.array(closest) - grid
        norm_dir = dir / np.vstack(np.linalg.norm(dir, axis=-1))
        dot = np.sum(np.array(normal) * norm_dir, axis=-1)

        inside_mask = dot.flatten() > 0.5
        inside_grid = grid[inside_mask]

        dist = np.linalg.norm(dir[inside_mask], axis=-1)
        center = inside_grid[np.argmax(dist)]
        center = obj.matrix_world @ mathutils.Vector(center)

        ops.object.delete()
        return np.array(center)

    def create_case_template_dirs(self):
        (self.case_template_dir / 'constant/triSurface').mkdir(parents=True, exist_ok=True)

    def generate_data(self, split_dir: Path):
        """
        Run £D simulations inside split_dir.
        :raises: RuntimeError if one case fails.
        """
        for case in track(glob.glob(f"{split_dir}/*"), description="Running cases"):
            process = subprocess.Popen(self.openfoam_bin, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                       stdout=subprocess.DEVNULL, text=True, start_new_session=True)
            process.communicate(f"{case}/Run")
            process.wait()
            if process.returncode != 0:
                self.raise_with_log_text(f'{case}', 'Failed to run ')

            write_momentum_error(case)

            if not self.is_sane(case):
                warn(f'Case {case} is malformed, will be deleted!')
                shutil.rmtree(case)

    @abstractmethod
    def generate_transformed_meshes(self, meshes_dir: Path, dest_dir: Path, rng: Random):
        pass

    @abstractmethod
    def generate_openfoam_cases(self, meshes_dir, dest_dir, case_config_dir: Path, rng: Random):
        pass
