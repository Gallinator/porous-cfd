import glob
import itertools
import json
import math
import shutil
from pathlib import Path
from random import Random

import bpy
import mathutils
from bpy import ops
from datagen.generator_2d import Generator2DBase


class Generator2DFixed(Generator2DBase):
    """
    Generator of the 2D cases with fixed boundary conditions.

    Porous objects are added to rectangular 2D duct. Data is augmented with rotation and scaling of the porous object.
    """

    def generate_openfoam_cases(self, meshes_dir, dest_dir, case_config_dir, rng: Random):
        meshes = glob.glob(f"{meshes_dir}/*.obj")
        for m in meshes:
            case_path = f"{dest_dir}/{Path(m).stem}"
            shutil.copytree(self.case_template_dir, case_path)
            shutil.copyfile(m, f"{case_path}/snappyHexMesh/constant/triSurface/mesh.obj")
            self.write_locations_in_mesh(f'{case_path}/snappyHexMesh', self.get_location_inside(m))

            self.set_decompose_par(f'{case_path}/snappyHexMesh')
            self.set_decompose_par(f'{case_path}/simpleFoam')

    def generate_transformed_meshes(self, meshes_dir: Path, dest_dir: Path, rng: Random):
        """
        Generates fixed boundary porous meshes.

        Rotation and scaling is applied from the transforms.json file using all the possible values combinations.
        """
        with open(f'{meshes_dir}/transforms.json', 'r') as f:
            ops.ed.undo_push()
            ops.object.select_all(action='SELECT')
            ops.object.delete()
            for mesh, transforms in json.load(f).items():
                self.import_mesh(f'{meshes_dir}/{mesh}')
                rotations = self.parse_rotations(transforms['rotation'])
                scales = self.parse_scale(transforms['scale'])
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
