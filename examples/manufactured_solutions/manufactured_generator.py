import glob
import json
import math
import shutil
from pathlib import Path
from random import Random
import bpy
import mathutils
from bpy import ops
from datagen.generator_2d import Generator2DBase


class GeneratorManufactured(Generator2DBase):
    """
    Generator for the manufactured solution data.

    This class only generates cases meshes.
    Data augmentation consists in rotation and scaling as defined in transforms.json using all possible values combinations.
    """

    def __init__(self, src_dir: str, openfoam_bin: str, n_procs: int, meta_only=False):
        super().__init__(src_dir, openfoam_bin, n_procs, meta_only)
        self.write_momentum = False
        # Disable plots as only geometry dta is generated
        self.save_plots = False

    def generate_openfoam_cases(self, meshes_dir: Path, dest_dir: Path, case_config_dir: Path, rng: Random):
        meshes = glob.glob(f"{meshes_dir}/*.obj")
        for m in meshes:
            case_path = f"{dest_dir}/{Path(m).stem}"
            shutil.copytree(self.case_template_dir, case_path)
            shutil.copyfile(m, f"{case_path}/snappyHexMesh/constant/triSurface/mesh.obj")
            self.write_locations_in_mesh(f'{case_path}/snappyHexMesh', self.get_location_inside(m))

            self.set_decompose_par(f'{case_path}/snappyHexMesh')
            self.set_decompose_par(f'{case_path}/simpleFoam')

    def generate_transformed_meshes(self, meshes_dir: Path, dest_dir: Path, rng: Random):
        with open(f'{meshes_dir}/transforms.json', 'r') as f:
            ops.ed.undo_push()
            ops.object.select_all(action='SELECT')
            ops.object.delete()
            for mesh, transforms in json.load(f).items():
                self.import_mesh(f'{meshes_dir}/{mesh}')
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
