import glob
import json
import math
import shutil
from pathlib import Path
import bpy
import mathutils
from bpy import ops
from datagen.generator_2d import Generator2DBase


class GeneratorManufactured(Generator2DBase):
    def generate_openfoam_cases(self, meshes_dir, dest_dir, case_config_dir, rng):
        dest_dir.mkdir(parents=True, exist_ok=True)

        meshes = glob.glob(f"{meshes_dir}/*.obj")
        for m in meshes:
            case_path = f"{dest_dir}/{Path(m).stem}"
            shutil.copytree(self.case_template_dir, case_path)
            shutil.copyfile(m, f"{case_path}/snappyHexMesh/constant/triSurface/mesh.obj")
            self.write_locations_in_mesh(f'{case_path}/snappyHexMesh', self.get_location_inside(m))

            self.set_decompose_par(f'{case_path}/snappyHexMesh')
            self.set_decompose_par(f'{case_path}/simpleFoam')

    def generate_transformed_meshes(self, meshes_dir: Path, dest_dir: Path, rng):
        dest_dir.mkdir(parents=True, exist_ok=True)

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
