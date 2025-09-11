import json
import math
from pathlib import Path

import bpy
import mathutils
from bpy import ops
from datagen.data_generator import build_arg_parser
from datagen.generator_2d_fixed import Generator2DFixed


class GeneratorManufactured(Generator2DFixed):
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


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    OPENFOAM_COMMAND = f'{args.openfoam_dir}/etc/openfoam'
    generator = GeneratorManufactured('../assets/manufactured', OPENFOAM_COMMAND, args.openfoam_procs, 0.5)
    generator.generate(args.data_root_dir)
