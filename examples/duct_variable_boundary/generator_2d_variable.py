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
from foamlib import FoamFile
from datagen.generator_2d import Generator2DBase


class Generator2DVariable(Generator2DBase):
    """
    Base generator for 2D variable boundary conditions OpenFOAM cases.

    The cases are set up using a rectangular duct and augmented porous objects. The variable boundary conditions are the Darcy and Forchheimer coefficients, the inlet velocity and the inlet angle.
    Data augmentation is taken from transforms.json while the boundary conditions from config.json using all possible values combinations respectively. Each case is dropped with probability p from the final dataset.
    The position of porous objects and the inlet velocities are jittered randomly by (0.5,0.1) and 0.015 m/s. The inlet angle is randomly sampled within the limits defined in config.json.
    """

    def generate_transformed_meshes(self, meshes_dir: Path, dest_dir: Path, rng: Random):
        with open(f'{meshes_dir}/transforms.json', 'r') as f:
            ops.ed.undo_push()
            ops.object.select_all(action='SELECT')
            ops.object.delete()
            for mesh, transforms in json.load(f).items():
                self.import_mesh(f'{meshes_dir}/{mesh}')
                rotations = self.parse_rotations(transforms['rotation'])
                scales = self.parse_scale(transforms['scale'])
                params = list(itertools.product(rotations, scales))
                for r, s in params:
                    if len(params) > 1 and rng.random() > self.drop_p:
                        continue
                    ops.object.select_all(action='SELECT')
                    ops.object.duplicate(linked=False)
                    obj = bpy.context.selected_objects[0]

                    obj.scale = mathutils.Vector((s[0], s[1], 1.0))
                    obj.rotation_euler = mathutils.Euler((0.0, 0.0, math.radians(-r)))
                    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

                    bpy.context.view_layer.objects.active = obj
                    bpy.ops.object.editmode_toggle()
                    bpy.ops.mesh.select_all(action='SELECT')

                    offset = (rng.random() * 0.15, (rng.random() - 0.5) * 2 * 0.1)
                    bpy.ops.transform.translate(value=(*offset, 0),
                                                orient_type='GLOBAL')
                    bpy.ops.object.editmode_toggle()

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

    def generate_openfoam_cases(self, meshes_dir: Path, dest_dir: Path, case_config_dir: Path, rng: Random):
        with open(f'{case_config_dir}/config.json', 'r') as config:
            config = json.load(config)['cfd params']
            params = list(itertools.product(config['inlet'], config['coeffs']))
            inlet_angles = self.parse_angles(config)
            for inlet_u, coeffs in params:
                meshes = glob.glob(f"{meshes_dir}/*.obj")
                for m in meshes:
                    if len(params) > 1 and rng.random() > self.drop_p:
                        continue
                    d = coeffs['d']
                    f = coeffs['f']
                    random_inlet_u = inlet_u + (rng.random() - 0.5) * 2 * 0.015
                    inlet_angle = min(inlet_angles) + (max(inlet_angles) - min(inlet_angles)) * rng.random()
                    inlet_angle_rad = math.radians(inlet_angle)
                    u_x, u_y = random_inlet_u * math.cos(inlet_angle_rad), random_inlet_u * math.sin(inlet_angle_rad)

                    case_path = f"{dest_dir}/{Path(m).stem}_d{d[0]}_{f[0]}_in{random_inlet_u:.4f}_a{inlet_angle:.2f}"
                    shutil.copytree(self.case_template_dir, case_path)
                    shutil.copyfile(m, f"{case_path}/snappyHexMesh/constant/triSurface/mesh.obj")

                    self.write_locations_in_mesh(f'{case_path}/snappyHexMesh', self.get_location_inside(m))
                    FoamFile(f'{case_path}/simpleFoam/0/U')['internalField'] = [u_x, u_y, 0]
                    fv_options = f'{case_path}/simpleFoam/system/fvOptions'

                    self.write_coefs(fv_options, d, 'd')
                    self.write_coefs(fv_options, f, 'f')

                    self.set_decompose_par(f'{case_path}/snappyHexMesh')
                    self.set_decompose_par(f'{case_path}/simpleFoam')
