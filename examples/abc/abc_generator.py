import glob
import math
from pathlib import Path
import shutil
import bpy
import mathutils
import numpy as np
from bpy import ops
from datagen.generator_3d import Generator3DBase


class AbcGenerator(Generator3DBase):
    def get_location_inside(self, mesh: str):
        ops.object.select_all(action='SELECT')
        ops.object.delete()
        self.import_mesh(mesh)
        ops.object.select_all(action='SELECT')
        obj = bpy.context.object
        verts = [obj.matrix_world @ v.co for v in obj.data.vertices]
        verts = np.array(verts)
        max_vertex = np.argmax(verts, axis=0)
        center = verts[max_vertex[-1], :] - np.array((0, 0, 0.001))
        ops.object.delete()
        return center

    def align_to_x(self, obj):
        sorted_dims = np.argsort(obj.dimensions)
        # Align to z
        if sorted_dims[-1] == 0:
            obj.rotation_euler = mathutils.Euler((0, math.pi / 2, 0))
        bpy.ops.object.transform_apply()

        # Align to y
        sorted_dims = np.argsort(obj.dimensions)
        if sorted_dims[1] == 0:
            obj.rotation_euler = mathutils.Euler((0, 0, math.pi / 2))

    def set_com_and_recenter(self, obj):
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        obj.location = [0, 0, 0]

    def rescale(self, obj):
        duct_size = np.array([1, 0.6, 0.6])
        delta = np.abs(np.array(obj.dimensions) - duct_size)
        max_dim = np.argmax(delta)
        tgt_scale = (duct_size[max_dim] * 0.65) / obj.dimensions[max_dim]
        obj.scale = obj.scale * tgt_scale

    def generate_transformed_meshes(self, meshes_dir, dest_dir: Path, rng):
        dest_dir.mkdir(parents=True, exist_ok=True)

        ops.object.select_all(action='SELECT')
        ops.object.delete()
        for mesh in glob.glob(f'{meshes_dir}/*.obj'):
            meshes_subfolder = dest_dir / f'{Path(mesh).stem}'
            meshes_subfolder.mkdir(exist_ok=True, parents=True)

            ops.object.select_all(action='DESELECT')

            self.import_mesh(f'{mesh}')
            obj = bpy.context.selected_objects[0]

            self.set_com_and_recenter(obj)
            bpy.ops.object.transform_apply()

            self.align_to_x(obj)
            bpy.ops.object.transform_apply()

            self.rescale(obj)
            bpy.ops.object.transform_apply()

            ops.wm.obj_export(filepath=f'{meshes_subfolder}/mesh.obj',
                              forward_axis='Y',
                              up_axis='Z',
                              export_materials=False,
                              export_selected_objects=True)

            # Delete original
            ops.object.select_all(action='SELECT')
            ops.object.delete()

    def generate_openfoam_cases(self, meshes_dir, dest_dir, case_config_dir, rng):
        dest_dir.mkdir(parents=True, exist_ok=True)

        for mesh_set in glob.glob(f'{meshes_dir}/*/'):
            case_path = f"{dest_dir}/{Path(mesh_set).name}"
            shutil.copytree(self.case_template_dir, case_path)

            shutil.copyfile(f"{mesh_set}mesh.obj", f"{case_path}/constant/triSurface/mesh.obj")

            self.set_decompose_par(f'{case_path}')
