import glob
import math
import pathlib
from pathlib import Path
import shutil
import subprocess
import bpy
import mathutils
import numpy as np
from foamlib import FoamFile
from rich.progress import track
from bpy import ops
from datagen.data_generator import DataGeneratorBase


class AbcGenerator(DataGeneratorBase):

    def add_porous_meshes_to_case(self, case_path, meshes):
        surface_extract = FoamFile(f'{case_path}/system/surfaceFeatureExtractDict')
        template_extract = surface_extract['mesh.obj'].as_dict()
        surface_extract.pop('mesh.obj')

        snappy_dict = FoamFile(f'{case_path}/system/snappyHexMeshDict')
        template_feat = snappy_dict['castellatedMeshControls']['features']
        snappy_dict['castellatedMeshControls']['features'] = []
        template_feat = template_feat[0]

        template_geometry = snappy_dict['geometry']['mesh.obj'].as_dict()
        snappy_dict['geometry'].pop('mesh.obj')

        template_surf = snappy_dict['castellatedMeshControls']['refinementSurfaces']['mesh'].as_dict()
        snappy_dict['castellatedMeshControls']['refinementSurfaces'].pop('mesh')

        template_region = snappy_dict['castellatedMeshControls']['refinementRegions']['mesh'].as_dict()
        snappy_dict['castellatedMeshControls']['refinementRegions'].pop('mesh')

        for m in sorted(meshes):
            surface_extract[f'{m}.obj'] = template_extract
            template_geometry['name'] = m
            snappy_dict['geometry'][f'{m}.obj'] = template_geometry
            template_feat['file'] = f'{m}.eMesh'
            snappy_dict['castellatedMeshControls']['features'].append(template_feat)
            snappy_dict['castellatedMeshControls']['refinementSurfaces'][m] = template_surf
            snappy_dict['castellatedMeshControls']['refinementRegions'][m] = template_region
            snappy_dict['castellatedMeshControls']['refinementSurfaces'][m]['insidePoint'] = (
                self.get_location_inside(f'{case_path}/constant/triSurface/{m}.obj'))

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

    def create_case_template_dirs(self):
        (self.case_template_dir / 'constant/triSurface').mkdir(parents=True, exist_ok=True)

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
        meshes = glob.glob(f'{meshes_dir}/*.obj')

        ops.object.select_all(action='SELECT')
        ops.object.delete()
        for i in range(10):
            mesh = rng.choice(meshes)

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

    def generate_data(self, split_dir: Path):

        for case in track(glob.glob(f"{split_dir}/*"), description="Running cases"):
            process = subprocess.Popen(self.openfoam_bin, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                       stdout=subprocess.DEVNULL, text=True)
            process.communicate(f"{case}/Run")
            process.wait()
            if process.returncode != 0:
                self.raise_with_log_text(f'{case}', 'Failed to run ')
