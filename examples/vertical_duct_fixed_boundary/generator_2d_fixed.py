import glob
import json
import math
import pathlib
import random
import shutil
from pathlib import Path
import bpy
import mathutils
from bpy import ops
from foamlib import FoamFile

from datagen.data_generator import build_arg_parser
from datagen.generator_2d import Generator2DBase
from visualization.common import plot_u_direction_change, plot_dataset_dist
from visualization.visualization_2d import plot_case


class Generator2DFixedHardTop(Generator2DBase):
    def get_location_inside(self, mesh: str):
        location = super().get_location_inside(mesh)
        location[-1] = 0
        return location

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

    def generate_openfoam_cases(self, meshes_dir, dest_dir, case_config_dir, rng):
        dest_dir.mkdir(parents=True, exist_ok=True)

        mesh_containers = glob.glob(f"{meshes_dir}/*/")
        for m_c in mesh_containers:
            case_path = f"{dest_dir}/{Path(m_c).stem}"
            shutil.copytree(self.case_template_dir, case_path)

            meshes = [pathlib.Path(s).stem for s in glob.glob(f"{m_c}/*.obj")]
            for m in meshes:
                shutil.copyfile(f'{m_c}/{m}.obj',
                                f"{case_path}/snappyHexMesh/constant/triSurface/{m}.obj")

            self.add_porous_meshes_to_case(f"{case_path}/snappyHexMesh", meshes)

            self.set_decompose_par(f'{case_path}/snappyHexMesh')
            self.set_decompose_par(f'{case_path}/simpleFoam')

    def generate_object(self, meshes_dir, src_meshes, rng):
        src_mesh = rng.choice(src_meshes)
        self.import_mesh(f'{meshes_dir}/{src_mesh}')
        ops.object.select_all(action='SELECT')
        src_mesh = bpy.context.selected_objects[0]

        src_mesh.rotation_euler = mathutils.Euler((0.0, 0.0, rng.random() * 2 * math.pi))

        meshes = [src_mesh]

        for i in range(random.randint(1, 4)):
            mesh = rng.choice(src_meshes)
            self.import_mesh(f'{meshes_dir}/{mesh}')
            obj = bpy.context.selected_objects[0]
            obj.select_set(True)

            obj.rotation_euler = mathutils.Euler((0.0, 0.0, rng.random() * 2 * math.pi))
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

            bpy.ops.object.editmode_toggle()
            bpy.ops.mesh.select_all(action='SELECT')

            offset = (((rng.random() - 0.5) * 2 * src_mesh.dimensions[0] / 2),
                      ((rng.random() - 0.5) * 2 * src_mesh.dimensions[1] / 2))

            bpy.ops.transform.translate(value=(*offset, 0),
                                        orient_type='GLOBAL')
            bpy.ops.object.editmode_toggle()

            meshes.append(obj)
        return meshes

    def merge_mehses(self, meshes):
        ops.object.select_all(action='DESELECT')
        mesh = meshes[0]
        mesh.select_set(True)
        for i, t in enumerate(meshes[:-1]):
            modifier = mesh.modifiers.new(name="Boolean", type='BOOLEAN')
            modifier.operation = 'UNION'
            modifier.object = meshes[i + 1]
            bpy.context.view_layer.objects.active = mesh
            bpy.ops.object.modifier_apply(modifier=modifier.name)
        return mesh

    def generate_transformed_meshes(self, meshes_dir: Path, dest_dir: Path, rng: random.Random):
        dest_dir.mkdir(parents=True, exist_ok=True)

        with open(f'{meshes_dir}/transforms.json', 'r') as f:
            dest_dir.mkdir(parents=True, exist_ok=True)
            ops.ed.undo_push()
            ops.object.select_all(action='SELECT')
            ops.object.delete()
            meshes = list(json.load(f).keys())
            for i in range(200):
                mesh_base_path = dest_dir / str(i)
                mesh_base_path.mkdir()
                gen_meshes = self.generate_object(meshes_dir, meshes, rng)
                obj = self.merge_mehses(gen_meshes)
                bpy.ops.object.select_all(action='DESELECT')
                obj.select_set(True)

                modifier = obj.modifiers.new(name="Remesh", type='REMESH')
                modifier.voxel_size = 0.002
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.modifier_apply(modifier=modifier.name)

                bpy.ops.object.transform_apply()

                ops.wm.obj_export(filepath=f'{mesh_base_path}/mesh.obj',
                                  forward_axis='Y',
                                  up_axis='Z',
                                  export_materials=False,
                                  export_selected_objects=True)

                # Delete all
                ops.object.select_all(action='SELECT')
                ops.object.delete()
