import glob
import itertools
import json
import pathlib
from pathlib import Path
import re
import shutil
import subprocess
import bpy
from foamlib import FoamFile
from rich.progress import track
from bpy import ops
from datagen.data_generator import DataGeneratorBase


class WindbreakGeneratorBase(DataGeneratorBase):

    def create_case_template_dirs(self):
        (self.case_template_dir / '/constant/triSurface').mkdir(parents=True, exist_ok=True)

    def merge_trees(self, trees):
        ops.object.select_all(action='DESELECT')
        windbreak = trees[0]
        windbreak.select_set(True)
        for i, t in enumerate(trees[:-1]):
            modifier = windbreak.modifiers.new(name="Boolean", type='BOOLEAN')
            modifier.operation = 'UNION'
            modifier.object = trees[i + 1]
            bpy.context.view_layer.objects.active = windbreak
            bpy.ops.object.modifier_apply(modifier=modifier.name)
        return windbreak

    def create_windbreak(self, src_tree, n_trees, scales, rng):
        trees = []
        prev_obj = src_tree
        for n in range(n_trees):
            ops.object.select_all(action='DESELECT')
            src_tree.select_set(True)
            ops.object.duplicate(linked=False)
            obj = bpy.context.selected_objects[0]

            scale_xy = self.get_random_in_range(*scales['xy'], rng=rng)
            scale_z = self.get_random_in_range(*scales['z'], rng=rng)
            obj.scale = (scale_xy, scale_xy, scale_z)
            rot_z = self.get_random_in_range(0, 360, rng=rng)
            obj.rotation_euler = (*obj.rotation_euler[0:2], rot_z)
            bpy.ops.object.transform_apply(scale=False, location=False, rotation=True)
            y_size = obj.dimensions[1]
            prev_y_size = prev_obj.dimensions[1]
            if n > 0:
                obj.location[1] = prev_obj.location[1] + prev_y_size / 2 + y_size / 2 * 0.8
            trees.append(obj)
            prev_obj = obj
        return trees

    def generate_transformed_meshes(self, meshes_dir, dest_dir: Path, rng):
        dest_dir.mkdir(parents=True, exist_ok=True)

        with open(f'{meshes_dir}/transforms.json', 'r') as f:
            ops.ed.undo_push()
            ops.object.select_all(action='SELECT')
            ops.object.delete()
            for mesh, transforms in json.load(f).items():
                self.import_mesh(f'{meshes_dir}/{mesh}')
                n_trees = transforms['n_trees']
                scales = transforms['scale']
                src_obj = bpy.context.selected_objects[0]
                ops.object.select_all(action='DESELECT')

                for i in range(transforms['n_windbreaks']):

                    trees = self.create_windbreak(src_obj, n_trees, scales, rng)

                    windbreak = self.merge_trees(trees)

                    bpy.ops.object.select_all(action='DESELECT')
                    windbreak.select_set(True)

                    modifier = windbreak.modifiers.new(name="Remesh", type='REMESH')
                    modifier.voxel_size = 0.2
                    bpy.context.view_layer.objects.active = windbreak
                    bpy.ops.object.modifier_apply(modifier=modifier.name)

                    bpy.context.view_layer.objects.active = windbreak
                    bpy.ops.object.transform_apply()
                    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME')
                    windbreak.location = [0, 0, windbreak.location[2]]

                    ops.wm.obj_export(filepath=f'{dest_dir}/{i}_{mesh}',
                                      forward_axis='Y',
                                      up_axis='Z',
                                      export_materials=False,
                                      export_selected_objects=True)
                    for t in trees:
                        t.select_set(True)
                    # Delete copies
                    ops.object.delete()
            # Delete original
            ops.object.select_all(action='SELECT')
            ops.object.delete()

            shutil.copytree(f'{meshes_dir}/houses', f'{dest_dir}/houses')

    def generate_openfoam_cases(self, meshes_dir, dest_dir, case_config_dir, rng):
        dest_dir.mkdir(parents=True, exist_ok=True)

        with open(f'{case_config_dir}/config.json', 'r') as config:
            config = json.load(config)['cfd params']
            meshes = glob.glob(f"{meshes_dir}/*.obj")
            houses = glob.glob(f'{meshes_dir}/houses/*.obj')
            params = list(itertools.product(meshes, config['inlet']))
            for m, inlet_ux in params:
                mesh_name = re.match('.+_(.+obj)', m)[1]
                d = config['trees'][mesh_name]['d']
                f = config['trees'][mesh_name]['f']
                case_path = f"{dest_dir}/{pathlib.Path(m).stem}_d{d[0]}_{f[0]}_in{inlet_ux}"
                shutil.copytree(self.case_template_dir, case_path)
                shutil.copyfile(m, f"{case_path}/constant/triSurface/mesh.obj")

                rand_house = houses[rng.randint(0, len(houses) - 1)]
                shutil.copyfile(rand_house, f"{case_path}/constant/triSurface/solid.obj")

                self.write_locations_in_mesh(f'{case_path}', self.get_location_inside(m))

                FoamFile(f'{case_path}/0/U')['internalField'] = [inlet_ux, 0, 0]

                fv_options = f'{case_path}/system/fvOptions'
                self.write_coefs(fv_options, d, 'd')
                self.write_coefs(fv_options, f, 'f')

                self.set_decompose_par(f'{case_path}')

    def generate_data(self, split_dir: Path):
        for case in track(glob.glob(f"{split_dir}/*"), description="Running cases"):
            process = subprocess.Popen(self.openfoam_bin, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                       stdout=subprocess.DEVNULL, text=True)
            process.communicate(f"{case}/Run")
            process.wait()
            if process.returncode != 0:
                self.raise_with_log_text(f'{case}', 'Failed to run ')
