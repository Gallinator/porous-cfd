import glob
import itertools
import json
import pathlib
from pathlib import Path
import re
import shutil
from random import Random

import bpy
from foamlib import FoamFile
from bpy import ops
from datagen.generator_3d import Generator3DBase
import bmesh
from mathutils.bvhtree import BVHTree
from bpy.types import Object


def get_bvh_tree(obj: Object) -> BVHTree:
    """
    Calculates the Bounding Volume Hierarchy tree for obj.
    """
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.transform(obj.matrix_world)
    bvh = BVHTree.FromBMesh(bm)
    bm.free()
    return bvh


class WindbreakGenerator(Generator3DBase):
    """
    Data generator for the 3D windbreaks experiment with variable inlet velocity.

    The data is generated from a sample of tree species models, each containing a single tree, and a sample of 3D house models.
    Each tree is randomly rotated and scaled on the z-axis and xy plane, then it is arranged into a row.
    All trees in a row are merged with a Union boolean modifier, then remeshed and smoothed using the Catmull-Clark subdivision.
    The number of trees in each windbreak and the number of windbreaks to generate for each species is defined in transforms.json.
    One house model is added to each case.
    The transforms.json defines the data augmentation while the config.json the inlet velocities.
    """

    def merge_trees(self, trees: list[Object]) -> Object:
        """
        Merges a row of trees into a windbreak model.
        :param trees: The trees forming the row.
        :return: The merged windbreak object.
        """
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

    def create_windbreak(self, src_tree: Object, n_trees: int, scales: dict, rng: Random) -> list[Object]:
        """
        Creates a windbreak starting from a single tree object.

        The tree is duplicated, randomly rotated and scaled. To the windbreak each tree is offset with respect to the original such that it intersect the other model.
        The intersection check is carried out using a Bounding Volume Hierarchy tree.
        :param src_tree: The source tree model.
        :param n_trees: The number of trees to place into the row.
        :param scales: Scaling factor dictionary containing the z and xy keys.
        :param rng: Used to randomize the rotation and scaling.
        :return: A list of trees arranged into a windbreak.
        """
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

            # Iteratively offset the current tree until intersection
            if n > 0:
                prev_bvh = get_bvh_tree(prev_obj)
                obj.location[1] = prev_obj.location[1] + prev_obj.dimensions[1] / 2
                while prev_bvh.overlap(get_bvh_tree(obj)) is None:
                    obj.location[1] = obj.location[1] - 0.1
            trees.append(obj)
            prev_obj = obj
        return trees

    def generate_transformed_meshes(self, meshes_dir: Path, dest_dir: Path, rng: Random):
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

    def generate_openfoam_cases(self, meshes_dir: Path, dest_dir: Path, case_config_dir: Path, rng: Random):
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
