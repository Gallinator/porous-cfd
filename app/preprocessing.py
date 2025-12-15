import shutil
from pathlib import Path
import bpy
import numpy as np


def create_session_folders(assets_dir: str, session_dir: str):
    assets_dir = Path(assets_dir)
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)

    shutil.copytree(assets_dir / "openfoam-case-template", session_dir / "assets" / "openfoam-case-template")
    shutil.copy(assets_dir / "data_config.json", session_dir / "assets")
    split_dir = session_dir / "assets" / "meshes" / "split"
    split_dir.mkdir(exist_ok=True, parents=True)
    shutil.copy(assets_dir / "transforms.json", split_dir)

    data_dir = session_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)


def path_to_obj(x: list[float], y: list[float], dest_path: str):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    mesh = bpy.data.meshes.new("mesh")
    obj = bpy.data.objects.new("object", mesh)

    coords = [(x, y, 0.05) for (x, y) in zip(x, y)]
    edges = [(i, (i + 1) % len(x)) for i in range(len(x))]
    face = np.arange(len(x))

    mesh.from_pydata(coords, edges, [face])
    mesh.update()

    bpy.context.collection.objects.link(obj)

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    bpy.ops.object.editmode_toggle()

    bpy.ops.mesh.extrude_region_move(
        TRANSFORM_OT_translate={"value": (-0, -0, -0.1),
                                "constraint_axis": (False, False, True)})

    bpy.ops.object.editmode_toggle()

    bpy.ops.wm.obj_export(filepath=f'{dest_path}/mesh.obj',
                          forward_axis='Y',
                          up_axis='Z',
                          export_materials=False,
                          export_selected_objects=True)

    bpy.ops.object.delete()
