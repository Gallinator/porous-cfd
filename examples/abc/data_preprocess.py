import glob
import os.path
import shutil
import subprocess
import tarfile
from functools import partial
from pathlib import Path
from urllib.request import urlopen
from rich.progress import Progress, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn, \
    track
from bpy import ops
import bpy
import bmesh


def download_from_gdrive(file_id, file_out):
    if os.path.exists(file_out):
        print(f'{file_out} already downloaded! Remove it manually download again.')
        return
    download_process = subprocess.run(
        ['wget',
         '--no-check-certificate',
         f'https://drive.usercontent.google.com/download?id={file_id}=t',
         '-O',
         file_out],
        check=True,
        cwd=os.getcwd(),
        stdout=subprocess.PIPE,
        text=True)


def download(url, dest_file):
    if os.path.exists(dest_file):
        print(f'{dest_file} already downloaded!')
        return
    progress = Progress(
        TextColumn("Downloading {task.fields[filename]}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        DownloadColumn(),
        "•",
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
    )
    response = urlopen(url)
    with progress:
        task = progress.add_task('download', filename=url.split('/')[-1], total=int(response.info()['Content-length']))
        with open(dest_file, 'wb') as f:
            for block in iter(partial(response.read, 32768), b""):
                f.write(block)
                progress.update(task, advance=len(block))


def extract(data_file, dest_dir):
    Path(dest_dir).mkdir(exist_ok=True, parents=True)
    with tarfile.open(data_file, 'r') as f:
        for c in track(f.getmembers(), description=f'Extracting {f.name}'):
            f.extract(c, dest_dir)


def is_manifold():
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.select_non_manifold()
    v, _, _ = bpy.context.active_object.data.count_selected_items()
    bpy.ops.object.editmode_toggle()
    return v == 0


def convert_to_obj(data_dir, meshconv_path):
    progress = Progress(
        TextColumn("Converting {task.fields[filename]}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•", TimeRemainingColumn(),
    )
    files = glob.glob(f'{data_dir}/**/*.off', recursive=True)
    with progress:
        task = progress.add_task('convert', total=len(files), filename='')
        for f in files:
            process = subprocess.Popen('/bin/bash',
                                       stdin=subprocess.PIPE,
                                       # stderr=subprocess.DEVNULL,
                                       stdout=subprocess.DEVNULL,
                                       text=True)
            progress.update(task, advance=1, filename=Path(f).name)
            process.communicate(f"{meshconv_path} {f} -c obj")
            process.wait()


def has_multiple_islands(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    vertices = bm.verts
    unseen_v_idx = {v.index for v in vertices[1:]}
    frontier_v = {vertices[0]}

    while len(frontier_v) > 0:
        v = frontier_v.pop()
        for e in v.link_edges:
            vertex_to_add = e.other_vert(v)
            if vertex_to_add.index in unseen_v_idx:
                frontier_v.add(vertex_to_add)
                unseen_v_idx.remove(vertex_to_add.index)

    bm.free()
    return len(unseen_v_idx) > 0


def get_volume(obj):
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.transform(obj.matrix_world)
    bmesh.ops.triangulate(bm, faces=bm.faces)

    volume = 0
    for f in bm.faces:
        v1 = f.verts[0].co
        v2 = f.verts[1].co
        v3 = f.verts[2].co
        volume += v1.dot(v2.cross(v3)) / 6

    bm.free()
    return volume


def is_object_good(obj, min_aspect, min_volume_ratio):
    bbox_volume = obj.dimensions[0] * obj.dimensions[1] * obj.dimensions[2]
    if bbox_volume <= 0:
        return False
    aspect = min(obj.dimensions) / max(obj.dimensions)
    volume = get_volume(obj)
    volume_ratio = volume / bbox_volume
    return aspect > min_aspect and volume_ratio > min_volume_ratio


def clean_scene():
    ops.object.select_all(action='SELECT')
    ops.object.delete()
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=False)


def copy_to_meshes(data_dir, meshes_dir, n_meshes, rng, min_aspect, min_volume_ratio):
    meshes_dir = Path(meshes_dir)
    meshes_dir.mkdir(exist_ok=True, parents=True)
    raw_meshes = glob.glob(f'{data_dir}/**/*.obj', recursive=True)
    meshes_to_copy = []
    i = 0
    for m in raw_meshes:
        if i >= n_meshes:
            break
        ops.object.select_all(action='DESELECT')

        ops.wm.obj_import(filepath=m, forward_axis='Y', up_axis='Z')
        obj = bpy.context.selected_objects[0]

        if not is_manifold():
            clean_scene()
            continue

        if has_multiple_islands(obj):
            bpy.ops.object.editmode_toggle()
            bpy.ops.mesh.separate(type='LOOSE')
            bpy.ops.object.editmode_toggle()
            ops.object.select_all(action='SELECT')
            for p in bpy.context.selected_objects:
                if is_object_good(p, min_aspect, min_volume_ratio):
                    ops.object.select_all(action='DESELECT')
                    p.select_set(True)
                    ops.wm.obj_export(filepath=f'{meshes_dir}/{Path(m).name}',
                                      forward_axis='Y',
                                      up_axis='Z',
                                      export_materials=False,
                                      export_selected_objects=True)
                    i += 1
                    break
        elif is_object_good(obj, min_aspect, min_volume_ratio):
            meshes_to_copy.append(m)
            i += 1

        clean_scene()

    for m in track(meshes_to_copy):
        name = Path(m).name
        shutil.copyfile(m, meshes_dir / name)
