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


def move_to_meshes(data_dir, meshes_dir, n_meshes, rng):
    meshes_dir = Path(meshes_dir)
    meshes_dir.mkdir(exist_ok=True, parents=True)
    files = glob.glob(f'{data_dir}/**/*.obj', recursive=True)
    meshes_to_move = rng.sample(files, n_meshes)
    for m in meshes_to_move:
        name = Path(m).name
        shutil.move(m, meshes_dir / name)
