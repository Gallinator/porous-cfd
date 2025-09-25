import glob
import subprocess
from abc import abstractmethod
from pathlib import Path
from rich.progress import track

from datagen.data_generator import DataGeneratorBase
from datagen.momentum_error import write_momentum_error


class Generator3DBase(DataGeneratorBase):
    def create_case_template_dirs(self):
        (self.case_template_dir / 'constant/triSurface').mkdir(parents=True, exist_ok=True)

    def generate_data(self, split_dir: Path):
        for case in track(glob.glob(f"{split_dir}/*"), description="Running cases"):
            process = subprocess.Popen(self.openfoam_bin, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                       stdout=subprocess.DEVNULL, text=True)
            process.communicate(f"{case}/Run")
            process.wait()
            if process.returncode != 0:
                self.raise_with_log_text(f'{case}', 'Failed to run ')

            write_momentum_error(case)

    @abstractmethod
    def generate_transformed_meshes(self, meshes_dir: Path, dest_dir: Path, rng):
        pass

    @abstractmethod
    def generate_openfoam_cases(self, meshes_dir, dest_dir, case_config_dir, rng):
        pass
