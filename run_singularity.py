import importlib
import os

# This script allows to run experiments on Singularity containers. This is a hacky way of avoiding the setting of python environment variables which do not work on some clusters.
# The script works by using the EXAMPLE environment variable, which gives the experiment directory, and the RUNCMD variable which gives the python script to run together with its arguments.
# Then the example package is loaded and the working directory is set.
# The environment variables are set by running the sbatch.sh script.

if __name__ == '__main__':
    example_dir = os.environ['EXAMPLE']
    cmd = os.environ['RUNCMD']
    example_pkg = example_dir.replace('/', '.')
    train_script = importlib.import_module(f'{example_pkg}.{cmd}')
    os.chdir(example_dir)
    train_script.run()
