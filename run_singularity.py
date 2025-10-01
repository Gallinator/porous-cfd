import importlib
import os

if __name__ == '__main__':
    example_dir = os.environ['EXAMPLE']
    cmd = os.environ['RUNCMD']
    example_pkg = example_dir.replace('/', '.')
    train_script = importlib.import_module(f'{example_pkg}.{cmd}')
    os.chdir(example_dir)
    train_script.run()
