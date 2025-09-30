import importlib
import os

if __name__ == '__main__':
    example_dir = os.environ['EXAMPLE']
    example_dir = example_dir.replace('/', '.')
    train_script = importlib.import_module(f'{example_dir}.evaluate')
    train_script.evaluate_model()
