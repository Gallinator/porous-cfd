from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from common.inference import create_case_plot_dir, build_arg_parser, create_plots_root, predict
from dataset.data_parser import parse_model_type
from dataset.foam_data import FoamData
from dataset.foam_dataset import FoamDataset
from models.pipn.pipn_baseline import PipnManufactured, PipnManufacturedPorousPp
from visualization.visualization_2d import plot_fields
from manufactured_dataset import ManufacturedDataset


def get_model(checkpoint):
    model_type = parse_model_type(checkpoint)
    match model_type:
        case 'pipn':
            return PipnManufactured.load_from_checkpoint(checkpoint)
        case 'pipn-pp':
            return PipnManufacturedPorousPp.load_from_checkpoint(checkpoint)
        case _:
            raise NotImplementedError


def sample_process_fn(data: FoamDataset, target: FoamData, predicted: FoamData, case_path: Path, plot_path: Path):
    plt.interactive(plot_path is None)
    plot_fields(f'Predicted', target['C'],
                predicted['U'],
                predicted['p'],
                target['cellToRegion'].numpy(),
                save_path=plot_path)
    plot_fields(f'Ground truth', target['C'],
                target['U'],
                target['p'],
                target['cellToRegion'].numpy(),
                save_path=plot_path)

    plt.interactive(False)

    u_error = predicted['U'] - target['U']
    p_error = predicted['p'] - target['p']
    plot_fields(f'Absolute error',
                target['C'],
                np.abs(u_error),
                np.abs(p_error),
                target['cellToRegion'].numpy(),
                False,
                plot_path)


def run():
    args = build_arg_parser().parse_args()
    rng = np.random.default_rng(8421)
    model = get_model(args.checkpoint)
    val_data = ManufacturedDataset(args.data_dir, args.n_internal, args.n_boundary, 50, 1, rng, args.meta_dir)
    predict(args, model, val_data, sample_process_fn)


if __name__ == '__main__':
    run()
