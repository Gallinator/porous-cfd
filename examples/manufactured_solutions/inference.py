from pathlib import Path

from matplotlib import pyplot as plt

import numpy as np

from common.inference import create_case_plot_dir, build_arg_parser, create_plots_root, predict
from dataset.foam_data import FoamData
from dataset.foam_dataset import FoamDataset
from models.pipn_baseline import PipnPorous
from visualization.visualization_2d import plot_fields
from manufactured_dataset import ManufacturedDataset


def sample_process_fn(data: FoamDataset, target: FoamData, predicted: FoamData, case_path: Path):
    case_plot_path = create_case_plot_dir(plots_path, case_path.name)

    plt.interactive(case_plot_path is None)
    plot_fields(f'Predicted', target['C'],
                predicted['U'],
                predicted['p'],
                target['cellToRegion'].numpy(),
                save_path=case_plot_path)
    plot_fields(f'Ground truth', target['C'],
                target['U'],
                target['p'],
                target['cellToRegion'].numpy(),
                save_path=case_plot_path)

    plt.interactive(False)

    u_error = predicted['U'] - target['U']
    p_error = predicted['p'] - target['p']
    plot_fields(f'Absolute error',
                target['C'],
                np.abs(u_error),
                np.abs(p_error),
                target['cellToRegion'].numpy(),
                False,
                case_plot_path)


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    rng = np.random.default_rng(8421)
    plots_path = create_plots_root(args)
    model = PipnPorous.load_from_checkpoint(args.checkpoint)
    val_data = ManufacturedDataset(args.data_dir, args.n_internal, args.n_boundary, 50, 1, rng, args.meta_dir)
    predict(args, model, val_data, sample_process_fn)
