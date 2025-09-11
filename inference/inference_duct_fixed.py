from pathlib import Path

from matplotlib import pyplot as plt

from common import create_plots_root, create_case_plot_dir
import numpy as np
from dataset.foam_data import FoamData
from dataset.foam_dataset import FoamDataset
from common import build_arg_parser, predict
from models.pipn_foam import PipnFoam
from visualization.visualization_2d import plot_fields


def sample_process_fn(data: FoamDataset, target: FoamData, predicted: FoamData, case_path: Path):
    case_plot_path = create_case_plot_dir(plots_path, case_path.name)

    points_scaler = data.normalizers['C'].to()
    u_scaler = data.normalizers['U'].to()
    p_scaler = data.normalizers['p'].to()

    raw_points = points_scaler.inverse_transform(target['C'])

    plt.interactive(case_plot_path is None)

    plot_fields('Predicted', raw_points, u_scaler.inverse_transform(predicted['U']),
                p_scaler.inverse_transform(predicted['p']), target['cellToRegion'], save_path=case_plot_path)
    plot_fields('Ground truth', raw_points, u_scaler.inverse_transform(target['U']),
                p_scaler.inverse_transform(target['p']), target['cellToRegion'], save_path=case_plot_path)

    plt.interactive(False)

    u_error = u_scaler.inverse_transform(predicted['U']) - u_scaler.inverse_transform(target['U'])
    p_error = p_scaler.inverse_transform(predicted['p']) - p_scaler.inverse_transform(target['p'])
    plot_fields('Absolute error', raw_points, np.abs(u_error), np.abs(p_error), target['cellToRegion'], False,
                save_path=case_plot_path)


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    rng = np.random.default_rng(8421)
    plots_path = create_plots_root(args)
    model = PipnFoam.load_from_checkpoint(args.checkpoint)
    val_data = FoamDataset(args.data_dir, args.n_internal, args.n_boundary, args.n_observations, rng, args.meta_dir)
    predict(args, model, val_data, sample_process_fn)
