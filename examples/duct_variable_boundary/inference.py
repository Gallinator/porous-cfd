from pathlib import Path

from matplotlib import pyplot as plt

import numpy as np

from common.inference import build_arg_parser, create_plots_root, predict, create_case_plot_dir
from dataset.foam_data import FoamData
from dataset.foam_dataset import FoamDataset
from models.pipn_foam import PipnFoam
from visualization.visualization_2d import plot_fields


def sample_process_fn(data: FoamDataset, predicted: FoamData, target: FoamData, case_path: Path):
    case_plot_path = create_case_plot_dir(plots_path, case_path.name)
    points_scaler = data.normalizers['C']
    u_scaler = val_data.normalizers['U']
    p_scaler = val_data.normalizers['p']
    d_scaler = val_data.normalizers['d']
    f_scaler = val_data.normalizers['f']

    d = np.max(d_scaler.inverse_transform(target['d']))
    f = np.max(f_scaler.inverse_transform(target['f']))
    inlet_u = np.max(u_scaler.inverse_transform(target['U-inlet']), axis=0)

    raw_points = points_scaler.inverse_transform(target['C'])

    plt.interactive(case_plot_path is None)
    plot_fields(f'Predicted D={d:.3f} F={f:.3f} Inlet=({inlet_u[0]:.3f} {inlet_u[1]:.3f})', raw_points,
                u_scaler.inverse_transform(predicted['U'][0]),
                p_scaler.inverse_transform(predicted['p'][0]), target['cellToRegion'], save_path=case_plot_path)
    plot_fields(f'Ground truth D={d:.3f} F={f:.3f} Inlet=({inlet_u[0]:.3f} {inlet_u[1]:.3f})', raw_points,
                u_scaler.inverse_transform(target['U']),
                p_scaler.inverse_transform(target['p']), target['cellToRegion'], save_path=case_plot_path)

    plt.interactive(False)

    u_error = u_scaler.inverse_transform(predicted['U'][0]) - u_scaler.inverse_transform(target['U'])
    p_error = p_scaler.inverse_transform(predicted['p'][0]) - p_scaler.inverse_transform(target['p'])
    plot_fields(f'Absolute error D={d:.3f} F={f:.3f} Inlet=({inlet_u[0]:.3f} {inlet_u[1]:.3f})', raw_points,
                np.abs(u_error),
                np.abs(p_error),
                target['cellToRegion'], False,
                case_plot_path)


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    rng = np.random.default_rng(8421)
    plots_path = create_plots_root(args)
    model = PipnFoam.load_from_checkpoint(args.checkpoint)
    val_data = FoamDataset(args.data_dir, args.n_internal, args.n_boundary, args.n_observations, rng, args.meta_dir)
    predict(args, model, val_data, sample_process_fn)
