from pathlib import Path

import torch
from matplotlib import pyplot as plt
import numpy as np
from common.inference import build_arg_parser, create_plots_root, predict, create_case_plot_dir
from dataset.data_parser import parse_model_type
from dataset.foam_data import FoamData
from dataset.foam_dataset import FoamDataset
from models.pi_gano.pi_gano import PiGano
from models.pi_gano.pi_gano_pp import PiGanoPp
from models.pi_gano.pi_gano_pp_full import PiGanoPpFull
from visualization.visualization_2d import plot_fields


def get_model(checkpoint):
    model_type = parse_model_type(checkpoint)
    match model_type:
        case 'pi-gano':
            return PiGano.load_from_checkpoint(checkpoint)
        case 'pi-gano-pp':
            return PiGanoPp.load_from_checkpoint(checkpoint)
        case 'pi-gano-pp-full':
            return PiGanoPpFull.load_from_checkpoint(checkpoint)
        case _:
            raise NotImplementedError


def sample_process_fn(data: FoamDataset, target: FoamData, predicted: FoamData, case_path: Path):
    case_plot_path = create_case_plot_dir(plots_path, case_path.name)
    points_scaler = data.normalizers['C'].to()
    u_scaler = val_data.normalizers['U'].to()
    p_scaler = val_data.normalizers['p'].to()
    d_scaler = val_data.normalizers['d'].to()
    f_scaler = val_data.normalizers['f'].to()

    d = torch.max(d_scaler.inverse_transform(target['d']))
    f = torch.max(f_scaler.inverse_transform(target['f']))
    inlet_u = torch.max(u_scaler.inverse_transform(target['U-inlet']), dim=0)[0]

    raw_points = points_scaler.inverse_transform(target['C'])

    plt.interactive(case_plot_path is None)
    plot_fields(f'Predicted D={d.item():.3f} F={f.item():.3f} Inlet=({inlet_u[0].item():.3f} {inlet_u[1].item():.3f})',
                raw_points,
                u_scaler.inverse_transform(predicted['U']),
                p_scaler.inverse_transform(predicted['p']),
                target['cellToRegion'].numpy(),
                save_path=case_plot_path)
    plot_fields(
        f'Ground truth D={d.item():.3f} F={f.item():.3f} Inlet=({inlet_u[0].item():.3f} {inlet_u[1].item():.3f})',
        raw_points,
        u_scaler.inverse_transform(target['U']),
        p_scaler.inverse_transform(target['p']),
        target['cellToRegion'].numpy(),
        save_path=case_plot_path)

    plt.interactive(False)

    u_error = u_scaler.inverse_transform(predicted['U']) - u_scaler.inverse_transform(target['U'])
    p_error = p_scaler.inverse_transform(predicted['p']) - p_scaler.inverse_transform(target['p'])
    plot_fields(
        f'Absolute error D={d.item():.3f} F={f.item():.3f} Inlet=({inlet_u[0].item():.3f} {inlet_u[1].item():.3f})',
        raw_points,
        np.abs(u_error),
        np.abs(p_error),
        target['cellToRegion'].numpy(),
        False,
        case_plot_path)


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    rng = np.random.default_rng(8421)
    plots_path = create_plots_root(args)
    model = get_model(args.checkpoint)
    val_data = FoamDataset(args.data_dir, args.n_internal, args.n_boundary, args.n_observations, rng, args.meta_dir)
    predict(args, model, val_data, sample_process_fn)
