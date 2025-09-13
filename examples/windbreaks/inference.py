from pathlib import Path
import numpy as np
import torch

from common.inference import create_case_plot_dir, build_arg_parser, create_plots_root, predict
from dataset.foam_data import FoamData
from dataset.foam_dataset import FoamDataset
from models.pi_gano import PiGano
from visualization.visualization_3d import plot_fields, plot_streamlines, plot_houses


def sample_process_fn(data: FoamDataset, predicted: FoamData, tgt: FoamData, case_path: Path):
    case_plot_path = create_case_plot_dir(plots_path, case_path.name)

    points_scaler = data.normalizers['C'].to()
    u_scaler = data.normalizers['U'].to()
    p_scaler = data.normalizers['p'].to()
    d_scaler = data.normalizers['d'].to()
    f_scaler = data.normalizers['f'].to()

    raw_points = points_scaler.inverse_transform(tgt['C']).numpy(force=True)

    d = torch.max(d_scaler.inverse_transform(tgt['d']))
    f = torch.max(f_scaler.inverse_transform(tgt['f']))
    inlet_ux = torch.max(u_scaler[0].inverse_transform(tgt['Ux-inlet']))

    plot_fields(f'Predicted D={d:.3f} F={f:.3f} Inlet={inlet_ux:.3f}',
                raw_points,
                u_scaler.inverse_transform(predicted['U']).numpy(force=True),
                p_scaler.inverse_transform(predicted['p']).numpy(force=True),
                tgt['cellToRegion'].numpy(force=True),
                save_path=case_plot_path)
    plot_streamlines('Predicted streamlines',
                     case_path,
                     raw_points,
                     u_scaler.inverse_transform(predicted['U']).numpy(force=True),
                     save_path=case_plot_path)

    plot_fields(f'Ground truth D={d:.3f} F={f:.3f} Inlet={inlet_ux:.3f}',
                raw_points,
                u_scaler.inverse_transform(tgt['U']).numpy(force=True),
                p_scaler.inverse_transform(tgt['p']).numpy(force=True),
                tgt['cellToRegion'].numpy(force=True),
                save_path=case_plot_path)
    plot_streamlines('True streamlines',
                     case_path,
                     raw_points,
                     u_scaler.inverse_transform(tgt['U'].numpy(force=True)),
                     save_path=case_plot_path)

    u_error = (u_scaler.inverse_transform(predicted['U']) - u_scaler.inverse_transform(tgt['U'])).numpy(force=True)
    p_error = p_scaler.inverse_transform(predicted['p']) - p_scaler.inverse_transform(tgt['p']).numpy(force=True)
    plot_fields(f'Absolute error D={d:.3f} F={f:.3f} Inlet={inlet_ux:.3f}',
                raw_points,
                np.abs(u_error),
                np.abs(p_error),
                tgt['cellToRegion'].numpy(force=True),
                save_path=case_plot_path)
    plot_streamlines('Error streamlines',
                     case_path,
                     raw_points,
                     np.abs(u_error),
                     save_path=case_plot_path)

    solid_points = points_scaler.inverse_transform(tgt['solid']['C']).numpy(force=True)
    solid_u_error = u_scaler.inverse_transform(predicted['solid']['U']) - u_scaler.inverse_transform(
        tgt['solid']['U']).numpy(force=True)
    solid_p_error = p_scaler.inverse_transform(predicted['solid']['p']) - p_scaler.inverse_transform(
        tgt['solid']['p']).numpy(force=True)
    plot_houses('House',
                solid_points,
                np.abs(solid_u_error),
                np.abs(solid_p_error),
                case_path / 'constant/triSurface/solid.obj',
                save_path=case_plot_path)


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    rng = np.random.default_rng(8421)
    plots_path = create_plots_root(args)
    model = PiGano.load_from_checkpoint(args.checkpoint)
    val_data = FoamDataset(args.data_dir, args.n_internal, args.n_boundary, args.n_observations, rng, args.meta_dir)
    predict(args, model, val_data, sample_process_fn)
