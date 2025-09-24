from pathlib import Path
import numpy as np
import torch
from common.inference import create_case_plot_dir, build_arg_parser, create_plots_root, predict
from dataset.data_parser import parse_model_type
from dataset.foam_data import FoamData
from dataset.foam_dataset import FoamDataset
from models.pi_gano.pi_gano import PiGano
from models.pi_gano.pi_gano_pp import PiGanoPp
from models.pi_gano.pi_gano_pp_full import PiGanoPpFull
from visualization.visualization_3d import plot_fields, plot_streamlines, plot_houses


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
    u_scaler = data.normalizers['U'].to()
    p_scaler = data.normalizers['p'].to()
    d_scaler = data.normalizers['d'].to()
    f_scaler = data.normalizers['f'].to()

    raw_points = points_scaler.inverse_transform(target['C']).numpy()

    d = torch.max(d_scaler.inverse_transform(target['d'])).item()
    f = torch.max(f_scaler.inverse_transform(target['f'])).item()
    inlet_ux = torch.max(u_scaler[0].inverse_transform(target['Ux-inlet']))

    plot_fields(f'Predicted D={d:.3f} F={f:.3f} Inlet={inlet_ux:.3f}',
                raw_points,
                u_scaler.inverse_transform(predicted['U']).numpy(),
                p_scaler.inverse_transform(predicted['p']).numpy(),
                target['cellToRegion'].numpy(),
                save_path=case_plot_path)
    plot_streamlines('Predicted streamlines',
                     case_path,
                     raw_points,
                     u_scaler.inverse_transform(predicted['U']).numpy(),
                     save_path=case_plot_path)

    plot_fields(f'Ground truth D={d:.3f} F={f:.3f} Inlet={inlet_ux:.3f}',
                raw_points,
                u_scaler.inverse_transform(target['U']).numpy(),
                p_scaler.inverse_transform(target['p']).numpy(),
                target['cellToRegion'].numpy(),
                save_path=case_plot_path)
    plot_streamlines('True streamlines',
                     case_path,
                     raw_points,
                     u_scaler.inverse_transform(target['U'].numpy()),
                     save_path=case_plot_path)

    u_error = (u_scaler.inverse_transform(predicted['U']) - u_scaler.inverse_transform(target['U'])).numpy()
    p_error = p_scaler.inverse_transform(predicted['p']) - p_scaler.inverse_transform(target['p']).numpy()
    plot_fields(f'Absolute error D={d:.3f} F={f:.3f} Inlet={inlet_ux:.3f}',
                raw_points,
                np.abs(u_error),
                np.abs(p_error),
                target['cellToRegion'].numpy(),
                save_path=case_plot_path)
    plot_streamlines('Error streamlines',
                     case_path,
                     raw_points,
                     np.abs(u_error),
                     save_path=case_plot_path)

    solid_points = points_scaler.inverse_transform(target['solid']['C']).numpy()
    solid_u_error = u_scaler.inverse_transform(predicted['solid']['U']) - u_scaler.inverse_transform(
        target['solid']['U'])
    solid_p_error = p_scaler.inverse_transform(predicted['solid']['p']) - p_scaler.inverse_transform(
        target['solid']['p'])
    plot_houses('House',
                solid_points,
                np.abs(solid_u_error.numpy()),
                np.abs(solid_p_error.numpy()),
                case_path / 'constant/triSurface/solid.obj',
                save_path=case_plot_path)


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    rng = np.random.default_rng(8421)
    plots_path = create_plots_root(args)
    model = get_model(args.checkpoint)
    val_data = FoamDataset(args.data_dir, args.n_internal, args.n_boundary, args.n_observations, rng, args.meta_dir)
    predict(args, model, val_data, sample_process_fn)
