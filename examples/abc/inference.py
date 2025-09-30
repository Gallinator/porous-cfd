from pathlib import Path
import numpy as np
import torch
from common.inference import create_case_plot_dir, build_arg_parser, create_plots_root, predict
from dataset.data_parser import parse_model_type
from dataset.foam_data import FoamData
from dataset.foam_dataset import FoamDataset
from models.pipn.pipn_foam import PipnFoam, PipnFoamPp, PipnFoamPpMrg, PipnFoamPpFull
from visualization.visualization_3d import plot_fields, plot_streamlines


def get_model(checkpoint):
    model_type = parse_model_type(checkpoint)
    match model_type:
        case 'pipn':
            return PipnFoam.load_from_checkpoint(checkpoint)
        case 'pipn-pp':
            return PipnFoamPp.load_from_checkpoint(checkpoint)
        case 'pipn-pp-mrg':
            return PipnFoamPpMrg.load_from_checkpoint(checkpoint)
        case 'pipn-pp-full':
            return PipnFoamPpFull.load_from_checkpoint(checkpoint)
        case _:
            raise NotImplementedError


def sample_process_fn(data: FoamDataset, target: FoamData, predicted: FoamData, case_path: Path, plots_path: Path):
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
    additional_meshes = {'mesh': 'oldlace'}

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
                     additional_meshes,
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
                     additional_meshes,
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
                     additional_meshes,
                     save_path=case_plot_path)


def inference():
    args = build_arg_parser().parse_args()
    rng = np.random.default_rng(8421)
    model = get_model(args.checkpoint)
    val_data = FoamDataset(args.data_dir, args.n_internal, args.n_boundary, args.n_observations, rng, args.meta_dir)
    predict(args, model, val_data, sample_process_fn)


if __name__ == '__main__':
    inference()
