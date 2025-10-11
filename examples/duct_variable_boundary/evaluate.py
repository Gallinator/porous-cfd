from pathlib import Path
from typing import Any
import numpy as np
import torch
from numpy.random import default_rng
from common.evaluation import build_arg_parser, evaluate, get_normalized_signed_distance, get_mean_max_error_distance, \
    extract_coef, extract_u_magnitude, extract_angle
from dataset.data_parser import parse_model_type
from dataset.foam_data import FoamData
from dataset.foam_dataset import FoamDataset
from models.pi_gano.pi_gano import PiGano
from models.pi_gano.pi_gano_pp import PiGanoPp
from models.pi_gano.pi_gano_pp_full import PiGanoPpFull
from visualization.common import plot_errors, plot_errors_vs_var, plot_errors_vs_multi_vars


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


def sample_process(data: FoamDataset, predicted: FoamData, target: FoamData, extras: FoamData) -> dict[str, Any]:
    data.normalizers['d'].to()
    d = extract_coef(target['d'], data.normalizers['d'])
    d = torch.round(d).to(torch.int64)
    data.normalizers['f'].to()
    f = extract_coef(target['f'], data.normalizers['f'])

    data.normalizers['U'].to()
    u_magnitude = extract_u_magnitude(target['inlet']['U-inlet'], data.normalizers['U'], 0.025)

    angle = extract_angle(target['inlet']['U'], data.normalizers['U'])

    return {'d': d, 'f': f, 'U inlet': u_magnitude, 'Angle': angle}


def postprocess_fn(data: FoamDataset, results: dict[str, Any], plots_path: Path):
    errors = np.concatenate([results['U error'], results['p error']], -1)

    per_case_mae = np.concatenate(np.mean(errors, axis=-2, keepdims=True))
    angles = torch.tensor(results['Angle']).flatten()
    mae_by_angle = [np.mean(per_case_mae[angles == a], axis=0, keepdims=True) for a in angles.unique()]
    mae_by_angle = np.concatenate(mae_by_angle)
    plot_errors_vs_var('MAE by inlet angle', mae_by_angle, angles.unique(), ['MAE', 'Angle'], plots_path)

    d, f = np.array(results['d']).flatten(), np.array(results['f']).flatten()
    u_inlet = np.array(results['U inlet']).flatten()
    plot_errors_vs_multi_vars('MAE heatmap', per_case_mae, d.astype(np.int64), u_inlet, ['D', 'U'], plots_path)


def run():
    args = build_arg_parser().parse_args()

    model = get_model(args.checkpoint)

    rng = default_rng(8421)
    data = FoamDataset(args.data_dir, args.n_internal, args.n_boundary, args.n_observations, rng, args.meta_dir,
                       extra_fields=['momentError', 'div(phi)'])

    evaluate(args, model, data, False, sample_process, postprocess_fn)


if __name__ == '__main__':
    run()
