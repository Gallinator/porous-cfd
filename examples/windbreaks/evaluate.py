from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.random import default_rng
from torch import Tensor
from torch.nn.functional import l1_loss
from common.evaluation import build_arg_parser, evaluate, get_normalized_signed_distance, get_mean_max_error_distance, \
    extract_coef, extract_u_magnitude
from dataset.data_parser import parse_model_type
from dataset.foam_data import FoamData
from dataset.foam_dataset import FoamDataset
from models.pi_gano.pi_gano import PiGano
from models.pi_gano.pi_gano_pp import PiGanoPp
from models.pi_gano.pi_gano_pp_full import PiGanoPpFull
from visualization.common import plot_data_dist, plot_errors, plot_errors_vs_multi_vars


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


def sample_process(data: FoamDataset, predicted: FoamData, target: FoamData, extras: Tensor) -> dict[str, Any]:
    u_scaler = data.normalizers['U'].to()
    p_scaler = data.normalizers['p'].to()

    solid_u_error = l1_loss(u_scaler.inverse_transform(predicted['solid']['U']),
                            u_scaler.inverse_transform(target['solid']['U']), reduction='none')
    solid_p_error = l1_loss(p_scaler.inverse_transform(predicted['solid']['p']),
                            p_scaler.inverse_transform(target['solid']['p']), reduction='none')

    c_scaler = data.normalizers['C'].to()
    all_points = c_scaler.inverse_transform(target['C'])
    interface_points = c_scaler.inverse_transform(target['interface']['C'])
    interface_dist = get_normalized_signed_distance(all_points, interface_points)

    data.normalizers['d'].to()
    d = extract_coef(target['d'], data.normalizers['d'])
    d = torch.round(d).to(torch.int64)
    data.normalizers['f'].to()
    f = extract_coef(target['f'], data.normalizers['f'])

    data.normalizers['U'].to()
    u_magnitude = extract_u_magnitude(target['inlet']['Ux-inlet'], data.normalizers['U'], 1e-6)

    return {'Interface distance': interface_dist,
            'U error solid': solid_u_error,
            'p error solid': solid_p_error,
            'd': d,
            'f': f,
            'U inlet': u_magnitude}


def postprocess_fn(data: FoamDataset, results: dict[str, Any], plots_path: Path):
    errors = np.concatenate([results['U error'], results['p error']], -1)
    max_error_from_interface = get_mean_max_error_distance(errors, 0.8, results['Interface distance'])
    plot_errors('Errors mean normalized distance from interface', max_error_from_interface, save_path=plots_path)

    u_solid_error = np.concatenate(results['U error solid'])
    p_solid_error = np.concatenate(results['p error solid'])
    plot_data_dist(f'Solid Absolute error distribution',
                   u_solid_error,
                   p_solid_error,
                   save_path=plots_path)
    solid_errors = np.concatenate([u_solid_error, p_solid_error], axis=-1)
    solid_mae = np.average(solid_errors, axis=0).tolist()
    plot_errors(f'Solid Average relative error', solid_mae, save_path=plots_path)

    per_case_mae = np.concatenate(np.mean(errors, axis=-2, keepdims=True))
    d, f = np.array(results['d']).flatten(), np.array(results['f']).flatten()
    u_inlet = np.array(results['U inlet']).flatten()
    plot_errors_vs_multi_vars('MAE heatmap', per_case_mae, d.astype(np.int64), u_inlet, ['D', 'U'], plots_path)


def run():
    args = build_arg_parser().parse_args()

    model = get_model(args.checkpoint)

    rng = default_rng(8421)
    data = FoamDataset(args.data_dir, args.n_internal, args.n_boundary, args.n_observations, rng, args.meta_dir)

    evaluate(args, model, data, True, sample_process, postprocess_fn)


if __name__ == '__main__':
    run()
