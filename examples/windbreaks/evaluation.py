from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.random import default_rng
from torch import Tensor
from torch.nn.functional import l1_loss
from common.evaluation import build_arg_parser, evaluate, get_normalized_signed_distance, get_mean_max_error_distance
from dataset.data_parser import parse_model_type
from dataset.foam_data import FoamData
from dataset.foam_dataset import FoamDataset
from models.pi_gano.pi_gano import PiGano
from models.pi_gano.pi_gano_pp import PiGanoPp
from models.pi_gano.pi_gano_pp_full import PiGanoPpFull
from visualization.common import plot_data_dist, plot_errors


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
    u_scaler = data.normalizers['U'].to(model.device)
    p_scaler = data.normalizers['p'].to(model.device)

    solid_u_error = l1_loss(u_scaler.inverse_transform(predicted['solid']['U']),
                            u_scaler.inverse_transform(target['solid']['U']), reduction='none')
    solid_p_error = l1_loss(p_scaler.inverse_transform(predicted['solid']['p']),
                            p_scaler.inverse_transform(target['solid']['p']), reduction='none')

    c_scaler = data.normalizers['C'].to()
    all_points = c_scaler.inverse_transform(target['C'])
    interface_points = c_scaler.inverse_transform(target['interface']['C'])
    interface_dist = get_normalized_signed_distance(all_points, interface_points)

    return {'Interface distance': interface_dist,
            'U error solid': solid_u_error,
            'p error solid': solid_p_error}


def postprocess_fn(data: FoamDataset, results: dict[str, Any], plots_path: Path):
    errors = torch.cat([results['U error'], results['p error']], -1)
    max_error_from_interface = get_mean_max_error_distance(errors, 0.8, results['Interface distance'])
    plot_errors('Errors mean normalized distance from interface', max_error_from_interface, save_path=plots_path)

    u_solid_error, p_solid_error = results['U solid error'].flatten(0, 1)
    p_solid_error = results['p solid error'].flatten(0, 1)
    plot_data_dist(f'{args.subdomain.capitalize()} Absolute error distribution',
                   u_solid_error,
                   p_solid_error,
                   save_path=plots_path)
    solid_errors = torch.cat([u_solid_error, p_solid_error], dim=-1)
    solid_mae = np.average(solid_errors, axis=0).tolist()
    plot_errors(f'{args.subdomain.capitalize()} Average relative error', solid_mae, save_path=plots_path)


if __name__ == '__main__':
    args = build_arg_parser().parse_args()

    model = get_model(args.checkpoint)

    rng = default_rng(8421)
    data = FoamDataset(args.data_dir, args.n_internal, args.n_boundary, args.n_observations, rng, args.meta_dir)

    evaluate(args, model, data, True, sample_process, postprocess_fn)
