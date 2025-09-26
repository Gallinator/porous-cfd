from pathlib import Path
from typing import Any
import torch
from functorch.dim import Tensor
from numpy.random import default_rng
from common.evaluation import build_arg_parser, evaluate, get_normalized_signed_distance, get_mean_max_error_distance
from dataset.data_parser import parse_model_type
from dataset.foam_data import FoamData
from dataset.foam_dataset import FoamDataset, StandardScaler, Normalizer
from models.pi_gano.pi_gano import PiGano
from models.pi_gano.pi_gano_pp import PiGanoPp
from models.pi_gano.pi_gano_pp_full import PiGanoPpFull
from visualization.common import plot_errors


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


def extract_coef(coef: Tensor, scaler: StandardScaler | Normalizer):
    coef = scaler.inverse_transform(coef)
    return torch.max(coef)[0]


def extract_u_magnitude(u: Tensor, scaler: StandardScaler):
    u_mag = scaler.inverse_transform(u)
    u_mag = torch.norm(u_mag, dim=-1)
    u_mag = torch.max(u_mag, dim=-2, keepdim=True)[0]
    # Assume values spaced by 0.025
    return round(u_mag * 1000 / 25) * 25 / 1000


def sample_process(data: FoamDataset, predicted: FoamData, target: FoamData, extras: FoamData) -> dict[str, Any]:
    c_scaler = data.normalizers['C'].to()
    all_points = c_scaler.inverse_transform(target['C'])
    interface_points = c_scaler.inverse_transform(target['interface']['C'])
    interface_dist = get_normalized_signed_distance(all_points, interface_points)

    data.normalizers['d'].to()
    d = extract_coef(target['d'], data.normalizers['d'])
    data.normalizers['f'].to()
    f = extract_coef(target['f'], data.normalizers['f'])

    data.normalizers['U'].to()
    u_magnitude = extract_u_magnitude(target['inlet']['Uinlet'], data.normalizers['U'])

    return {'Interface distance': interface_dist, 'd': d, 'f': f, 'U inlet': u_magnitude}


def postprocess_fn(data: FoamDataset, results: dict[str, Any], plots_path: Path):
    errors = torch.cat([results['U error'], results['p error']], -1)
    max_error_from_interface = get_mean_max_error_distance(errors, 0.8, results['Interface distance'])
    plot_errors('Errors mean normalized distance from interface', max_error_from_interface, save_path=plots_path)


if __name__ == '__main__':
    args = build_arg_parser().parse_args()

    model = get_model(args.checkpoint)

    rng = default_rng(8421)
    data = FoamDataset(args.data_dir, args.n_internal, args.n_boundary, args.n_observations, rng, args.meta_dir,
                       extra_fields=['momentError', 'div(phi)'])

    evaluate(args, model, data, True, sample_process, postprocess_fn)
