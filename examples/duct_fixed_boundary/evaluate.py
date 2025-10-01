from pathlib import Path
from typing import Any
import numpy as np
from numpy.random import default_rng
from common.evaluation import build_arg_parser, evaluate, get_normalized_signed_distance, get_mean_max_error_distance
from dataset.data_parser import parse_model_type
from dataset.foam_data import FoamData
from dataset.foam_dataset import FoamDataset
from models.pipn.pipn_foam import PipnFoam, PipnFoamPpMrg, PipnFoamPpFull, PipnFoamPp
from visualization.common import plot_errors


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


def sample_process(data: FoamDataset, predicted: FoamData, target: FoamData, extras: FoamData) -> dict[str, Any]:
    c_scaler = data.normalizers['C'].to()
    all_points = c_scaler.inverse_transform(target['C'])
    interface_points = c_scaler.inverse_transform(target['interface']['C'])

    interface_dist = get_normalized_signed_distance(all_points, interface_points)
    return {'Interface distance': interface_dist}


def postprocess_fn(data: FoamDataset, results: dict[str, Any], plots_path: Path):
    errors = np.concatenate([results['U error'], results['p error']], -1)
    max_error_from_interface = get_mean_max_error_distance(errors, 0.8, results['Interface distance'])
    plot_errors('Errors mean normalized distance from interface', max_error_from_interface, save_path=plots_path)


def run():
    args = build_arg_parser().parse_args()

    model = get_model(args.checkpoint)

    rng = default_rng(8421)
    data = FoamDataset(args.data_dir, args.n_internal, args.n_boundary, args.n_observations, rng, args.meta_dir,
                       extra_fields=['momentError', 'div(phi)'])

    evaluate(args, model, data, True, sample_process, postprocess_fn)


if __name__ == '__main__':
    run()
