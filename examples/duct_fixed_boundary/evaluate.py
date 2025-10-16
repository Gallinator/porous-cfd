from pathlib import Path
from typing import Any
import numpy as np
from numpy.random import default_rng
from common.evaluation import build_arg_parser, evaluate, get_normalized_signed_distance, get_mean_max_error_distance, \
    get_pressure_drop
from dataset.data_parser import parse_model_type
from dataset.foam_data import FoamData
from dataset.foam_dataset import FoamDataset
from models.pipn.pipn_foam import PipnFoam, PipnFoamPpMrg, PipnFoamPpFull, PipnFoamPp
from visualization.common import plot_errors, plot_multi_bar


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
    p_scaler = data.normalizers['p'].to()
    tgt_drop = get_pressure_drop(p_scaler.inverse_transform(target['inlet']['p']),
                                 p_scaler.inverse_transform(target['outlet']['p']))
    pred_drop = get_pressure_drop(p_scaler.inverse_transform(predicted['inlet']['p']),
                                  p_scaler.inverse_transform(predicted['outlet']['p']))

    return {'Predicted drop': pred_drop.item(),
            'Target drop': tgt_drop.item()}


def postprocess_fn(data: FoamDataset, results: dict[str, Any], plots_path: Path):
    mean_tgt_drop = np.mean(results['Predicted drop'])
    mean_pred_drop = np.mean(results['Target drop'])
    plot_multi_bar('Pressure drop', {'Predicted': [mean_pred_drop], 'True': [mean_tgt_drop]},
                   ['$p$'], plots_path)
    print(f'Pressure drop error: {abs(mean_pred_drop - mean_tgt_drop)}')


def run():
    args = build_arg_parser().parse_args()

    model = get_model(args.checkpoint)

    rng = default_rng(8421)
    data = FoamDataset(args.data_dir, args.n_internal, args.n_boundary, args.n_observations, rng, args.meta_dir,
                       extra_fields=['momentError', 'div(phi)'])

    evaluate(args, model, data, True, sample_process, postprocess_fn)


if __name__ == '__main__':
    run()
