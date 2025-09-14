from pathlib import Path
import numpy as np
import torch
from numpy.random import default_rng
from scipy.stats._mstats_basic import trimmed_mean
from torch.nn.functional import l1_loss

from common.evaluation import save_mae_to_csv, build_arg_parser, evaluate
from dataset.foam_data import FoamData
from dataset.foam_dataset import FoamDataset
from models.pipn_foam import PipnFoam
from visualization.common import plot_data_dist, plot_residuals, plot_errors


def sample_process(data: FoamDataset, predicted: FoamData, target: FoamData, extras: FoamData) -> tuple:
    u_scaler = data.normalizers['U'].to(model.device)
    p_scaler = data.normalizers['p'].to(model.device)

    # Domain
    u_error = l1_loss(u_scaler.inverse_transform(predicted['U']),
                      u_scaler.inverse_transform(target['U']), reduction='none')
    p_error = l1_loss(p_scaler.inverse_transform(predicted['p']),
                      p_scaler.inverse_transform(target['p']), reduction='none')
    error = torch.cat([u_error, p_error], dim=-1)

    zones_ids = target['cellToRegion']

    # Equation residuals
    target_residuals = torch.cat([target['internal']['momentError'], target['internal']['div(phi)']], dim=-1)

    return error, extras['Momentum'], extras['div'], target_residuals, zones_ids


def postprocess_fn(data: FoamDataset, results: tuple, plots_path: Path):
    errors, predicted_momentum, predicted_div, target_residuals, zones_ids = results

    errors = np.concatenate(errors)
    zones_ids = np.array(zones_ids).flatten()
    plot_data_dist('Absolute error distribution',
                   errors[..., :data.n_dims],
                   errors[..., data.n_dims:],
                   save_path=plots_path)
    mae = np.average(errors, axis=0).tolist()
    plot_errors('Average relative error', mae, save_path=plots_path)

    fluid_mae = np.average(errors[zones_ids < 1, :], axis=0).tolist()
    porous_mae = np.average(errors[zones_ids > 0, :], axis=0).tolist()
    plot_errors('Porous region MAE', porous_mae, save_path=plots_path)
    plot_errors('Fluid region MAE', fluid_mae, save_path=plots_path)

    predicted_div = np.concatenate(predicted_div)
    predicted_momentum = np.concatenate(predicted_momentum)
    plot_data_dist('Absolute residuals',
                   np.abs(predicted_momentum),
                   np.abs(predicted_div),
                   save_path=plots_path)

    target_residuals = np.concatenate(target_residuals)
    predicted_residuals = np.concatenate([predicted_momentum, predicted_div], axis=-1)
    pred_res_avg = trimmed_mean(np.abs(predicted_residuals), limits=[0, 0.05], axis=0)
    cfd_res_avg = trimmed_mean(np.abs(target_residuals), limits=[0, 0.05], axis=0)
    plot_residuals(pred_res_avg, cfd_res_avg, trim=0.05, save_path=plots_path)

    if args.save_plots:
        errors_dict = {'Total': mae, 'Fluid': fluid_mae, 'Porous': porous_mae}
        save_mae_to_csv(errors_dict, ['Ux', 'Uy'] + ['p'], plots_path)


if __name__ == '__main__':
    args = build_arg_parser().parse_args()

    model = PipnFoam.load_from_checkpoint(args.checkpoint)

    rng = default_rng(8421)
    data = FoamDataset(args.data_dir, args.n_internal, args.n_boundary, args.n_observations, rng, args.meta_dir,
                       extra_fields=['momentError', 'div(phi)'])

    evaluate(args, model, data, True, sample_process, postprocess_fn)
