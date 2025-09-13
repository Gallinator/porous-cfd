from pathlib import Path
import numpy as np
import torch
from numpy.random import default_rng
from scipy.stats._mstats_basic import trimmed_mean
from torch import Tensor
from torch.nn.functional import l1_loss

from common.evaluation import save_mae_to_csv, build_arg_parser, evaluate
from dataset.foam_data import FoamData
from dataset.foam_dataset import FoamDataset
from models.pipn_baseline import PipnPorous
from visualization.common import plot_data_dist, plot_residuals, plot_errors
from manufactured_dataset import ManufacturedDataset


def sample_process(data: FoamDataset, predicted: FoamData, target: FoamData, extras: Tensor) -> tuple:
    # Domain
    u_error = l1_loss(predicted['U'], target['U'], reduction='none')
    p_error = l1_loss(predicted['p'], target['p'], reduction='none')
    error = torch.cat([u_error, p_error], dim=-1).numpy(force=True)

    zones_ids = target['cellToRegion'].numpy(force=True)

    # Equation residuals
    predicted_residuals = extras.numpy(force=True)

    return error, predicted_residuals, zones_ids


def postprocess_fn(data: FoamDataset, results: tuple, plots_path: Path):
    errors, predicted_residuals, zones_ids = results

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

    pred_residuals = np.concatenate(predicted_residuals)
    cfd_residuals = np.zeros_like(pred_residuals)
    plot_data_dist('Absolute residuals',
                   np.abs(pred_residuals[..., :data.n_dims]),
                   np.abs(pred_residuals[..., data.n_dims:]),
                   save_path=plots_path)

    pred_res_avg = trimmed_mean(np.abs(pred_residuals), limits=[0, 0.05], axis=0)
    cfd_res_avg = trimmed_mean(np.abs(cfd_residuals), limits=[0, 0.05], axis=0)
    plot_residuals(pred_res_avg, cfd_res_avg, trim=0.05, save_path=plots_path)

    if args.save_plots:
        errors_dict = {'Total': mae, 'Fluid': fluid_mae, 'Porous': porous_mae}
        save_mae_to_csv(errors_dict, ['Ux', 'Uy'][:model.dims] + ['p'], plots_path)


if __name__ == '__main__':
    args = build_arg_parser().parse_args()

    model = PipnPorous.load_from_checkpoint(args.checkpoint)

    rng = default_rng(8421)
    data = ManufacturedDataset(args.data_dir, args.n_internal, args.n_boundary, 50, 1, rng, args.meta_dir)

    evaluate(args, model, data, True, sample_process, postprocess_fn)
