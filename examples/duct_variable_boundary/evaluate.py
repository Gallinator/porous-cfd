from pathlib import Path
import numpy as np
import torch
from numpy.random import default_rng
from scipy.stats._mstats_basic import trimmed_mean
from torch.nn.functional import l1_loss

from common.evaluation import save_mae_to_csv, build_arg_parser, evaluate
from dataset.data_parser import parse_model_type
from dataset.foam_data import FoamData
from dataset.foam_dataset import FoamDataset
from models.pi_gano.pi_gano import PiGano
from models.pi_gano.pi_gano_pp import PiGanoPp
from models.pi_gano.pi_gano_pp_full import PiGanoPpFull
from visualization.common import plot_data_dist, plot_residuals, plot_errors


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


def sample_process(data: FoamDataset, predicted: FoamData, target: FoamData, extras: FoamData) -> tuple:
    u_scaler = data.normalizers['U'].to()
    p_scaler = data.normalizers['p'].to()

    # Domain
    u_error = l1_loss(u_scaler.inverse_transform(predicted['U']),
                      u_scaler.inverse_transform(target['U']), reduction='none')
    p_error = l1_loss(p_scaler.inverse_transform(predicted['p']),
                      p_scaler.inverse_transform(target['p']), reduction='none')
    error = torch.cat([u_error, p_error], dim=-1)

    zones_ids = target['cellToRegion']

    return (error,
            extras['Momentum'],
            extras['div'],
            target['internal']['momentError'],
            target['internal']['div(phi)'],
            zones_ids)


def postprocess_fn(data: FoamDataset, results: tuple, plots_path: Path):
    errors, predicted_momentum, predicted_div, target_momentum, target_div, zones_ids = results

    errors = torch.cat(errors)
    zones_ids = torch.cat(zones_ids).flatten()
    plot_data_dist('Absolute error distribution',
                   errors[..., :model.dims],
                   errors[..., model.dims:],
                   save_path=plots_path)
    mae = torch.mean(errors, dim=0).tolist()
    plot_errors('Average relative error', mae, save_path=plots_path)

    fluid_mae = torch.mean(errors[zones_ids < 1, :], dim=0).tolist()
    porous_mae = torch.mean(errors[zones_ids > 0, :], dim=0).tolist()
    plot_errors('Porous region MAE', porous_mae, save_path=plots_path)
    plot_errors('Fluid region MAE', fluid_mae, save_path=plots_path)

    predicted_momentum, predicted_div = torch.cat(predicted_momentum), torch.cat(predicted_div)
    target_momentum, target_div = torch.cat(target_momentum), torch.cat(target_div)
    plot_data_dist('Absolute residuals',
                   np.abs(predicted_momentum),
                   np.abs(predicted_div),
                   save_path=plots_path)

    predicted_residuals = torch.cat([predicted_momentum, predicted_div], -1)
    target_residuals = torch.cat([target_momentum, target_div], -1)
    pred_res_avg = trimmed_mean(torch.abs(predicted_residuals), limits=[0, 0.05], axis=0)
    cfd_res_avg = trimmed_mean(torch.abs(target_residuals), limits=[0, 0.05], axis=0)
    plot_residuals(pred_res_avg, cfd_res_avg, trim=0.05, save_path=plots_path)

    if args.save_plots:
        errors_dict = {'Total': mae, 'Fluid': fluid_mae, 'Porous': porous_mae}
        save_mae_to_csv(errors_dict, ['Ux', 'Uy'][:model.dims] + ['p'], plots_path)


if __name__ == '__main__':
    args = build_arg_parser().parse_args()

    model = get_model(args.checkpoint)

    rng = default_rng(8421)
    data = FoamDataset(args.data_dir, args.n_internal, args.n_boundary, args.n_observations, rng, args.meta_dir,
                       extra_fields=['momentError', 'div(phi)'])

    evaluate(args, model, data, True, sample_process, postprocess_fn)
