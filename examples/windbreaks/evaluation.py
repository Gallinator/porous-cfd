from pathlib import Path
import numpy as np
import torch
from numpy.random import default_rng
from scipy.stats._mstats_basic import trimmed_mean
from torch import Tensor
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


def sample_process(data: FoamDataset, predicted: FoamData, target: FoamData, extras: Tensor) -> tuple:
    u_scaler = data.normalizers['U'].to(model.device)
    p_scaler = data.normalizers['p'].to(model.device)

    u_error = l1_loss(u_scaler.inverse_transform(predicted['U']),
                      u_scaler.inverse_transform(target['U']), reduction='none')
    p_error = l1_loss(p_scaler.inverse_transform(predicted['p']),
                      p_scaler.inverse_transform(target['p']), reduction='none')
    error = torch.cat([u_error, p_error], dim=-1)

    zones_ids = target['cellToRegion'].numpy(force=True)

    solid_u_error = l1_loss(u_scaler.inverse_transform(predicted['solid']['U']),
                            u_scaler.inverse_transform(target['solid']['U']), reduction='none')
    solid_p_error = l1_loss(p_scaler.inverse_transform(predicted['solid']['p']),
                            p_scaler.inverse_transform(target['solid']['p']), reduction='none')
    solid_error = torch.cat([solid_u_error, solid_p_error], dim=-1)

    # Equation residuals
    predicted_residuals = extras.numpy(force=True)
    target_residuals = torch.cat([target['internal']['momentError'], target['internal']['div(phi)']], dim=-1)

    return error, solid_error, predicted_residuals, target_residuals, zones_ids


def postprocess_fn(data: FoamDataset, results: tuple, plots_path: Path):
    errors, solid_errors, predicted_residuals, target_residuals, zones_ids = results

    errors, zones_ids = np.concatenate(errors)
    zones_ids = np.array(zones_ids).flatten()
    plot_data_dist('Absolute error distribution',
                   errors[..., :model.dims],
                   errors[..., model.dims:],
                   save_path=plots_path)
    mae = np.average(errors, axis=0).tolist()
    plot_errors('Average relative error', mae, save_path=plots_path)

    solid_errors = np.concatenate(solid_errors)
    plot_data_dist(f'{args.subdomain.capitalize()} Absolute error distribution',
                   solid_errors[..., :model.dims],
                   solid_errors[..., model.dims:],
                   save_path=plots_path)
    additional_mae = np.average(solid_errors, axis=0).tolist()
    plot_errors(f'{args.subdomain.capitalize()} Average relative error', additional_mae, save_path=plots_path)

    fluid_mae = np.average(errors[zones_ids < 1, :], axis=0).tolist()
    porous_mae = np.average(errors[zones_ids > 0, :], axis=0).tolist()
    plot_errors('Porous region MAE', porous_mae, save_path=plots_path)
    plot_errors('Fluid region MAE', fluid_mae, save_path=plots_path)

    pred_residuals = np.concatenate(predicted_residuals)
    cfd_residuals = np.concatenate(target_residuals)
    plot_data_dist('Absolute residuals',
                   np.abs(pred_residuals[..., :model.dims]),
                   np.abs(pred_residuals[..., model.dims:]),
                   save_path=plots_path)

    pred_res_avg = trimmed_mean(np.abs(pred_residuals), limits=[0, 0.05], axis=0)
    cfd_res_avg = trimmed_mean(np.abs(cfd_residuals), limits=[0, 0.05], axis=0)
    plot_residuals(pred_res_avg, cfd_res_avg, trim=0.05, save_path=plots_path)

    if args.save_plots:
        errors_dict = {'Total': mae, 'Fluid': fluid_mae, 'Porous': porous_mae,
                       f'{args.subdomain.capitalize()}': additional_mae}
        save_mae_to_csv(errors_dict, ['Ux', 'Uy', 'Uz'][:model.dims] + ['p'], plots_path)


if __name__ == '__main__':
    args = build_arg_parser().parse_args()

    model = get_model(args.checkpoint)

    rng = default_rng(8421)
    data = FoamDataset(args.data_dir, args.n_internal, args.n_boundary, args.n_observations, rng, args.meta_dir)

    evaluate(args, model, data, True, sample_process, postprocess_fn)
