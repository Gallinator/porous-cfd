import argparse
import csv
import os
import time
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
import numpy as np
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import RichProgressBar
from numpy.random import default_rng
from scipy.stats._mstats_basic import trimmed_mean
from torch.nn.functional import l1_loss
from torch.utils.data import DataLoader
from dataset.data_parser import parse_meta
from dataset.foam_data import FoamData
from dataset.foam_dataset import FoamDataset, collate_fn
from models.pi_gano import PiGano
from models.pipn_foam import PipnFoam
from models.pipn_pp_foam import PipnPpFoam
from visualization import plot_data_dist, plot_timing, plot_errors, plot_residuals


def save_mae_to_csv(errors: dict[str:list], fields_labels, plots_path):
    with open(f'{plots_path}/mae.csv', 'w') as f:
        data = [['Region', *fields_labels]]
        for e in errors.items():
            data.append([e[0], *e[1]])
        csv.writer(f).writerows(data)


def load_model(model_type: str, checkpoint):
    match model_type:
        case 'pipn-foam':
            return PipnFoam.load_from_checkpoint(checkpoint)
        case 'pipn-pp-foam':
            return PipnPpFoam.load_from_checkpoint(checkpoint)
        case 'pi-gano-3d':
            return PiGano.load_from_checkpoint(checkpoint)
    raise NotImplementedError(f'{model_type} is not available!')


def build_arg_parser() -> ArgumentParser:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--save-plots', action="store_true",
                            help='save all the inference plots', default=False)
    last_model = sorted(os.listdir('lightning_logs'))[-1]
    default_model_path = Path('lightning_logs') / last_model / 'model.ckpt'
    arg_parser.add_argument('--checkpoint', type=str, default=default_model_path)
    arg_parser.add_argument('--data-dir', type=str, default='data/val')
    arg_parser.add_argument('--meta-dir', type=str, default='data/train')
    arg_parser.add_argument('--n-internal', type=int,
                            help='number of internal points to sample', default=3000)
    arg_parser.add_argument('--n-boundary', type=int,
                            help='number of internal points to sample', default=700)
    arg_parser.add_argument('--n-observations', type=int,
                            help='number of observation points to sample', default=1200)
    arg_parser.add_argument('--precision', type=str, default='bf16-mixed')
    arg_parser.add_argument('--subdomain', type=str, default='')
    return arg_parser


if __name__ == '__main__':
    args = build_arg_parser().parse_args()

    plots_path = None
    if args.save_plots:
        matplotlib.use('Agg')
        plots_path = Path(args.checkpoint).parent / 'plots' / Path(args.data_dir).name / 'stats'
        plots_path.mkdir(exist_ok=True, parents=True)

    model = load_model(args.model, args.checkpoint)
    model.verbose_predict = True

    error_labels = {'momentx': None, 'momenty': None, 'momentz': None, 'div': None,
                    'moment': ['momentx', 'momenty', 'momentz']}
    rng = default_rng(8421)
    val_data = FoamDataset(args.data_dir,
                           ['C', 'U', 'p', 'cellToRegion', 'd', 'f', 'momentError', 'div(phi)'],
                           args.n_internal,
                           args.n_boundary,
                           args.n_observations,
                           rng,
                           {'Ux': 'inlet'},
                           {'Scale': ['d', 'f'], 'Standardize': ['C', 'U', 'p']},
                           args.meta_dir)
    val_loader = DataLoader(val_data, 2, False, num_workers=8, pin_memory=True, collate_fn=collate_fn)

    trainer = Trainer(logger=False,
                      enable_checkpointing=False,
                      inference_mode=False,
                      callbacks=[RichProgressBar()],
                      precision=args.precision)

    start_time = time.perf_counter()
    pred = trainer.predict(model, dataloaders=val_loader)
    inference_time = time.perf_counter() - start_time
    avg_inference_time = inference_time / len(val_data)
    val_timing = parse_meta(args.meta_dir)['Timing']
    plot_timing([inference_time, val_timing['Total'] / 1e3],
                [avg_inference_time, val_timing['Average'] / 1e3],
                plots_path)

    errors, pred_residuals, cfd_residuals, zones_ids = [], [], [], []
    additional_errors = []
    u_scaler = val_data.normalizers['U'].to(model.device)
    p_scaler = val_data.normalizers['p'].to(model.device)

    for p, tgt_data in zip(pred, val_loader):
        pred_data, phys_data = p
        pred_data = FoamData(pred_data, model.pred_labels, tgt_data.domain)
        # Total error
        u_error = l1_loss(u_scaler.inverse_transform(pred_data['U']),
                          u_scaler.inverse_transform(tgt_data['U']), reduction='none')
        p_error = l1_loss(p_scaler.inverse_transform(pred_data['p']),
                          p_scaler.inverse_transform(tgt_data['p']), reduction='none')
        pde_error = torch.cat([u_error, p_error], dim=-1)
        errors.extend(pde_error.numpy(force=True))
        zones_ids.extend(tgt_data['cellToRegion'].numpy(force=True))

        if args.subdomain != '':
            additional_u_error = l1_loss(u_scaler.inverse_transform(pred_data['solid']['U']),
                                         u_scaler.inverse_transform(tgt_data['solid']['U']), reduction='none')
            additional_p_error = l1_loss(p_scaler.inverse_transform(pred_data['solid']['p']),
                                         p_scaler.inverse_transform(tgt_data['solid']['p']), reduction='none')
            additional_pde_error = torch.cat([additional_u_error, additional_p_error], dim=-1)
            additional_errors.extend(additional_pde_error.numpy(force=True))

        # Equation residuals
        pred_residuals.extend(phys_data.numpy(force=True))
        cfd_res = torch.cat([tgt_data['internal']['momentError'], tgt_data['internal']['div(phi)']], dim=-1)
        cfd_residuals.extend(cfd_res.numpy(force=True))

    errors, zones_ids = np.concatenate(errors), np.array(zones_ids).flatten()
    plot_data_dist('Absolute error distribution',
                   errors[..., :model.dims],
                   errors[..., model.dims:],
                   save_path=plots_path)
    mae = np.average(errors, axis=0).tolist()
    plot_errors('Average relative error', mae, save_path=plots_path)

    if args.subdomain != '':
        additional_errors = np.concatenate(additional_errors)
        plot_data_dist(f'{args.subdomain.capitalize()} Absolute error distribution',
                       additional_errors[..., :model.dims],
                       additional_errors[..., model.dims:],
                       save_path=plots_path)
        additional_mae = np.average(additional_errors, axis=0).tolist()
        plot_errors(f'{args.subdomain.capitalize()} Average relative error', additional_mae, save_path=plots_path)

    fluid_mae = np.average(errors[zones_ids < 1, :], axis=0).tolist()
    porous_mae = np.average(errors[zones_ids > 0, :], axis=0).tolist()
    plot_errors('Porous region MAE', porous_mae, save_path=plots_path)
    plot_errors('Fluid region MAE', fluid_mae, save_path=plots_path)

    pred_residuals = np.concatenate(pred_residuals)
    cfd_residuals = np.concatenate(cfd_residuals)
    plot_data_dist('Absolute residuals',
                   np.abs(pred_residuals[..., :model.dims]),
                   np.abs(pred_residuals[..., model.dims:]),
                   save_path=plots_path)

    pred_res_avg = trimmed_mean(np.abs(pred_residuals), limits=[0, 0.05], axis=0)
    cfd_res_avg = trimmed_mean(np.abs(cfd_residuals), limits=[0, 0.05], axis=0)
    plot_residuals(pred_res_avg, cfd_res_avg, trim=0.05, save_path=plots_path)

    if args.save_plots:
        errors_dict = {'Total': mae,
                       'Fluid': fluid_mae,
                       'Porous': porous_mae}
        if args.subdomain != '':
            errors_dict[f'{args.subdomain.capitalize()}'] = additional_mae
        save_mae_to_csv(errors_dict, ['Ux', 'Uy', 'Uz'][:model.dims] + ['p'], plots_path)
