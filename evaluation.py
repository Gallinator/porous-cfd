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
from data_parser import parse_meta
from foam_dataset import FoamDataset, PdeData, collate_fn
from models.pi_gano import PiGano
from visualization import plot_data_dist, plot_timing, plot_errors, plot_residuals


def save_mae_to_csv(errors: dict[str:list], fields_labels, plots_path):
    with open(f'{plots_path}/mae.csv', 'w') as f:
        data = [['Region', *fields_labels]]
        for e in errors.items():
            data.append([e[0], *e[1]])
        csv.writer(f).writerows(data)


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
    arg_parser.add_argument('--precision', type=str, default='32-true')
    return arg_parser


if __name__ == '__main__':
    args = build_arg_parser().parse_args()

    plots_path = None
    if args.save_plots:
        matplotlib.use('Agg')
        plots_path = Path(args.checkpoint).parent / 'plots' / Path(args.data_dir).name / 'stats'
        plots_path.mkdir(exist_ok=True, parents=True)

    model = PiGano.load_from_checkpoint(args.checkpoint)
    model.verbose_predict = True

    rng = default_rng(8421)
    val_data = FoamDataset(args.data_dir, args.n_internal, args.n_boundary, args.n_observations, args.meta_dir, rng=rng)
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
    solid_errors = []
    pde_scaler = val_data.standard_scaler[3:7].to(model.device)
    for p, tgt_data in zip(pred, val_loader):
        pred_data, phys_data = p
        pred_data = PdeData(pred_data, val_data.domain_dict)
        batch_error = l1_loss(pde_scaler.inverse_transform(pred_data.data),
                              pde_scaler.inverse_transform(tgt_data.pde.data), reduction='none')

        errors.extend(batch_error.numpy(force=True))
        zones_ids.extend(tgt_data.zones_ids)

        solid_error = l1_loss(pde_scaler.inverse_transform(pred_data['solid'].data),
                              pde_scaler.inverse_transform(tgt_data['solid'].pde.data), reduction='none')
        solid_errors.extend(solid_error.numpy(force=True))

        pred_residuals.extend(phys_data.numpy(force=True))
        cfd_res = torch.cat([tgt_data.mom_x, tgt_data.mom_y, tgt_data.mom_z, tgt_data.div], dim=-1)
        cfd_residuals.extend(cfd_res[..., :args.n_internal, :].numpy(force=True))

    errors, zones_ids = np.concatenate(errors), np.array(zones_ids).flatten()
    error_data = PdeData(errors)
    plot_data_dist('Absolute error distribution', error_data.u, error_data.p, save_path=plots_path)

    solid_errors = np.concatenate(solid_errors)
    solid_error_data = PdeData(solid_errors)
    plot_data_dist('Solid Absolute error distribution', solid_error_data.u, solid_error_data.p, save_path=plots_path)

    mae = np.average(errors, axis=0)
    plot_errors('Average relative error', mae.tolist(), save_path=plots_path)

    solid_mae = np.average(solid_errors, axis=0)
    plot_errors('Solid Average relative error', solid_mae.tolist(), save_path=plots_path)

    porous_mae, fluid_mae = np.average(errors[zones_ids > 0, :], axis=0), np.average(errors[zones_ids < 1, :], axis=0)
    plot_errors('Porous region MAE', porous_mae.tolist(), save_path=plots_path)
    plot_errors('Fluid region MAE', fluid_mae.tolist(), save_path=plots_path)

    pred_residuals = np.concatenate(pred_residuals)
    cfd_residuals = np.concatenate(cfd_residuals)
    plot_data_dist('Absolute residuals', np.abs(pred_residuals[..., 0:3]), np.abs(pred_residuals[..., 3:4]),
                   save_path=plots_path)

    pred_res_avg = trimmed_mean(np.abs(pred_residuals), limits=[0, 0.05], axis=0)
    cfd_res_avg = trimmed_mean(np.abs(cfd_residuals), limits=[0, 0.05], axis=0)
    plot_residuals(pred_res_avg, cfd_res_avg, trim=0.05, save_path=plots_path)

    if args.save_plots:
        save_mae_to_csv({'Solid': solid_mae.tolist(), 'Total': mae.tolist()},
                        ['ux', 'uy', 'uz', 'p'],
                        plots_path)
