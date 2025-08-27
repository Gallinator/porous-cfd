import argparse
import os
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import RichProgressBar
from numpy.random import default_rng
from scipy.stats._mstats_basic import trimmed_mean
from torch.nn.functional import l1_loss
from torch.utils.data import DataLoader
from foam_dataset import FoamDataset, PdeData
from models.pipn import Pipn, FoamData
from models.pipn_pp import PipnPp
from visualization import plot_data_dist, plot_errors, plot_residuals


def build_arg_parser() -> ArgumentParser:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--save-plots', action="store_true",
                            help='save all the inference plots', default=False)
    last_model = sorted(os.listdir('lightning_logs'))[-1]
    default_model_path = Path('lightning_logs') / last_model / 'checkpoints'
    default_model_path /= os.listdir(default_model_path)[0]
    arg_parser.add_argument('--checkpoint', type=str, default=default_model_path)
    arg_parser.add_argument('--data-dir', type=str, default='data/standard')
    arg_parser.add_argument('--meta-dir', type=str, default='data/standard')
    arg_parser.add_argument('--n-internal', type=int,
                            help='number of internal points to sample', default=667)
    arg_parser.add_argument('--n-boundary', type=int,
                            help='number of internal points to sample', default=168)
    arg_parser.add_argument('--plusplus', action="store_true",
                            help='save all the inference plots', default=False)
    return arg_parser


if __name__ == '__main__':
    args = build_arg_parser().parse_args()

    plots_path = None
    if args.save_plots:
        plots_path = Path(args.checkpoint).parent / 'plots' / Path(args.data_dir).name / 'stats'
        plots_path.mkdir(exist_ok=True, parents=True)

    model = PipnPp.load_from_checkpoint(args.checkpoint) if args.plusplus else Pipn.load_from_checkpoint(
        args.checkpoint)
    model.verbose_predict = True

    rng = default_rng(8421)
    val_data = FoamDataset(args.data_dir, args.n_internal, args.n_boundary, args.meta_dir, rng=rng)
    val_loader = DataLoader(val_data, 2, False, num_workers=8, pin_memory=True)

    trainer = Trainer(logger=False, enable_checkpointing=False, inference_mode=False, callbacks=[RichProgressBar()])

    start_time = time.perf_counter()
    pred = trainer.predict(model, dataloaders=val_loader)

    errors, pred_residuals, cfd_residuals, zones_ids = [], [], [], []
    for i, (p, t) in enumerate(zip(pred, val_loader)):
        pred_data, phys_data = p
        tgt_data = FoamData(t)
        error = l1_loss(pred_data, tgt_data.pde.data, reduction='none')
        errors.extend(error.numpy(force=True))
        zones_ids.extend(tgt_data.zones_ids)

        pred_residuals.extend(phys_data.numpy(force=True))
        res_tgt = torch.zeros_like(tgt_data.fx)
        cfd_res = torch.cat([res_tgt, res_tgt, res_tgt], dim=-1)
        cfd_residuals.extend(cfd_res[..., :args.n_internal, :].numpy(force=True))

    errors, zones_ids = np.concatenate(errors), np.array(zones_ids).flatten()
    error_data = PdeData(errors)
    plot_data_dist('Absolute error distribution', error_data.u, error_data.p, save_path=plots_path)

    mae = np.average(errors, axis=0)
    plot_errors('MAE', mae.tolist(), save_path=plots_path)

    porous_mae, fluid_mae = np.average(errors[zones_ids > 0,:], axis=0), np.average(errors[zones_ids < 1,:], axis=0)
    plot_errors('Porous region MAE', porous_mae.tolist(), save_path=plots_path)
    plot_errors('Fluid region MAE', fluid_mae.tolist(), save_path=plots_path)

    pred_residuals = np.concatenate(pred_residuals)
    cfd_residuals = np.concatenate(cfd_residuals)
    plot_data_dist('Absolute residuals', np.abs(pred_residuals[:, 0:2]), np.abs(pred_residuals[:, 2:3]),
                   save_path=plots_path)

    pred_res_avg = trimmed_mean(np.abs(pred_residuals), limits=[0, 0.05], axis=0)
    cfd_res_avg = trimmed_mean(np.abs(cfd_residuals), limits=[0, 0.05], axis=0)
    plot_residuals(pred_res_avg, cfd_res_avg, trim=0.05, save_path=plots_path)
