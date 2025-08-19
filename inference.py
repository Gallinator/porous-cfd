import argparse
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from lightning import Trainer
from matplotlib import pyplot as plt
from rich.progress import track
from torch.utils.data import DataLoader

from foam_dataset import FoamDataset, PdeData
from models.pipn import Pipn, FoamData
from visualization import plot_fields

CHECKPOINT_PATH = 'lightning_logs/version_22/checkpoints/epoch=402-step=806.ckpt'


def build_arg_parser() -> ArgumentParser:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--save-plots', action="store_true",
                            help='save all the inference plots', default=False)
    last_model = sorted(os.listdir('lightning_logs'))[-1]
    default_model_path = Path('lightning_logs') / last_model / 'checkpoints'
    default_model_path /= os.listdir(default_model_path)[0]
    arg_parser.add_argument('--checkpoint', type=str, default=default_model_path)
    arg_parser.add_argument('--data-dir', type=str, default='data/val')
    arg_parser.add_argument('--meta-dir', type=str, default='data/train')
    arg_parser.add_argument('--n-internal', type=int,
                            help='number of internal points to sample', default=1000)
    arg_parser.add_argument('--n-boundary', type=int,
                            help='number of internal points to sample', default=200)
    arg_parser.add_argument('--n-observations', type=int,
                            help='number of observation points to sample', default=500)
    return arg_parser


if __name__ == '__main__':
    args = build_arg_parser().parse_args()

    plots_path = None
    if args.save_plots:
        plots_path = Path('plots')
        plots_path.mkdir(exist_ok=True)

    model = Pipn.load_from_checkpoint(args.checkpoint)

    val_data = FoamDataset(args.data_dir, args.n_internal, args.n_boundary, args.n_observations, args.meta_dir)
    val_loader = DataLoader(val_data, 1, False, num_workers=8, pin_memory=True)

    trainer = Trainer(logger=False, enable_checkpointing=False)
    predictions = trainer.predict(model, dataloaders=val_loader)
    for i, (tgt, pred) in enumerate(track(list(zip(val_data, predictions)), description='Saving plots...')):
        pred = PdeData(pred).numpy()

        case_plot_path = None
        if plots_path is not None:
            case_plot_path = plots_path / Path(args.data_dir).name / str(i)
            case_plot_path.mkdir(exist_ok=True, parents=True)

        tgt = FoamData(tgt).numpy()

        plt.interactive(case_plot_path is None)

        plot_fields('Predicted', pred.points, pred.u[0], pred.p[0], tgt.zones_ids, save_path=case_plot_path)
        plot_fields('Ground truth', pred.points, tgt.pde.u, tgt.pde.p, tgt.zones_ids, save_path=case_plot_path)

        plt.interactive(False)

        u_error = pred.u[0] - tgt.pde.u
        p_error = pred.p[0] - tgt.pde.p
        plot_fields('Absolute error', pred.points, np.abs(u_error), np.abs(p_error), tgt.zones_ids, False,
                    save_path=case_plot_path)
