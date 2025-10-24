import argparse
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Callable

import matplotlib
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import RichProgressBar
from rich.progress import track
from torch.utils.data import DataLoader

from dataset.foam_data import FoamData
from dataset.foam_dataset import collate_fn, FoamDataset


def build_arg_parser() -> ArgumentParser:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--save-plots', action="store_true",
                            help='save all the inference plots', default=False)
    last_model = sorted(os.listdir('lightning_logs'))[-1]
    default_model_path = Path('lightning_logs') / last_model / 'model.ckpt'
    arg_parser.add_argument('--checkpoint', type=str, default=default_model_path)
    arg_parser.add_argument('--data-dir', type=str, default='data/test')
    arg_parser.add_argument('--meta-dir', type=str, default='data/train')
    arg_parser.add_argument('--n-internal', type=int,
                            help='number of internal points to sample', default=1000)
    arg_parser.add_argument('--n-boundary', type=int,
                            help='number of boundary points to sample', default=200)
    arg_parser.add_argument('--n-observations', type=int,
                            help='number of observation points to sample', default=500)
    arg_parser.add_argument('--precision', type=str, default='bf16-mixed')
    return arg_parser


def create_plots_root(args):
    plots_path = None
    if args.save_plots:
        plots_path = Path(args.checkpoint).parent / 'plots' / Path(args.data_dir).name
        plots_path.mkdir(exist_ok=True, parents=True)
    return plots_path


def create_case_plot_dir(plots_root, case_name):
    case_plot_dir = None
    if plots_root is not None:
        case_plot_dir = plots_root / case_name
        case_plot_dir.mkdir(exist_ok=True, parents=True)
    return case_plot_dir


def predict(args, model, data: FoamDataset,
            result_process_fn: Callable[[FoamDataset, FoamData, FoamData, Path, Path], None]):
    torch.manual_seed(8421)
    data_loader = DataLoader(data, 1, False, num_workers=8, pin_memory=True, collate_fn=collate_fn)

    trainer = Trainer(logger=False,
                      enable_checkpointing=False,
                      callbacks=[RichProgressBar()],
                      precision=args.precision)
    predictions = trainer.predict(model, dataloaders=data_loader)

    if args.save_plots:
        default_backend = matplotlib.get_backend()
        matplotlib.use('Agg')

    plots_base_path = create_plots_root(args)

    for i, (target, predicted) in enumerate(track(list(zip(data, predictions)), description='Saving plots...')):
        case_path = Path(data.samples[i])
        target = target.to('cpu')
        predicted = predicted.to('cpu').squeeze()

        case_plot_path = create_case_plot_dir(plots_base_path, case_path.name)
        result_process_fn(data, target, predicted, case_path, case_plot_path)

    if args.save_plots:
        matplotlib.use(default_backend)
