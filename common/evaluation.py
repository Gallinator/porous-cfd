import argparse
import csv
import os
import time
from argparse import ArgumentParser
from collections.abc import Callable
from pathlib import Path

import matplotlib
import numpy as np
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import RichProgressBar
from torch import Tensor, cdist
from torch.utils.data import DataLoader
from dataset.data_parser import parse_meta
from dataset.foam_data import FoamData
from dataset.foam_dataset import FoamDataset, collate_fn
from visualization.common import plot_timing


def create_plots_root_dir(args):
    plots_path = None
    if args.save_plots:
        matplotlib.use('Agg')
        plots_path = Path(args.checkpoint).parent / 'plots' / Path(args.data_dir).name / 'stats'
        plots_path.mkdir(exist_ok=True, parents=True)
    return plots_path


def get_normalized_signed_distance(points: Tensor, target: Tensor):
    dist = cdist(points, target)
    dist = torch.min(dist, dim=-1)[0].unsqueeze(-1)
    return dist / torch.max(dist).item()


def get_mean_max_error_distance(errors, quantile, interface_dist):
    q_mask = errors > np.quantile(errors, quantile, axis=-2, keepdims=True)
    q_dist = []
    for m in np.split(q_mask, errors.shape[-1], axis=-1):
        m = m.flatten()
        q_dist.append(np.mean(interface_dist[m]))
    return q_dist


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
                            help='number of internal points to sample', default=1000)
    arg_parser.add_argument('--n-boundary', type=int,
                            help='number of boundary points to sample', default=200)
    arg_parser.add_argument('--n-observations', type=int,
                            help='number of observation points to sample', default=500)
    arg_parser.add_argument('--precision', type=str, default='bf16-mixed')
    arg_parser.add_argument('--model', type=str)
    return arg_parser


def evaluate(args, model, data: FoamDataset, enable_timing,
             sample_process_fn: Callable[[FoamDataset, FoamData, FoamData, FoamData], tuple],
             postprocess_fn: Callable[[FoamDataset, tuple, Path], None]):
    model.verbose_predict = True
    plots_path = create_plots_root_dir(args)

    data_loader = DataLoader(data, 2, False, num_workers=8, pin_memory=True, collate_fn=collate_fn)

    trainer = Trainer(logger=False,
                      enable_checkpointing=False,
                      inference_mode=False,
                      callbacks=[RichProgressBar()],
                      precision=args.precision)
    start_time = time.perf_counter()
    predictions = trainer.predict(model, dataloaders=data_loader)
    inference_time = time.perf_counter() - start_time
    avg_inference_time = inference_time / len(data)
    if enable_timing:
        cfd_timing = parse_meta(args.meta_dir)['Timing']
        plot_timing([inference_time, cfd_timing['Total'] / 1e3],
                    [avg_inference_time, cfd_timing['Average'] / 1e3],
                    plots_path)

    results = []
    for predicted, target in zip(predictions, data_loader):
        pde, extras = predicted
        pde.data = pde.data.to('cpu').detach()
        extras.data = extras.data.to('cpu').detach()
        for i, r in enumerate(sample_process_fn(data, pde, target, extras)):
            if len(results) == i:
                results.append([*r])
            else:
                results[i].extend(r)

    postprocess_fn(data, results, plots_path)
