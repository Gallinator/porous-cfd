import argparse
import csv
import os
import time
from argparse import ArgumentParser
from collections.abc import Callable
from pathlib import Path
from typing import Any
import matplotlib
import numpy as np
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import RichProgressBar
from scipy.stats._mstats_basic import trimmed_mean
from torch import Tensor, cdist
from torch.nn.functional import l1_loss
from torch.utils.data import DataLoader
from dataset.data_parser import parse_meta
from dataset.foam_data import FoamData
from dataset.foam_dataset import FoamDataset, collate_fn, StandardScaler, Normalizer
from visualization.common import plot_timing, box_plot, plot_data_dist, plot_residuals, plot_errors


def create_plots_root_dir(args):
    plots_path = None
    if args.save_plots:
        matplotlib.use('Agg')
        plots_path = Path(args.checkpoint).parent / 'plots' / Path(args.data_dir).name / 'stats'
        plots_path.mkdir(exist_ok=True, parents=True)
    return plots_path


def extract_coef(coef: Tensor, scaler: StandardScaler | Normalizer):
    coef = scaler.inverse_transform(coef)[..., 0:1]
    return torch.max(coef, dim=-2, keepdim=True)[0]


def extract_u_magnitude(u: Tensor, scaler: StandardScaler, spacing):
    u_mag = scaler.inverse_transform(u)
    u_mag = torch.norm(u_mag, dim=-1, keepdim=True)
    u_mag = torch.max(u_mag, dim=-2, keepdim=True)[0]
    return torch.round(u_mag / spacing) * spacing


def extract_angle(u: Tensor, scaler: StandardScaler):
    u = scaler.inverse_transform(u)
    u_mag = torch.norm(u, dim=-1, keepdim=True)
    a = torch.arccos(u[..., 0:1] / u_mag)
    a = torch.max(a, dim=-2, keepdim=True)[0]
    return torch.rad2deg(a)


def get_normalized_signed_distance(points: Tensor, target: Tensor):
    dist = cdist(points, target)
    dist = torch.min(dist, dim=-1)[0].unsqueeze(-1)
    return dist / torch.max(dist).item()


def get_mean_max_error_distance(errors, quantile, interface_dist):
    q_mask = errors > np.quantile(errors, quantile, axis=-2, keepdims=True)
    q_dist = []
    # Loop over each batch
    for mask, dist in zip(q_mask, interface_dist):
        # Extract mean distance for each field
        dim_masks = np.split(mask, errors.shape[-1], axis=-1)
        field_dists = [dist[m.flatten()] for m in dim_masks]
        means = [np.mean(d) for d in field_dists]
        q_dist.append(np.array(means))
    # Average over all batches
    return np.mean(np.stack(q_dist), axis=0)


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
    return arg_parser


def get_common_data(data: FoamDataset, predicted: FoamData, target: FoamData, extras: FoamData) -> dict[str, Any]:
    predicted_u, predicted_p = predicted['U'], predicted['p']
    target_u, target_p = target['U'], target['p']
    if 'U' in data.normalizers:
        data.normalizers['U'].to()
        predicted_u = data.normalizers['U'].to().inverse_transform(predicted_u)
        target_u = data.normalizers['U'].to().inverse_transform(target_u)
    if 'p' in data.normalizers:
        data.normalizers['p'].to()
        predicted_p = data.normalizers['p'].to().inverse_transform(predicted_p)
        target_p = data.normalizers['p'].to().inverse_transform(target_p)

    u_error = l1_loss(predicted_u, target_u, reduction='none')
    p_error = l1_loss(predicted_p, target_p, reduction='none')

    predicted_div, predicted_momentum = extras['div'], extras['Momentum']
    target_div, target_momentum = torch.zeros_like(predicted_div), torch.zeros_like(predicted_momentum)

    if 'momentError' in target and 'div(phi)' in target:
        target_div = target['internal']['div(phi)']
        target_momentum = target['internal']['momentError']

    return {'U error': u_error,
            'p error': p_error,
            'Predicted momentum': predicted_momentum,
            'Predicted divergence': predicted_div,
            'Target momentum': target_momentum,
            'Target divergence': target_div,
            'Region id': target['cellToRegion']}


def plot_common_data(data: dict, plots_path):
    errors = np.concatenate([data['U error'], data['p error']], axis=-1)
    max_error_per_case = np.max(errors, axis=1)
    box_labels = ['$U_x$', '$U_y$', '$U_z$'][:errors.shape[-1]] + ['$p$']
    box_plot('Maximum errors per case',
             [*np.hsplit(max_error_per_case, errors.shape[-1])],
             box_labels,
             plots_path)

    u_errors, p_errors = np.concatenate(data['U error']), np.concatenate(data['p error'])
    plot_data_dist('Absolute error distribution', u_errors, p_errors, save_path=plots_path)

    errors = np.concatenate([u_errors, p_errors], -1)
    mae = np.mean(errors, axis=0).tolist()
    plot_errors('Average relative error', mae, save_path=plots_path)

    zones_ids = data['Region id'].flatten()
    fluid_mae = np.mean(errors[zones_ids < 1, :], axis=0).tolist()
    plot_errors('Fluid region MAE', fluid_mae, save_path=plots_path)

    porous_mae = np.mean(errors[zones_ids > 0, :], axis=0).tolist()
    plot_errors('Porous region MAE', porous_mae, save_path=plots_path)

    predicted_div = np.concatenate(data['Predicted divergence'])
    predicted_momentum = np.concatenate(data['Predicted momentum'])

    plot_data_dist('Absolute residuals', np.abs(predicted_momentum), np.abs(predicted_div), save_path=plots_path)

    target_div = np.concatenate(data['Target momentum'])
    target_momentum = np.concatenate(data['Target divergence'])
    target_residuals = np.concatenate([target_momentum, target_div], axis=-1)
    predicted_residuals = np.concatenate([predicted_momentum, predicted_div], axis=-1)
    pred_res_avg = trimmed_mean(np.abs(predicted_residuals), limits=[0, 0.05], axis=0)
    cfd_res_avg = trimmed_mean(np.abs(target_residuals), limits=[0, 0.05], axis=0)
    plot_residuals(pred_res_avg, cfd_res_avg, trim=0.05, save_path=plots_path)

    if plots_path is not None:
        errors_dict = {'Total': mae, 'Fluid': fluid_mae, 'Porous': porous_mae}
        save_mae_to_csv(errors_dict, box_labels, plots_path)


def evaluate(args, model, data: FoamDataset, enable_timing,
             sample_process_fn: Callable[[FoamDataset, FoamData, FoamData, FoamData], dict[str, Any]],
             postprocess_fn: Callable[[FoamDataset, dict[str, Any], Path], None]):
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

    if args.save_plots:
        default_backend = matplotlib.get_backend()
        matplotlib.use('Agg')

    if enable_timing:
        cfd_timing = parse_meta(args.meta_dir)['Timing']
        plot_timing([inference_time, cfd_timing['Total'] / 1e3],
                    [avg_inference_time, cfd_timing['Average'] / 1e3],
                    plots_path)

    results = None
    for predicted, target in zip(predictions, data_loader):
        pde, extras = predicted
        pde.data = pde.data.to('cpu').detach()
        extras.data = extras.data.to('cpu').detach()

        sample_data = get_common_data(data, pde, target, extras)
        sample_data.update(sample_process_fn(data, pde, target, extras))
        if results is None:
            results = dict.fromkeys(sample_data.keys(), [])

        for k, v in sample_data.items():
            old_v = results[k]
            results[k] = old_v + [v]

    for k, v in results.items():
        if isinstance(v[0], Tensor):
            results[k] = np.concatenate([i.numpy() for i in v])

    plot_common_data(results, plots_path)
    postprocess_fn(data, results, plots_path)

    if args.save_plots:
        matplotlib.use(default_backend)
