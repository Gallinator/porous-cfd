import argparse
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import RichProgressBar
from torch.utils.data import DataLoader
from rich.progress import track
from dataset.foam_data import FoamData
from dataset.foam_dataset import FoamDataset, collate_fn
from models.pi_gano import PiGano
from visualization import plot_fields, plot_streamlines, plot_houses


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

    rng = np.random.default_rng(8421)

    plots_path = None
    if args.save_plots:
        plots_path = Path(args.checkpoint).parent / 'plots' / Path(args.data_dir).name
        plots_path.mkdir(exist_ok=True, parents=True)

    model = PiGano.load_from_checkpoint(args.checkpoint)

    val_data = FoamDataset(args.data_dir, args.n_internal, args.n_boundary, args.n_observations, rng, args.meta_dir)
    val_loader = DataLoader(val_data, 1, False, num_workers=8, pin_memory=True, collate_fn=collate_fn)

    trainer = Trainer(logger=False,
                      enable_checkpointing=False,
                      callbacks=[RichProgressBar()],
                      precision=args.precision)
    predictions = trainer.predict(model, dataloaders=val_loader)

    for i, (tgt, pred) in enumerate(track(list(zip(val_data, predictions)), description='Saving plots...')):
        pred = FoamData(pred[0], model.pred_labels, tgt.domain)

        case_plot_path = None
        if plots_path is not None:
            case_plot_path = plots_path / Path(val_data.samples[i]).name
            case_plot_path.mkdir(exist_ok=True, parents=True)

        points_scaler = val_data.normalizers['C'].to()
        u_scaler = val_data.normalizers['U'].to()
        p_scaler = val_data.normalizers['p'].to()
        d_scaler = val_data.normalizers['d'].to()
        f_scaler = val_data.normalizers['f'].to()

        raw_points = points_scaler.inverse_transform(tgt['C']).numpy(force=True)

        d = torch.max(d_scaler.inverse_transform(tgt['d']))
        f = torch.max(f_scaler.inverse_transform(tgt['f']))
        inlet_ux = torch.max(u_scaler[0].inverse_transform(tgt['Ux-inlet']))

        plot_fields(f'Predicted D={d:.3f} F={f:.3f} Inlet={inlet_ux:.3f}',
                    raw_points,
                    u_scaler.inverse_transform(pred['U']).numpy(force=True),
                    p_scaler.inverse_transform(pred['p']).numpy(force=True),
                    tgt['cellToRegion'].numpy(force=True),
                    save_path=case_plot_path)
        plot_streamlines('Predicted streamlines',
                         val_data.samples[i],
                         raw_points,
                         u_scaler.inverse_transform(pred['U']).numpy(force=True),
                         save_path=case_plot_path)

        plot_fields(f'Ground truth D={d:.3f} F={f:.3f} Inlet={inlet_ux:.3f}',
                    raw_points,
                    u_scaler.inverse_transform(tgt['U']).numpy(force=True),
                    p_scaler.inverse_transform(tgt['p']).numpy(force=True),
                    tgt['cellToRegion'].numpy(force=True),
                    save_path=case_plot_path)
        plot_streamlines('True streamlines',
                         val_data.samples[i],
                         raw_points,
                         u_scaler.inverse_transform(tgt['U'].numpy(force=True)),
                         save_path=case_plot_path)

        u_error = (u_scaler.inverse_transform(pred['U']) - u_scaler.inverse_transform(tgt['U'])).numpy(force=True)
        p_error = p_scaler.inverse_transform(pred['p']) - p_scaler.inverse_transform(tgt['p']).numpy(force=True)
        plot_fields(f'Absolute error D={d:.3f} F={f:.3f} Inlet={inlet_ux:.3f}',
                    raw_points,
                    np.abs(u_error),
                    np.abs(p_error),
                    tgt['cellToRegion'].numpy(force=True),
                    save_path=case_plot_path)
        plot_streamlines('Error streamlines',
                         val_data.samples[i],
                         raw_points,
                         np.abs(u_error),
                         save_path=case_plot_path)

        solid_points = points_scaler.inverse_transform(tgt['solid']['C']).numpy(force=True)
        solid_u_error = u_scaler.inverse_transform(pred['solid']['U']) - u_scaler.inverse_transform(
            tgt['solid']['U']).numpy(force=True)
        solid_p_error = p_scaler.inverse_transform(pred['solid']['p']) - p_scaler.inverse_transform(
            tgt['solid']['p']).numpy(force=True)
        plot_houses('House',
                    solid_points,
                    np.abs(solid_u_error),
                    np.abs(solid_p_error),
                    f'{val_data.samples[i]}/constant/triSurface/solid.obj',
                    save_path=case_plot_path)
