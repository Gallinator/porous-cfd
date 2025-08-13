import argparse
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from lightning import Trainer
from matplotlib import pyplot as plt
from rich.progress import track
from torch_geometric.loader import DataLoader

from foam_dataset import FoamDataset, PdeData
from models.pi_gano_pp import PiGanoPP
from visualization import plot_fields


def build_arg_parser() -> ArgumentParser:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--save-plots', action="store_true",
                            help='save all the inference plots', default=False)
    last_model = sorted(os.listdir('lightning_logs'))[-1]
    default_model_path = Path('lightning_logs') / last_model / 'last.ckpt'
    arg_parser.add_argument('--checkpoint', type=str, default=default_model_path)
    arg_parser.add_argument('--data-dir', type=str, default='data/val')
    arg_parser.add_argument('--meta-dir', type=str, default='data/train/raw')
    return arg_parser


if __name__ == '__main__':
    args = build_arg_parser().parse_args()

    plots_path = None
    if args.save_plots:
        plots_path = Path('plots')
        plots_path.mkdir(exist_ok=True)

    model = PiGanoPP.load_from_checkpoint(args.checkpoint)

    val_data = FoamDataset(args.data_dir, 1000, 200, 500, args.meta_dir)
    val_loader = DataLoader(val_data, 1, False, num_workers=8, pin_memory=True)

    trainer = Trainer(logger=False, enable_checkpointing=False)
    predictions = trainer.predict(model, dataloaders=val_loader)

    for i, (tgt, pred) in enumerate(track(list(zip(val_data, predictions)), description='Saving plots...')):
        pred = PdeData(pred)

        case_plot_path = None
        if plots_path is not None:
            case_plot_path = plots_path / Path(args.data_dir).name / str(i)
            case_plot_path.mkdir(exist_ok=True, parents=True)

        points_scaler = val_data.standard_scaler[0:2].to_torch()
        u_scaler = val_data.standard_scaler[2:4].to_torch()
        p_scaler = val_data.standard_scaler[4].to_torch()

        raw_points = points_scaler.inverse_transform(tgt.pos.numpy(force=True))
        ids = tgt.zones_ids.numpy(force=True)

        plt.interactive(case_plot_path is None)

        plot_fields('Predicted', raw_points, u_scaler.inverse_transform(pred.u),
                    p_scaler.inverse_transform(pred.p), ids, save_path=case_plot_path)
        plot_fields('Ground truth', raw_points, u_scaler.inverse_transform(tgt.pde.u.numpy(force=True)),
                    p_scaler.inverse_transform(tgt.pde.p.numpy(force=True)), ids, save_path=case_plot_path)

        plt.interactive(False)

        u_error = u_scaler.inverse_transform(pred.u) - u_scaler.inverse_transform(tgt.pde.u.numpy(force=True))
        p_error = p_scaler.inverse_transform(pred.p) - p_scaler.inverse_transform(tgt.pde.p.numpy(force=True))
        plot_fields('Absolute error', raw_points, np.abs(u_error), np.abs(p_error), ids,
                    False, case_plot_path)
