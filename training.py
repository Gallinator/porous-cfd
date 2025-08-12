import argparse
import os
from argparse import ArgumentParser

import torch
from lightning.pytorch.callbacks import RichProgressBar, LearningRateMonitor
from torch_geometric.data import DataLoader
from foam_dataset import FoamDataset
from foam_dataset import FoamData
from models.pipn_pp import PipnPP
import lightning as L


def build_arg_parser() -> ArgumentParser:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--n-internal', type=int,
                            help='number of internal points to sample', default=1000)
    arg_parser.add_argument('--n-boundary', type=int,
                            help='number of internal points to sample', default=200)
    arg_parser.add_argument('--n-observations', type=int,
                            help='number of observation points to sample', default=500)
    arg_parser.add_argument('--batch-size', type=int, default=13)
    arg_parser.add_argument('--precision', type=str, default='32-true')
    arg_parser.add_argument('--epochs', type=int, default=3000)
    arg_parser.add_argument('--logs-dir', type=str, default=os.getcwd())
    return arg_parser


if __name__ == '__main__':
    args = build_arg_parser().parse_args()

    torch.set_float32_matmul_precision('high')

    batch_size = args.batch_size
    n_internal = args.n_internal
    n_boundary = args.n_boundary
    n_obs = args.n_observations
    epochs = args.epochs

    train_data = FoamDataset('data/train', n_internal, n_boundary, n_obs)
    train_loader = DataLoader(train_data, batch_size, True, num_workers=8)
    val_data = FoamDataset('data/val', n_internal, n_boundary, n_obs, 'data/train/raw')
    val_loader = DataLoader(val_data, batch_size, False, num_workers=8, pin_memory=True)

    scalers = {'U': train_data.standard_scaler[2:4],
               'p': train_data.standard_scaler[4],
               'Points': train_data.standard_scaler[0:2]}
    model = PipnPP(n_internal, n_boundary, scalers)

    trainer = L.Trainer(max_epochs=epochs,
                        callbacks=[RichProgressBar(), LearningRateMonitor()],
                        log_every_n_steps=int(batch_size / len(train_data)),
                        precision=args.precision,
                        default_root_dir=args.logs_dir)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
