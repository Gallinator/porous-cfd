import argparse
import os
from argparse import ArgumentParser

import torch
from lightning.pytorch.callbacks import RichProgressBar, LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader
from numpy.random import default_rng
from foam_dataset import FoamDataset
from models.pi_gano import PiGano
import lightning as L


def build_arg_parser() -> ArgumentParser:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--n-internal', type=int,
                            help='number of internal points to sample', default=1000)
    arg_parser.add_argument('--n-boundary', type=int,
                            help='number of internal points to sample', default=10)
    arg_parser.add_argument('--n-observations', type=int,
                            help='number of observation points to sample', default=10)
    arg_parser.add_argument('--batch-size', type=int, default=13)
    arg_parser.add_argument('--precision', type=str, default='32-true')
    arg_parser.add_argument('--epochs', type=int, default=3000)
    arg_parser.add_argument('--logs-dir', type=str, default=os.getcwd())
    arg_parser.add_argument('--train-dir', type=str, default='data/train')
    arg_parser.add_argument('--val-dir', type=str, default='data/val')
    return arg_parser


if __name__ == '__main__':
    args = build_arg_parser().parse_args()

    torch.set_float32_matmul_precision('high')

    batch_size = args.batch_size
    n_internal = args.n_internal
    n_boundary = args.n_boundary
    n_obs = args.n_observations
    epochs = args.epochs

    rng = default_rng(8421)
    train_data = FoamDataset(args.train_dir, n_internal, n_boundary, n_obs, rng=rng)
    train_loader = DataLoader(train_data, batch_size, True, num_workers=8)
    val_data = FoamDataset(args.val_dir, n_internal, n_boundary, n_obs, args.train_dir, rng=rng)
    val_loader = DataLoader(val_data, batch_size, False, num_workers=8, pin_memory=True)

    scalers = {'U': train_data.standard_scaler[3:6],
               'p': train_data.standard_scaler[6],
               'Points': train_data.standard_scaler[0:3],
               'd': train_data.d_normalizer,
               'f': train_data.f_normalizer}

    model = PiGano(train_data.domain_dict, scalers)

    checkpoint_callback = ModelCheckpoint(filename='checkpoint-{epoch:d}', every_n_epochs=500, save_top_k=-1)

    trainer = L.Trainer(max_epochs=epochs,
                        callbacks=[RichProgressBar(), checkpoint_callback, LearningRateMonitor()],
                        log_every_n_steps=int(len(train_data) / batch_size),
                        precision=args.precision,
                        default_root_dir=args.logs_dir)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    trainer.save_checkpoint(f'{trainer.log_dir}/model.ckpt')
