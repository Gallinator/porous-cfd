import argparse
import os
from argparse import ArgumentParser
import torch
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint
from numpy.random import default_rng
from torch.utils.data import DataLoader
from foam_dataset import FoamDataset
from models.pipn import Pipn
import lightning as L


def build_arg_parser() -> ArgumentParser:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--n-internal', type=int,
                            help='number of internal points to sample', default=667)
    arg_parser.add_argument('--n-boundary', type=int,
                            help='number of internal points to sample', default=168)
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
    epochs = args.epochs

    rng = default_rng(8421)
    train_data = FoamDataset('data/standard', n_internal, n_boundary, rng=rng)
    train_loader = DataLoader(train_data, batch_size, True, num_workers=8)
    val_data = FoamDataset('data/unseen', n_internal, n_boundary, 'data/standard', rng=rng)
    val_loader = DataLoader(val_data, batch_size, False, num_workers=8, pin_memory=True)

    model = Pipn(n_internal, n_boundary)

    checkpoint_callback = ModelCheckpoint(filename='checkpoint-{epoch:d}', every_n_epochs=500, save_top_k=-1)

    trainer = L.Trainer(max_epochs=epochs,
                        callbacks=[RichProgressBar(), checkpoint_callback],
                        log_every_n_steps=int(len(train_data) / batch_size),
                        precision=args.precision,
                        default_root_dir=args.logs_dir)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    trainer.save_checkpoint(f'{trainer.log_dir}/model.ckpt')
