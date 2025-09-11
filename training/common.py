import argparse
import os
from argparse import ArgumentParser

import torch
from lightning.pytorch.callbacks import RichProgressBar, LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
import lightning as L

from dataset.foam_dataset import collate_fn


def build_arg_parser() -> ArgumentParser:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--n-internal', type=int,
                            help='number of internal points to sample', default=1000)
    arg_parser.add_argument('--n-boundary', type=int,
                            help='number of internal points to sample', default=200)
    arg_parser.add_argument('--n-observations', type=int,
                            help='number of observation points to sample', default=500)
    arg_parser.add_argument('--batch-size', type=int, default=13)
    arg_parser.add_argument('--precision', type=str, default='bf16-mixed')
    arg_parser.add_argument('--epochs', type=int, default=3000)
    arg_parser.add_argument('--logs-dir', type=str, default=os.getcwd())
    arg_parser.add_argument('--train-dir', type=str, default='data/train')
    arg_parser.add_argument('--val-dir', type=str, default='data/val')
    arg_parser.add_argument('--model', type=str)
    return arg_parser


def train(args, model, train_data: Dataset, val_data: Dataset):
    train_loader = DataLoader(train_data, args.batch_size, True, num_workers=8, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, args.batch_size, False, num_workers=8, pin_memory=True, collate_fn=collate_fn)

    torch.set_float32_matmul_precision('high')

    checkpoint_callback = ModelCheckpoint(filename='checkpoint-{epoch:d}', every_n_epochs=500, save_top_k=-1)

    trainer = L.Trainer(max_epochs=args.epochs,
                        callbacks=[RichProgressBar(), checkpoint_callback, LearningRateMonitor()],
                        log_every_n_steps=int(len(train_data) / args.batch_size),
                        precision=args.precision,
                        default_root_dir=args.logs_dir)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    trainer.save_checkpoint(f'{trainer.log_dir}/model.ckpt')
