import argparse
import json
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from lightning.pytorch.callbacks import RichProgressBar, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
import lightning as L

from dataset.foam_dataset import collate_fn
from models.model_base import PorousPinnBase


def get_log_steps(n_data, batch_size):
    return (n_data // batch_size) + min(1, n_data % batch_size)


def build_arg_parser() -> ArgumentParser:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--n-internal', type=int,
                            help='number of internal points to sample', default=1000)
    arg_parser.add_argument('--n-boundary', type=int,
                            help='number of internal points to sample', default=200)
    arg_parser.add_argument('--n-observations', type=int,
                            help='number of observation points to sample', default=500)
    arg_parser.add_argument('--batch-size', type=int, default=13)
    arg_parser.add_argument('--precision', type=str, default='bf16-mixed',
                            help='model weight precision. Supports mixed precision')
    arg_parser.add_argument('--epochs', type=int, default=3000)
    arg_parser.add_argument('--logs-dir', type=str, default=os.getcwd(),
                            help='base directory to save mdel weights. By default lightning_logs')
    arg_parser.add_argument('--train-dir', type=str, default='data/train',
                            help='directory containing the training data')
    arg_parser.add_argument('--val-dir', type=str, default='data/val',
                            help='directory containing the validation data')
    arg_parser.add_argument('--model', type=str,
                            help='model type. The available models depend on the experiment')
    arg_parser.add_argument('--name', type=str, default=None,
                            help='experiment name. The results will be saved inside a directory with this name')
    arg_parser.add_argument('--checkpoint', type=str, default=None,
                            help='path of the model weights. Use to finetune an existing model')
    arg_parser.add_argument('--loss-scaler', type=str, default='fixed',
                            help='loss scaler. Currently supports fixed and relobralo')
    return arg_parser


def train(args: Namespace, model: PorousPinnBase, train_data: Dataset, val_data: Dataset):
    """
    Trains the model using the provided model and datasets.

    The training is carried out with PyTorch Lightning and the training parameters are saved into a model_meta.json file inside logs_dir/name/model_meta.json.
    During training a checkpoint is saved every 500 epochs and also when the training ends. The last model weights are saved into logs_dir/name/model.ckpt.
    """
    train_loader = DataLoader(train_data, args.batch_size, True, num_workers=8, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, args.batch_size, False, num_workers=8, pin_memory=True, collate_fn=collate_fn)

    torch.set_float32_matmul_precision('high')
    torch.manual_seed(8421)

    checkpoint_callback = ModelCheckpoint(filename='checkpoint-{epoch:d}', every_n_epochs=500, save_top_k=-1)
    logger = TensorBoardLogger(save_dir='', version=args.name)

    trainer = L.Trainer(max_epochs=args.epochs,
                        callbacks=[RichProgressBar(), checkpoint_callback, LearningRateMonitor()],
                        logger=logger,
                        log_every_n_steps=get_log_steps(len(train_data), args.batch_size),
                        precision=args.precision,
                        default_root_dir=args.logs_dir)

    Path(trainer.log_dir).mkdir(exist_ok=True, parents=True)
    with open(f'{trainer.log_dir}/model_meta.json', 'w') as f:
        model_meta = {'Model type': args.model,
                      'N internal': args.n_internal,
                      'N boundary': args.n_boundary,
                      'N observations': args.n_observations,
                      'Precision': args.precision,
                      'Batch size': args.batch_size}
        f.write(json.dumps(model_meta, indent=4))

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.checkpoint)

    trainer.save_checkpoint(f'{trainer.log_dir}/model.ckpt')
