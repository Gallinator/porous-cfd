import argparse
from argparse import ArgumentParser

from lightning.pytorch.callbacks import RichProgressBar
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
    return arg_parser


if __name__ == '__main__':
    args = build_arg_parser().parse_args()

    batch_size = args.batch_size
    n_internal = args.n_internal
    n_boundary = args.n_boundary
    n_obs = args.n_observations
    epochs = args.epochs

    train_data = FoamDataset('data/train', n_internal, n_boundary)
    train_loader = DataLoader(train_data, batch_size, True, num_workers=8)
    val_data = FoamDataset('data/val', n_internal, n_boundary)
    val_loader = DataLoader(val_data, batch_size, False, num_workers=8, pin_memory=True)

    model = Pipn(n_internal, n_boundary)

    trainer = L.Trainer(max_epochs=epochs,
                        callbacks=[RichProgressBar()],
                        log_every_n_steps=2,
                        precision=args.precision,
                        val_check_interval=2)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
