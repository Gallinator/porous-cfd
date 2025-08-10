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
                            help='number of internal points to sample', default=1000)
    arg_parser.add_argument('--n-boundary', type=int,
                            help='number of internal points to sample', default=200)
    arg_parser.add_argument('--n-observations', type=int,
                            help='number of observation points to sample', default=500)
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

    train_data = FoamDataset('data/train', n_internal, n_boundary, n_obs)
    train_loader = DataLoader(train_data, batch_size, True, num_workers=8)
    val_data = FoamDataset('data/val', n_internal, n_boundary, n_obs, 'data/train')
    val_loader = DataLoader(val_data, batch_size, False, num_workers=8, pin_memory=True)

    scalers = {'U': train_data.standard_scaler[2:4],
               'p': train_data.standard_scaler[4],
               'Points': train_data.standard_scaler[0:2]}
    model = Pipn(n_internal, n_boundary, scalers)

    trainer = L.Trainer(max_epochs=epochs,
                        callbacks=[RichProgressBar()],
                        log_every_n_steps=int(batch_size / len(train_data)),
                        precision=args.precision)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
