import argparse
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from lightning import Trainer
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from foam_dataset import FoamDataset, PdeData, FoamData
from models.pi_gano import PiGano
from visualization import plot_fields

CHECKPOINT_PATH = 'lightning_logs/version_63/checkpoints/epoch=1458-step=2918.ckpt'


def build_arg_parser() -> ArgumentParser:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--save-plots', action="store_true",
                            help='save all the inference plots', default=False)
    default_model_path = Path('lightning_logs') / last_model / 'last.ckpt'
    arg_parser.add_argument('--checkpoint', type=str, default=default_model_path)
    arg_parser.add_argument('--data-dir', type=str, default='data/val')
    arg_parser.add_argument('--meta-dir', type=str, default='data/train/raw')
    return arg_parser


model = PiGano.load_from_checkpoint(CHECKPOINT_PATH)

val_data = FoamDataset('data/val_unseen', 1000, 200, 500, 'data/train')
val_loader = DataLoader(val_data, 1, False, num_workers=8, pin_memory=True)

trainer = Trainer(logger=False, enable_checkpointing=False)
pred = trainer.predict(model, dataloaders=val_loader)[1]
pred = PdeData(pred).numpy()

tgt = FoamData(val_data[1]).numpy()

points_scaler = val_data.standard_scaler[0:2]
u_scaler = val_data.standard_scaler[2:4]
p_scaler = val_data.standard_scaler[4]

raw_points = points_scaler.inverse_transform(tgt.points)

plt.interactive(True)
plot_fields('Predicted', raw_points, u_scaler.inverse_transform(pred.u[0]),
            p_scaler.inverse_transform(pred.p[0]), tgt.zones_ids)
plot_fields('Ground truth', raw_points, u_scaler.inverse_transform(tgt.pde.u),
            p_scaler.inverse_transform(tgt.pde.p), tgt.zones_ids)

plt.interactive(False)

u_error = u_scaler.inverse_transform(pred.u[0]) - u_scaler.inverse_transform(tgt.pde.u)
p_error = p_scaler.inverse_transform(pred.p[0]) - p_scaler.inverse_transform(tgt.pde.p)
plot_fields('Absolute error', raw_points, np.abs(u_error), np.abs(p_error), tgt.zones_ids, plot_streams=False)
