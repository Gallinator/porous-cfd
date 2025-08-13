import argparse
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from lightning import Trainer
from matplotlib import pyplot as plt
from torch_geometric.loader import DataLoader

from foam_dataset import FoamDataset, PdeData
from models.pipn_pp import PipnPP, FoamData
from visualization import plot_fields

CHECKPOINT_PATH = 'lightning_logs/version_22/checkpoints/epoch=402-step=806.ckpt'


def build_arg_parser() -> ArgumentParser:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--save-plots', action="store_true",
                            help='save all the inference plots', default=False)
    default_model_path = Path('lightning_logs') / last_model / 'last.ckpt'
    arg_parser.add_argument('--checkpoint', type=str, default=default_model_path)
    arg_parser.add_argument('--data-dir', type=str, default='data/val')
    arg_parser.add_argument('--meta-dir', type=str, default='data/train/raw')
    return arg_parser


model = PipnPP.load_from_checkpoint(CHECKPOINT_PATH)

val_data = FoamDataset('data/val_unseen', 1000, 200, 500, 'data/train/raw')
val_loader = DataLoader(val_data, 1, False, num_workers=8, pin_memory=True)

trainer = Trainer(logger=False, enable_checkpointing=False)
pred = trainer.predict(model, dataloaders=val_loader)[0]
pred = PdeData(pred).numpy()

tgt = val_data[0]

points_scaler = val_data.standard_scaler[0:2]
u_scaler = val_data.standard_scaler[2:4]
p_scaler = val_data.standard_scaler[4]

raw_points = points_scaler.inverse_transform(tgt.pos.numpy(force=True))
ids = tgt.zones_ids.numpy(force=True)

plt.interactive(True)
plot_fields('Predicted', raw_points, u_scaler.inverse_transform(pred.u),
            p_scaler.inverse_transform(pred.p), ids)
plot_fields('Ground truth', raw_points, u_scaler.inverse_transform(tgt.pde.u.numpy(force=True)),
            p_scaler.inverse_transform(tgt.pde.p.numpy(force=True)), ids)

plt.interactive(False)

u_error = u_scaler.inverse_transform(pred.u) - u_scaler.inverse_transform(tgt.pde.u.numpy(force=True))
p_error = p_scaler.inverse_transform(pred.p) - p_scaler.inverse_transform(tgt.pde.p.numpy(force=True))
plot_fields('Absolute error', raw_points, np.abs(u_error), np.abs(p_error), ids, plot_streams=False)
