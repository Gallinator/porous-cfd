import numpy as np
import torch
from lightning import Trainer
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from foam_dataset import FoamDataset, PdeData
from models.pipn import Pipn, FoamData
from visualization import plot_fields

CHECKPOINT_PATH = 'lightning_logs/version_22/checkpoints/epoch=402-step=806.ckpt'

model = Pipn.load_from_checkpoint(CHECKPOINT_PATH)

train_data = FoamDataset('data/train', 667, 168)
val_data = FoamDataset('data/val', 667, 168, )
val_loader = DataLoader(val_data, 1, False, num_workers=8, pin_memory=True)

trainer = Trainer(logger=False, enable_checkpointing=False)
pred = trainer.predict(model, dataloaders=val_loader)[0]
pred = PdeData(pred).numpy()

tgt = FoamData(val_data[0]).numpy()

plt.interactive(True)
plot_fields('Predicted', tgt.points, pred.u[0], pred.p[0])
plot_fields('Ground truth', tgt.points, tgt.pde.u, tgt.pde.p)

plt.interactive(False)

u_error = pred.u[0] - tgt.pde.u
p_error = pred.p[0] - tgt.pde.p
plot_fields('Absolute error', tgt.points, np.abs(u_error), np.abs(p_error))
