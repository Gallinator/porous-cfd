import time

import numpy as np
from lightning import Trainer
from torch.nn.functional import l1_loss
from torch.utils.data import DataLoader
from data_parser import parse_meta
from foam_dataset import FoamDataset, PdeData
from models.pipn import Pipn, FoamData
from visualization import plot_data_dist, plot_timing, plot_errors, plot_residuals

CHECKPOINT_PATH = 'lightning_logs/version_41_no_tnet_tanh/checkpoints/epoch=1122-step=2246.ckpt'

model = Pipn.load_from_checkpoint(CHECKPOINT_PATH)

val_data = FoamDataset('data/val', 1000, 200, 500, 'data/train')
val_loader = DataLoader(val_data, 2, False, num_workers=8, pin_memory=True)

trainer = Trainer(logger=False, enable_checkpointing=False, inference_mode=False)

start_time = time.perf_counter()
pred = trainer.predict(model, dataloaders=val_loader)
inference_time = time.perf_counter() - start_time
avg_inference_time = inference_time / len(val_data)
val_timing = parse_meta('data/val_unseen')['Timing']
plot_timing([inference_time, val_timing['Total'] / 1e3],
            [avg_inference_time, val_timing['Average'] / 1e3])

errors = []
pde_scaler = val_data.standard_scaler[2:5].to_torch()
for p, t in zip(pred, val_loader):
    tgt_data = FoamData(t)
    error = l1_loss(pde_scaler.inverse_transform(p),
                    pde_scaler.inverse_transform(tgt_data.pde.data), reduction='none')
    errors.extend(error.numpy(force=True))
errors = np.concatenate(errors)
error_data = PdeData(errors)
plot_data_dist('Absolute error distribution', error_data.u, error_data.p, None)

mae = np.average(errors, axis=0)
plot_errors(mae.tolist())