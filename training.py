from lightning.pytorch.callbacks import RichProgressBar
from torch.utils.data import DataLoader
from foam_dataset import FoamDataset
from models.pipn import Pipn
import lightning as L

BATCH_SIZE = 13
N_INTERNAL = 1000
N_BOUNDARY = 200
N_OBS = 500

train_data = FoamDataset('data/train', N_INTERNAL, N_BOUNDARY, N_OBS)
train_loader = DataLoader(train_data, BATCH_SIZE, True, num_workers=8)
val_data = FoamDataset('data/val', N_INTERNAL, N_BOUNDARY, N_OBS, train_data.meta)
val_loader = DataLoader(val_data, BATCH_SIZE, False, num_workers=8, pin_memory=True)

scalers = {'U': train_data.standard_scaler[2:4],
           'p': train_data.standard_scaler[4],
           'Points': train_data.standard_scaler[0:2]}
model = Pipn(N_INTERNAL, N_BOUNDARY, scalers)

trainer = L.Trainer(max_epochs=-1,
                    callbacks=[RichProgressBar()],
                    log_every_n_steps=2,
                    precision='16-mixed',
                    val_check_interval=2)

trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
