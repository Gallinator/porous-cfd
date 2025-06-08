from lightning.pytorch.callbacks import RichProgressBar
from torch.utils.data import DataLoader
from foam_dataset import FoamDataset
from models.pipn import Pipn
import lightning as L

BATCH_SIZE = 13
N_INTERNAL = 1000
N_BOUNDARY = 200

train_data = FoamDataset('data/train', N_INTERNAL, N_BOUNDARY)
train_loader = DataLoader(train_data, BATCH_SIZE, True, num_workers=8)
val_data = FoamDataset('data/val', N_INTERNAL, N_BOUNDARY)
val_loader = DataLoader(val_data, BATCH_SIZE, False, num_workers=8, pin_memory=True)

model = Pipn(N_INTERNAL, N_BOUNDARY)

trainer = L.Trainer(max_epochs=-1,
                    callbacks=[RichProgressBar()],
                    log_every_n_steps=2,
                    precision='16-mixed',
                    val_check_interval=2)

trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
