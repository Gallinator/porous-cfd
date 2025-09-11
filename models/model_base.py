import lightning as L
import torch
from torch import Tensor, autograd
from dataset.foam_data import FoamData


class ModelBase(L.LightningModule):
    def __init__(self, mu=0.001):
        super().__init__()
        self.mu = mu

    def transfer_batch_to_device(self, batch: FoamData, device: torch.device, dataloader_idx: int) -> FoamData:
        dev_data = batch.data.to(device)
        dev_domain = {d: s.to(device) for d, s in batch.domain.items()}
        return FoamData(dev_data, batch.labels, dev_domain)

    def calculate_gradients(self, outputs: Tensor, inputs: Tensor) -> Tensor:
        return autograd.grad(outputs, inputs,
                             grad_outputs=torch.ones_like(outputs),
                             retain_graph=True, create_graph=True)[0]

    def enable_internal_autograd(self, batch: FoamData) -> tuple[Tensor, Tensor]:
        internal_points = batch['internal']['C']
        internal_points.requires_grad = True
        return internal_points, torch.cat([internal_points, batch['boundary']['C']], dim=-2)
