import torch
from lightning import LightningModule
from torch import nn, Tensor, autograd
from torch.nn.functional import mse_loss, l1_loss

from dataset.foam_data import FoamData
from models.losses import ContinuityLoss, LossLogger, MomentumLoss


class Encoder(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.local_feature = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self.global_feature = nn.Sequential(
            nn.Linear(64, 96),
            nn.Tanh(),
            nn.Linear(96, 128),
            nn.Tanh(),
            nn.Linear(128, 1024),
            nn.Tanh()
        )

    def forward(self, pos: Tensor) -> tuple[Tensor, Tensor]:
        local_features = self.local_feature(pos)
        global_feature = self.global_feature(local_features)
        global_feature = torch.max(global_feature, dim=1, keepdim=True)[0]
        return local_features, global_feature


class Decoder(nn.Module):
    def __init__(self, n_pde: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(1088, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, n_pde)
        )

    def forward(self, local_features: Tensor, global_feature: Tensor) -> Tensor:
        x = torch.concatenate([local_features, global_feature], 2)
        return self.decoder(x)


class PipnFluid(LightningModule):
    def __init__(self, n_features: int, n_pde: int, nu=0.001):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = Encoder(n_features)
        self.decoder = Decoder(n_pde)

        self.training_loss_togger = LossLogger(self, 'Train loss',
                                               'Train loss continuity',
                                               'Train loss momentum x',
                                               'Train loss momentum y',
                                               'Train loss p',
                                               'Train loss ux',
                                               'Train loss uy',
                                               'Obs loss p',
                                               'Obs loss ux',
                                               'Obs loss uy',
                                               'Train p error',
                                               'Train ux error',
                                               'Train uy error')
        self.val_loss_logger = LossLogger(self, 'Val error p',
                                          'Val error ux',
                                          'Val error uy')
        self.nu = nu

        self.momentum_x_loss = MomentumLoss(self.nu)
        self.momentum_y_loss = MomentumLoss(self.nu)
        self.continuity_loss = ContinuityLoss()

        self.pred_labels = {'Ux': None, 'Uy': None, 'p': None, 'U': ['Ux', 'Uy']}

    def forward(self, pos: Tensor) -> Tensor:
        local_features, global_feature = self.encoder.forward(pos)
        exp_global = global_feature.repeat(1, local_features.shape[-2], 1)
        y = self.decoder.forward(local_features, exp_global)
        return y

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

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

    def differentiate_field(self, points, ui: Tensor, i: int, j: int) -> tuple:
        d_ui = self.calculate_gradients(ui, points)
        d_ui_i, d_ui_j = d_ui[:, :, i:i + 1], d_ui[:, :, j:j + 1]
        dd_ui_i = self.calculate_gradients(d_ui_i, points)[:, :, i:i + 1]
        dd_ui_j = self.calculate_gradients(d_ui_j, points)[:, :, j:j + 1]
        return d_ui_i, (d_ui_j, dd_ui_i, dd_ui_j)

    def training_step(self, batch: FoamData, batch_idx: int):
        internal_points, all_points = self.enable_internal_autograd(batch)

        pred = self.forward(all_points)
        pred_data = FoamData(pred, self.pred_labels, batch.domain)

        # i=0 is x, j=1 is y
        d_ux_x, diff_x = self.differentiate_field(internal_points, pred_data['internal']['Ux'], 0, 1)
        # i=1 is y, j=0 is x
        d_uy_y, diff_y = self.differentiate_field(internal_points, pred_data['internal']['Uy'], 1, 0)

        d_p = self.calculate_gradients(pred_data['internal']['p'], internal_points)
        d_p_x, d_p_y = d_p[:, :, 0:1], d_p[:, :, 1:2]

        boundary_p_loss = mse_loss(pred_data['boundary']['p'], batch['boundary']['p'])
        boundary_ux_loss = mse_loss(pred_data['boundary']['Ux'], batch['boundary']['Ux'])
        boundary_uy_loss = mse_loss(pred_data['boundary']['Uy'], batch['boundary']['Uy'])

        cont_loss = self.continuity_loss(d_ux_x, d_uy_y)
        mom_loss_x = self.momentum_x_loss(pred_data['Ux'], pred_data['Uy'], d_p_x, d_ux_x, *diff_x)
        mom_loss_y = self.momentum_y_loss(pred_data['Uy'], pred_data['Ux'], d_p_y, d_uy_y, *diff_y)

        obs_ux_loss = mse_loss(pred_data['obs']['Ux'], batch['obs']['Ux'])
        obs_uy_loss = mse_loss(pred_data['obs']['Uy'], batch['obs']['Uy'])
        obs_p_loss = mse_loss(pred_data['obs']['p'], batch['obs']['p'])

        loss = (cont_loss +
                mom_loss_x +
                mom_loss_y +
                boundary_p_loss +
                boundary_ux_loss +
                boundary_uy_loss +
                obs_p_loss +
                obs_ux_loss +
                obs_uy_loss)

        self.training_loss_togger.log(len(batch.data),
                                      loss,
                                      cont_loss,
                                      mom_loss_x,
                                      mom_loss_y,
                                      boundary_p_loss,
                                      boundary_ux_loss,
                                      boundary_uy_loss,
                                      obs_p_loss,
                                      obs_ux_loss,
                                      obs_uy_loss,
                                      l1_loss(pred_data['p'],
                                              batch['p']),
                                      l1_loss(pred_data['Ux'],
                                              batch['Ux']),
                                      l1_loss(pred_data['Uy'],
                                              batch['Uy']))

        return loss

    def validation_step(self, batch: FoamData):
        pred = self.forward(batch['C'])
        pred_data = FoamData(pred, self.pred_labels, batch.domain)

        p_error = l1_loss(pred_data['p'], batch['p'])
        ux_error = l1_loss(pred_data['Ux'], batch['Ux'])
        uy_error = l1_loss(pred_data['Uy'], batch['Uy'])
        self.val_loss_logger.log(p_error, ux_error, uy_error)

    def predict_step(self, batch: FoamData) -> Tensor:
        return self.forward(batch['C'])
