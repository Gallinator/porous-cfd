import torch
from torch import nn, Tensor
from torch.nn.functional import l1_loss, mse_loss
from torch.optim.lr_scheduler import ExponentialLR

from dataset.foam_data import FoamData
from models.losses import LossLogger, MomentumLoss2d, ContinuityLoss
from models.model_base import ModelBase


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.local_feature = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self.global_feature = nn.Sequential(
            nn.Linear(65, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 1024),
            nn.Tanh()
        )

    def forward(self, x: Tensor, zones_ids: Tensor) -> tuple[Tensor, Tensor]:
        local_features = self.local_feature(x)
        global_feature = self.global_feature(torch.concatenate([local_features, zones_ids], dim=2))
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


class PipnPorous(ModelBase):
    def __init__(self):
        super().__init__(0.01)
        self.save_hyperparameters()
        self.encoder = Encoder()
        self.decoder = Decoder(3)
        self.d = 50
        self.f = 1
        self.training_loss_togger = LossLogger(self, 'Train loss',
                                               'Train loss continuity',
                                               'Train loss momentum x',
                                               'Train loss momentum y',
                                               'Train loss p',
                                               'Train loss ux',
                                               'Train loss uy',
                                               'Train p error',
                                               'Train ux error',
                                               'Train uy error')
        self.val_loss_logger = LossLogger(self, 'Val error p',
                                          'Val error ux',
                                          'Val error uy')
        self.momentum_x_loss = MomentumLoss2d(self.mu, self.d, self.f)
        self.momentum_y_loss = MomentumLoss2d(self.mu, self.d, self.f)
        self.continuity_loss = ContinuityLoss()
        self.verbose_predict = False
        self.pred_labels = {'Ux': None, 'Uy': None, 'p': None, 'U': ['Ux', 'Uy']}

    def forward(self, x: Tensor, zones_ids: Tensor) -> Tensor:
        local_features, global_feature = self.encoder.forward(x, zones_ids)
        exp_global = global_feature.repeat(1, local_features.shape[-2], 1)
        return self.decoder.forward(local_features, exp_global)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, eps=1e-6)
        scheduler = ExponentialLR(optimizer, 0.9995)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def differentiate_field(self, points, ui: Tensor, i: int, j: int) -> tuple:
        d_ui = self.calculate_gradients(ui, points)
        d_ui_i, d_ui_j = d_ui[:, :, i:i + 1], d_ui[:, :, j:j + 1]
        dd_ui_i = self.calculate_gradients(d_ui_i, points)[:, :, i:i + 1]
        dd_ui_j = self.calculate_gradients(d_ui_j, points)[:, :, j:j + 1]
        return d_ui_i, (d_ui_j, dd_ui_i, dd_ui_j)

    def training_step(self, batch: FoamData):
        internal_points, all_points = self.enable_internal_autograd(batch)
        pred = self.forward(all_points, batch['cellToRegion'])
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
        mom_loss_x = self.momentum_x_loss(pred_data['internal']['Ux'], pred_data['internal']['Uy'], d_p_x,
                                          batch['internal']['cellToRegion'], batch['internal']['fx'], d_ux_x,
                                          *diff_x)
        mom_loss_y = self.momentum_y_loss(pred_data['internal']['Uy'], pred_data['internal']['Ux'], d_p_y,
                                          batch['internal']['cellToRegion'], batch['internal']['fy'], d_uy_y,
                                          *diff_y)

        loss = (cont_loss +
                mom_loss_x +
                mom_loss_y +
                boundary_p_loss +
                boundary_ux_loss +
                boundary_uy_loss)

        self.training_loss_togger.log(len(batch.data),
                                      loss,
                                      cont_loss,
                                      mom_loss_x,
                                      mom_loss_y,
                                      boundary_p_loss,
                                      boundary_ux_loss,
                                      boundary_uy_loss,
                                      l1_loss(pred_data['p'], batch['p']),
                                      l1_loss(pred_data['Ux'], batch['Ux']),
                                      l1_loss(pred_data['Uy'], batch['Uy']))
        return loss

    def validation_step(self, batch: FoamData):
        pred = self.forward(batch['C'], batch['cellToRegion'])
        pred_data = FoamData(pred, self.pred_labels, batch.domain)

        p_error = l1_loss(pred_data['p'], batch['p'])
        ux_error = l1_loss(pred_data['Ux'], batch['Ux'])
        uy_error = l1_loss(pred_data['Uy'], batch['Uy'])
        self.val_loss_logger.log(len(batch.data), p_error, ux_error, uy_error)

    def predict_step(self, batch: FoamData) -> tuple[Tensor, Tensor] | Tensor:
        if self.verbose_predict:
            torch.set_grad_enabled(True)

            internal_points, all_points = self.enable_internal_autograd(batch)
            pred = self.forward(all_points, batch['cellToRegion'])
            pred_data = FoamData(pred, self.pred_labels, batch.domain)

            # i=0 is x, j=1 is y
            d_ux_x, diff_x = self.differentiate_field(internal_points, pred_data['Ux'], 0, 1)
            # i=1 is y, j=0 is x
            d_uy_y, diff_y = self.differentiate_field(internal_points, pred_data['Uy'], 1, 0)
            d_p = self.calculate_gradients(pred_data['p'], internal_points)
            d_p_x, d_p_y = d_p[:, :, 0:1], d_p[:, :, 1:2]

            cont = self.continuity_loss.func(d_ux_x, d_uy_y)
            momentum_x = self.momentum_x_loss.func(pred_data['internal']['Ux'],
                                                   pred_data['internal']['Uy'],
                                                   d_p_x,
                                                   batch['internal']['cellToRegion'],
                                                   batch['internal']['fx'],
                                                   d_ux_x,
                                                   *diff_x)
            momentum_y = self.momentum_y_loss.func(pred_data['internal']['Uy'],
                                                   pred_data['internal']['Ux'],
                                                   d_p_y, batch['internal']['cellToRegion'],
                                                   batch['internal']['fy'],
                                                   d_uy_y,
                                                   *diff_y)
            torch.set_grad_enabled(True)
            return pred_data.data, torch.cat([momentum_x, momentum_y, cont], dim=2)
        else:
            return self.forward(batch['C'], batch['cellToRegion'])
