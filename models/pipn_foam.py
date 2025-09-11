import torch
from torch import nn, Tensor
from torch.nn.functional import mse_loss, l1_loss
from torch.optim.lr_scheduler import ExponentialLR

from dataset.foam_data import FoamData
from dataset.foam_dataset import StandardScaler
from models.losses import MomentumLoss2dScaled, ContinuityLossScaled, LossLogger
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
            nn.Linear(65, 96),
            nn.Tanh(),
            nn.Linear(96, 128),
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
            nn.Dropout(0.05),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Dropout(0.05),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, n_pde)
        )

    def forward(self, local_features: Tensor, global_feature: Tensor) -> Tensor:
        x = torch.concatenate([local_features, global_feature], 2)
        return self.decoder(x)


class PipnFoam(ModelBase):
    def __init__(self, scalers: dict[str, StandardScaler]):
        super().__init__(1489.4e-6)
        self.save_hyperparameters()
        self.encoder = Encoder()
        self.decoder = Decoder(3)
        self.d = 14000
        self.f = 17.11
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

        self.u_scaler = scalers['U']
        self.p_scaler = scalers['p']
        self.points_scaler = scalers['C']

        self.momentum_x_loss = MomentumLoss2dScaled(0, 1, self.mu, self.d, self.f, self.u_scaler, self.points_scaler,
                                                    self.p_scaler)
        self.momentum_y_loss = MomentumLoss2dScaled(1, 0, self.mu, self.d, self.f, self.u_scaler, self.points_scaler,
                                                    self.p_scaler)
        self.continuity_loss = ContinuityLossScaled(self.u_scaler, self.points_scaler)
        self.pred_labels = {'Ux': None, 'Uy': None, 'p': None, 'U': ['Ux', 'Uy']}
        self.verbose_predict=False

    def to(self, *args, **kwargs):
        super().to(*args, *kwargs)
        self.u_scaler.to(*args, *kwargs)
        self.p_scaler.to(*args, *kwargs)
        self.points_scaler.to(*args, *kwargs)
        return self

    def forward(self, x: Tensor, zones_ids: Tensor) -> Tensor:
        local_features, global_feature = self.encoder.forward(x, zones_ids)
        exp_global = global_feature.repeat(1, local_features.shape[-2], 1)
        return self.decoder.forward(local_features, exp_global)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = ExponentialLR(optimizer, 0.999)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def differentiate_field(self, points, ui: Tensor, i: int, j: int) -> tuple:
        d_ui = self.calculate_gradients(ui, points)
        d_ui_i, d_ui_j = d_ui[:, :, i:i + 1], d_ui[:, :, j:j + 1]
        dd_ui_i = self.calculate_gradients(d_ui_i, points)[:, :, i:i + 1]
        dd_ui_j = self.calculate_gradients(d_ui_j, points)[:, :, j:j + 1]
        return d_ui_i, (d_ui_j, dd_ui_i, dd_ui_j)

    def training_step(self, batch: FoamData, batch_idx: int):
        internal_points, all_points = self.enable_internal_autograd(batch)

        pred = self.forward(all_points, batch['cellToRegion'])
        pred_data = FoamData(pred, self.pred_labels, batch.domain)
        # i=0 is x, j=1 is y
        d_ux_x, x_diff = self.differentiate_field(internal_points, pred_data['Ux'], 0, 1)
        # i=1 is y, j=0 is x
        d_uy_y, y_diff = self.differentiate_field(internal_points, pred_data['Uy'], 1, 0)

        d_p = self.calculate_gradients(pred_data['internal']['p'], internal_points)
        d_p_x, d_p_y = d_p[:, :, 0:1], d_p[:, :, 1:2]

        obs_ux_loss = mse_loss(pred_data['obs']['Ux'], batch['obs']['Ux'])
        obs_uy_loss = mse_loss(pred_data['obs']['Uy'], batch['obs']['Uy'])
        obs_p_loss = mse_loss(pred_data['obs']['p'], batch['obs']['p'])

        boundary_p_loss = mse_loss(pred_data['boundary']['p'], batch['boundary']['p'])
        boundary_ux_loss = mse_loss(pred_data['boundary']['Ux'], batch['boundary']['Ux'])
        boundary_uy_loss = mse_loss(pred_data['boundary']['Uy'], batch['boundary']['Uy'])

        cont_loss = self.continuity_loss(d_ux_x, d_uy_y)
        mom_loss_x = self.momentum_x_loss(pred_data['internal']['Ux'],
                                          pred_data['internal']['Uy'],
                                          d_p_x,
                                          batch['internal']['cellToRegion'],
                                          d_ux_x,
                                          *x_diff)
        mom_loss_y = self.momentum_y_loss(pred_data['internal']['Uy'],
                                          pred_data['internal']['Ux'],
                                          d_p_y,
                                          batch['internal']['cellToRegion'],
                                          d_uy_y,
                                          *y_diff)
        loss = (cont_loss +
                mom_loss_x +
                mom_loss_y +
                boundary_p_loss +
                boundary_ux_loss +
                boundary_uy_loss +
                obs_p_loss * 100 +
                obs_ux_loss * 100 +
                obs_uy_loss * 100)

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
                                      l1_loss(self.p_scaler.inverse_transform(pred_data['p']),
                                              self.p_scaler.inverse_transform(batch['p'])),
                                      l1_loss(self.u_scaler[0].inverse_transform(pred_data['Ux']),
                                              self.u_scaler[0].inverse_transform(batch['Ux'])),
                                      l1_loss(self.u_scaler[1].inverse_transform(pred_data['Uy']),
                                              self.u_scaler[1].inverse_transform(batch['Uy'])), )

        return loss

    def validation_step(self, batch: FoamData):
        pred = self.forward(batch['C'], batch['cellToRegion'])
        pred_data = FoamData(pred, self.pred_labels, batch.domain)
        p_error = l1_loss(self.p_scaler.inverse_transform(pred_data['p']),
                          self.p_scaler.inverse_transform(batch['p']))
        ux_error = l1_loss(self.u_scaler[0].inverse_transform(pred_data['Ux']),
                           self.u_scaler[0].inverse_transform(batch['Ux']))
        uy_error = l1_loss(self.u_scaler[1].inverse_transform(pred_data['Uy']),
                           self.u_scaler[1].inverse_transform(batch['Uy']))
        self.val_loss_logger.log(len(batch.data), p_error, ux_error, uy_error)

    def predict_step(self, batch: FoamData) -> tuple[Tensor, Tensor] | Tensor:
        if self.verbose_predict:
            torch.set_grad_enabled(True)
            internal_points, all_points = self.enable_internal_autograd(batch)

            pred = self.forward(all_points, batch['cellToRegion'])
            pred_data = FoamData(pred, self.pred_labels, batch.domain)

            # i=0 is x, j=1 is y
            d_ux_x, x_diff = self.differentiate_field(internal_points, pred_data['Ux'], 0, 1)
            # i=1 is y, j=0 is x
            d_uy_y, y_diff = self.differentiate_field(internal_points, pred_data['Uy'], 1, 0)

            d_p = self.calculate_gradients(pred_data['internal']['p'], internal_points)
            d_p_x, d_p_y = d_p[:, :, 0:1], d_p[:, :, 1:2]

            cont = self.continuity_loss.func(d_ux_x, d_uy_y)
            momentum_x = self.momentum_x_loss.func(pred_data['internal']['Ux'],
                                                   pred_data['internal']['Uy'],
                                                   d_p_x,
                                                   batch['internal']['cellToRegion'],
                                                   d_ux_x,
                                                   *x_diff)
            momentum_y = self.momentum_y_loss.func(pred_data['internal']['Uy'],
                                                   pred_data['internal']['Ux'],
                                                   d_p_y,
                                                   batch['internal']['cellToRegion'],
                                                   d_uy_y,
                                                   *y_diff)
            torch.set_grad_enabled(False)
            return pred_data.data, torch.cat([momentum_x, momentum_y, cont], dim=2)
        else:
            return self.forward(batch['C'], batch['cellToRegion'])
