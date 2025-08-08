import torch
from torch import nn, Tensor, autograd
from torch.nn.functional import mse_loss, l1_loss
import lightning as L
from foam_dataset import PdeData, FoamData, StandardScaler, Normalizer
from models.losses import MomentumLoss, ContinuityLoss, LossLogger


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.local_feature = nn.Sequential(
            nn.Linear(7, 64),
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

    def forward(self, x: Tensor, zones_ids: Tensor, d: Tensor, f: Tensor, inlet_ux: Tensor) -> tuple[Tensor, Tensor]:
        x = torch.cat([x, zones_ids, d, f, inlet_ux], dim=-1)
        local_features = self.local_feature(x)
        local_features = torch.concatenate([local_features, zones_ids], dim=-1)
        global_feature = self.global_feature(local_features)
        global_feature = torch.max(global_feature, dim=1, keepdim=True)[0]
        return local_features, global_feature


class Decoder(nn.Module):
    def __init__(self, n_pde: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(1089, 512),
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
        x = torch.concatenate([local_features, global_feature], -1)
        return self.decoder(x)


class Pipn(L.LightningModule):
    def __init__(self, domain_dict, scalers: dict[str, StandardScaler | Normalizer]):
        super().__init__()
        self.save_hyperparameters()
        self.domain_dict = domain_dict
        self.encoder = Encoder()
        self.decoder = Decoder(3)
        self.mu = 1489.4e-6  # As rho=1 mu and nu are the same
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
        self.points_scaler = scalers['Points']
        self.d_scaler = scalers['d']
        self.f_scaler = scalers['f']

        self.momentum_x_loss = MomentumLoss(0, 1, self.mu, self.u_scaler, self.points_scaler, self.p_scaler,
                                            self.d_scaler, self.f_scaler)
        self.momentum_y_loss = MomentumLoss(1, 0, self.mu, self.u_scaler, self.points_scaler, self.p_scaler,
                                            self.d_scaler, self.f_scaler)
        self.continuity_loss = ContinuityLoss(self.u_scaler, self.points_scaler)
        self.verbose_predict = False

    def forward(self, x: Tensor, porous: Tensor, d: Tensor, f: Tensor, inlet_ux: Tensor) -> Tensor:
        local_features, global_feature = self.encoder.forward(x, porous, d, f, inlet_ux)
        exp_global = global_feature.repeat(1, local_features.shape[-2], 1)
        return self.decoder.forward(local_features, exp_global)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001, eps=1e-4)

    def calculate_gradients(self, outputs: Tensor, inputs: Tensor) -> Tensor:
        return autograd.grad(outputs, inputs,
                             grad_outputs=torch.ones_like(outputs),
                             retain_graph=True, create_graph=True)[0]

    def differentiate_field(self, points, ui: Tensor, i: int, j: int) -> tuple:
        d_ui = self.calculate_gradients(ui, points)
        d_ui_i, d_ui_j = d_ui[:, :, i:i + 1], d_ui[:, :, j:j + 1]
        dd_ui_i = self.calculate_gradients(d_ui_i, points)[:, :, i:i + 1]
        dd_ui_j = self.calculate_gradients(d_ui_j, points)[:, :, j:j + 1]
        return d_ui_i, (d_ui_j, dd_ui_i, dd_ui_j)

    def training_step(self, batch: list, batch_idx: int):
        in_data = FoamData(batch, self.domain_dict)

        internal_points = in_data['internal'].points
        internal_points.requires_grad = True
        in_points = torch.cat([internal_points, in_data['boundary'].points], dim=-2)

        pred = self.forward(in_points, in_data.zones_ids, in_data.d, in_data.f, in_data.inlet_ux)
        pred_data = PdeData(pred, self.domain_dict)

        # i=0 is x, j=1 is y
        d_ux_x, x_diff = self.differentiate_field(internal_points, pred_data['internal'].ux, 0, 1)
        # i=1 is y, j=0 is x
        d_uy_y, y_diff = self.differentiate_field(internal_points, pred_data['internal'].uy, 1, 0)

        d_p = self.calculate_gradients(pred_data['internal'].p, internal_points)
        d_p_x, d_p_y = d_p[:, :, 0:1], d_p[:, :, 1:2]

        obs_ux_loss = mse_loss(pred_data.ux.gather(1, in_data.obs_samples[..., 0:1]), in_data.obs.pde.ux)
        obs_uy_loss = mse_loss(pred_data.uy.gather(1, in_data.obs_samples[..., 0:1]), in_data.obs.pde.uy)
        obs_p_loss = mse_loss(pred_data.p.gather(1, in_data.obs_samples[..., 0:1]), in_data.obs.pde.p)

        boundary_p_loss = mse_loss(pred_data['boundary'].p, in_data['boundary'].pde.p)
        boundary_ux_loss = mse_loss(pred_data['boundary'].ux, in_data['boundary'].pde.ux)
        boundary_uy_loss = mse_loss(pred_data['boundary'].uy, in_data['boundary'].pde.uy)

        cont_loss = self.continuity_loss(d_ux_x, d_uy_y)
        mom_loss_x = self.momentum_x_loss(pred_data['internal'].ux,
                                          pred_data['internal'].uy,
                                          d_p_x,
                                          in_data['internal'].zones_ids,
                                          in_data['internal'].d,
                                          d_ux_x,
                                          *x_diff)

        mom_loss_y = self.momentum_y_loss(pred_data['internal'].uy,
                                          pred_data['internal'].ux,
                                          d_p_y,
                                          in_data['internal'].zones_ids,
                                          in_data['internal'].d,
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

        self.training_loss_togger.log(loss,
                                      cont_loss,
                                      mom_loss_x,
                                      mom_loss_y,
                                      boundary_p_loss,
                                      boundary_ux_loss,
                                      boundary_uy_loss,
                                      obs_p_loss,
                                      obs_ux_loss,
                                      obs_uy_loss,
                                      l1_loss(self.p_scaler.inverse_transform(pred_data.p),
                                              self.p_scaler.inverse_transform(in_data.pde.p)),
                                      l1_loss(self.u_scaler[0].inverse_transform(pred_data.ux),
                                              self.u_scaler[0].inverse_transform(in_data.pde.ux)),
                                      l1_loss(self.u_scaler[1].inverse_transform(pred_data.uy),
                                              self.u_scaler[1].inverse_transform(in_data.pde.uy)))

        return loss

    def validation_step(self, batch: list):
        batch_data = FoamData(batch)
        pred = self.forward(batch_data.points, batch_data.zones_ids, batch_data.d, batch_data.f, batch_data.inlet_ux)
        pred_data = PdeData(pred)
        p_error = l1_loss(self.p_scaler.inverse_transform(pred_data.p),
                          self.p_scaler.inverse_transform(batch_data.pde.p))
        ux_error = l1_loss(self.u_scaler[0].inverse_transform(pred_data.ux),
                           self.u_scaler[0].inverse_transform(batch_data.pde.ux))
        uy_error = l1_loss(self.u_scaler[1].inverse_transform(pred_data.uy),
                           self.u_scaler[1].inverse_transform(batch_data.pde.uy))
        self.val_loss_logger.log(p_error, ux_error, uy_error)

    def predict_step(self, batch: Tensor) -> tuple[Tensor, Tensor] | Tensor:
        in_data = FoamData(batch, self.domain_dict)
        if self.verbose_predict:
            torch.set_grad_enabled(True)
            internal_points = in_data['internal'].points
            internal_points.requires_grad = True
            in_points = torch.cat([internal_points, in_data['boundary'].points], dim=-2)

            pred = self.forward(in_points, in_data.zones_ids, in_data.d, in_data.f, in_data.inlet_ux)
            pred_data = PdeData(pred, self.domain_dict)

            # i=0 is x, j=1 is y
            d_ux_x, x_diff = self.differentiate_field(internal_points, pred_data['internal'].ux, 0, 1)
            # i=1 is y, j=0 is x
            d_uy_y, y_diff = self.differentiate_field(internal_points, pred_data['internal'].uy, 1, 0)
            d_p = self.calculate_gradients(pred_data['internal'].p, internal_points)
            d_p_x, d_p_y = d_p[:, :, 0:1], d_p[:, :, 1:2]

            momentum_x = self.momentum_x_loss.func(pred_data['internal'].ux,
                                                   pred_data['internal'].uy,
                                                   d_p_x,
                                                   in_data['internal'].zones_ids,
                                                   in_data['internal'].d,
                                                   d_ux_x,
                                                   *x_diff)
            momentum_y = self.momentum_y_loss.func(pred_data['internal'].uy,
                                                   pred_data['internal'].ux,
                                                   d_p_y,
                                                   in_data['internal'].zones_ids,
                                                   in_data['internal'].d,
                                                   d_uy_y,
                                                   *y_diff)
            cont = self.continuity_loss.f(d_ux_x, d_uy_y)

            return pred_data.data, torch.cat([momentum_x, momentum_y, cont], dim=2)
        else:
            return self.forward(in_data.points, in_data.zones_ids, in_data.d, in_data.f, in_data.inlet_ux)
