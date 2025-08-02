import torch
from torch import nn, Tensor, autograd
from torch.nn.functional import mse_loss, l1_loss
import lightning as L
from foam_dataset import PdeData, FoamData, StandardScaler
from models.losses import MomentumLoss, ContinuityLoss, LossLogger, BoundaryLoss


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.local_feature = nn.Sequential(
            nn.Linear(4, 64),
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
        x = torch.cat([x, zones_ids], dim=2)
        local_features = self.local_feature(x)
        local_features = torch.concatenate([local_features, zones_ids], dim=2)
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
        x = torch.concatenate([local_features, global_feature], 2)
        return self.decoder(x)


class Pipn(L.LightningModule):
    def __init__(self, n_internal: int, n_boundary: int, scalers: dict[str, StandardScaler]):
        super().__init__()
        self.save_hyperparameters()
        self.n_internal = n_internal
        self.n_boundary = n_boundary
        self.encoder = Encoder()
        self.decoder = Decoder(4)
        self.mu = 1489.4e-6  # As rho=1 mu and nu are the same
        self.d = 14000
        self.training_loss_togger = LossLogger(self, 'Train loss',
                                               'Train loss continuity',
                                               'Train loss momentum x',
                                               'Train loss momentum y',
                                               'Train loss momentum z',
                                               'Train loss p',
                                               'Train loss ux',
                                               'Train loss uy',
                                               'Train loss uz',
                                               'Obs loss p',
                                               'Obs loss ux',
                                               'Obs loss uy',
                                               'Obs loss uz',
                                               'Train p error',
                                               'Train ux error',
                                               'Train uy error',
                                               'Train uz error')
        self.val_loss_logger = LossLogger(self, 'Val error p',
                                          'Val error ux',
                                          'Val error uy',
                                          'Val error uz')

        self.u_scaler = scalers['U']
        self.p_scaler = scalers['p']
        self.points_scaler = scalers['Points']

        self.momentum_x_loss = MomentumLoss(0, 1, self.mu, self.d, n_internal,
                                            self.u_scaler, self.points_scaler, self.p_scaler)
        self.momentum_y_loss = MomentumLoss(1, 0, self.mu, self.d, n_internal,
                                            self.u_scaler, self.points_scaler, self.p_scaler)
        self.continuity_loss = ContinuityLoss(n_internal, self.u_scaler, self.points_scaler)
        self.boundary_loss = BoundaryLoss(n_internal)
        self.verbose_predict = False

    def forward(self, x: Tensor, zones_ids: Tensor) -> Tensor:
        local_features, global_feature = self.encoder.forward(x, zones_ids)
        exp_global = global_feature.repeat(1, local_features.shape[-2], 1)
        return self.decoder.forward(local_features, exp_global)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001, eps=1e-4)

    def calculate_gradients(self, outputs: Tensor, inputs: Tensor) -> Tensor:
        return autograd.grad(outputs, inputs,
                             grad_outputs=torch.ones_like(outputs),
                             retain_graph=True, create_graph=True)[0]

    def differentiate_field(self, points, ui: Tensor, i: int, j: int, k: int) -> tuple:
        d_ui = self.calculate_gradients(ui, points)
        d_ui_i, d_ui_j, d_ui_k = d_ui[:, :, i:i + 1], d_ui[:, :, j:j + 1], d_ui[:, :, k:k + 1]
        dd_ui_i = self.calculate_gradients(d_ui_i, points)[:, :, i:i + 1]
        dd_ui_j = self.calculate_gradients(d_ui_j, points)[:, :, j:j + 1]
        dd_ui_k = self.calculate_gradients(d_ui_j, points)[:, :, k:k + 1]
        return d_ui_i, (d_ui_j, d_ui_k, dd_ui_i, dd_ui_j, dd_ui_k)

    def training_step(self, batch: list, batch_idx: int):
        in_data = FoamData(batch)
        in_data.points.requires_grad = True

        pred = self.forward(in_data.points, in_data.zones_ids)
        pred_data = PdeData(pred)
        # i=0 is x, j=1 is y
        d_ux_x, d_ux_y, dd_ux_x, dd_ux_y = self.differentiate_field(in_data.points, pred_data.ux, 0, 1)
        # i=1 is y, j=0 is x
        d_uy_y, d_uy_x, dd_uy_y, dd_uy_x = self.differentiate_field(in_data.points, pred_data.uy, 1, 0)

        d_p = self.calculate_gradients(pred_data.p, in_data.points)
        d_p_x, d_p_y, d_p_z = d_p[..., 0:1], d_p[..., 1:2], d_p[..., 2:3]

        obs_ux_loss = mse_loss(pred_data.ux.gather(1, in_data.obs_samples), in_data.obs_ux)
        obs_uy_loss = mse_loss(pred_data.uy.gather(1, in_data.obs_samples), in_data.obs_uy)
        obs_uz_loss = mse_loss(pred_data.uz.gather(1, in_data.obs_samples), in_data.obs_uz)
        obs_p_loss = mse_loss(pred_data.p.gather(1, in_data.obs_samples), in_data.obs_p)

        boundary_p_loss = self.boundary_loss(pred_data.p, in_data.pde.p)
        boundary_ux_loss = self.boundary_loss(pred_data.ux, in_data.pde.ux)
        boundary_uy_loss = self.boundary_loss(pred_data.uy, in_data.pde.uy)
        boundary_uz_loss = self.boundary_loss(pred_data.uz, in_data.pde.uz)

        cont_loss = self.continuity_loss(d_ux_x, d_uy_y)
        mom_loss_x = self.momentum_x_loss(pred_data.ux, d_ux_x, d_ux_y, pred_data.uy, dd_ux_x, dd_ux_y, d_p_x,
                                          in_data.zones_ids)

        mom_loss_y = self.momentum_y_loss(pred_data.uy, d_uy_y, d_uy_x, pred_data.ux, dd_uy_y, dd_uy_x, d_p_y,
                                          in_data.zones_ids)

        loss = (cont_loss +
                mom_loss_x +
                mom_loss_y +
                boundary_p_loss +
                boundary_ux_loss +
                boundary_uy_loss +
                boundary_uz_loss +
                obs_p_loss * 1000 +
                obs_ux_loss * 1000 +
                obs_uy_loss * 1000 +
                obs_uz_loss * 1000)

        self.training_loss_togger.log(loss,
                                      cont_loss,
                                      mom_loss_x,
                                      mom_loss_y,
                                      boundary_p_loss,
                                      boundary_ux_loss,
                                      boundary_uy_loss,
                                      boundary_uz_loss,
                                      obs_p_loss,
                                      obs_ux_loss,
                                      obs_uy_loss,
                                      obs_uz_loss,
                                      l1_loss(self.p_scaler.inverse_transform(pred_data.p),
                                              self.p_scaler.inverse_transform(in_data.pde.p)),
                                      l1_loss(self.u_scaler[0].inverse_transform(pred_data.ux),
                                              self.u_scaler[0].inverse_transform(in_data.pde.ux)),
                                      l1_loss(self.u_scaler[1].inverse_transform(pred_data.uy),
                                              self.u_scaler[1].inverse_transform(in_data.pde.uy)),
                                      l1_loss(self.u_scaler[2].inverse_transform(pred_data.uz),
                                              self.u_scaler[2].inverse_transform(in_data.pde.uz)))

        return loss

    def validation_step(self, batch: list):
        batch_data = FoamData(batch)
        pred = self.forward(batch_data.points, batch_data.zones_ids)
        pred_data = PdeData(pred)
        p_error = l1_loss(self.p_scaler.inverse_transform(pred_data.p),
                          self.p_scaler.inverse_transform(batch_data.pde.p))
        ux_error = l1_loss(self.u_scaler[0].inverse_transform(pred_data.ux),
                           self.u_scaler[0].inverse_transform(batch_data.pde.ux))
        uy_error = l1_loss(self.u_scaler[1].inverse_transform(pred_data.uy),
                           self.u_scaler[1].inverse_transform(batch_data.pde.uy))
        uz_error = l1_loss(self.u_scaler[2].inverse_transform(pred_data.uy),
                           self.u_scaler[2].inverse_transform(batch_data.pde.uy))
        self.val_loss_logger.log(p_error, ux_error, uy_error, uz_error)

    def predict_step(self, batch: Tensor) -> tuple[Tensor, Tensor] | Tensor:
        in_data = FoamData(batch)
        if self.verbose_predict:
            torch.set_grad_enabled(True)
            in_data.points.requires_grad = True
            pred = self.forward(in_data.points, in_data.zones_ids)
            pred_data = PdeData(pred)

            # i=0 is x, j=1 is y
            d_ux_x, d_ux_y, dd_ux_x, dd_ux_y = self.differentiate_field(in_data.points, pred_data.ux, 0, 1)
            # i=1 is y, j=0 is x
            d_uy_y, d_uy_x, dd_uy_y, dd_uy_x = self.differentiate_field(in_data.points, pred_data.uy, 1, 0)
            d_p = self.calculate_gradients(pred_data.p, in_data.points)
            d_p_x, d_p_y = d_p[:, :, 0:1], d_p[:, :, 1:2]

            momentum_x = self.momentum_x_loss.f(pred_data.ux, d_ux_x, d_ux_y, pred_data.uy, dd_ux_x, dd_ux_y, d_p_x,
                                                in_data.zones_ids)
            momentum_y = self.momentum_y_loss.f(pred_data.uy, d_uy_y, d_uy_x, pred_data.ux, dd_uy_y, dd_uy_x, d_p_y,
                                                in_data.zones_ids)
            cont = self.continuity_loss.f(d_ux_x, d_uy_y)

            return pred_data.data, torch.cat([momentum_x, momentum_y, cont], dim=2)
        else:
            return self.forward(in_data.points, in_data.zones_ids)
