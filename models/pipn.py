import torch
from torch import nn, Tensor, autograd
from torch.nn.functional import mse_loss, l1_loss
import lightning as L

from foam_dataset import PdeData, FoamData
from models.losses import MomentumLoss, BoundaryLoss, ContinuityLoss, LossLogger


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.local_feature = nn.Sequential(
            nn.Conv1d(2, 64, 1),
            nn.Tanh(),
            nn.Conv1d(64, 64, 1),
            nn.Tanh()
        )

        self.global_feature = nn.Sequential(
            nn.Conv1d(65, 64, 1),
            nn.Tanh(),
            nn.Conv1d(64, 128, 1),
            nn.Tanh(),
            nn.Conv1d(128, 1024, 1),
            nn.Tanh()
        )

    def forward(self, x: Tensor, zones_ids: Tensor) -> tuple[Tensor, Tensor]:
        local_features = self.local_feature(x)
        local_features = torch.concatenate([local_features, zones_ids], dim=1)
        global_feature = self.global_feature(local_features)
        global_feature = torch.max(global_feature, dim=2, keepdim=True)[0]
        return local_features, global_feature


class Decoder(nn.Module):
    def __init__(self, n_pde: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv1d(1089, 512, 1),
            nn.Tanh(),
            nn.Conv1d(512, 256, 1),
            nn.Tanh(),
            nn.Conv1d(256, 128, 1),
            nn.Tanh(),
            nn.Conv1d(128, n_pde, 1)
        )

    def forward(self, local_features: Tensor, global_feature: Tensor) -> Tensor:
        x = torch.concatenate([local_features, global_feature], 1)
        return self.decoder(x)


class Pipn(L.LightningModule):
    def __init__(self, n_internal: int, n_boundary: int):
        super().__init__()
        self.save_hyperparameters()
        self.n_internal = n_internal
        self.n_boundary = n_boundary
        self.encoder = Encoder()
        self.decoder = Decoder(3)
        self.mu = 0.01
        self.d = 100
        self.momentum_loss = MomentumLoss(self.mu, self.d, n_internal)
        self.continuity_loss = ContinuityLoss(n_internal)
        self.boundary_loss = BoundaryLoss(n_internal)
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

    def forward(self, x: Tensor, porous: Tensor) -> Tensor:
        x = x.transpose(dim0=1, dim1=2)

        local_features, global_feature = self.encoder.forward(x, porous.transpose(dim0=1, dim1=2))

        # Expand global feature
        exp_global = global_feature.repeat(1, 1, local_features.shape[-1])

        pde = self.decoder.forward(local_features, exp_global)
        return pde.transpose(dim0=1, dim1=2)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.002, eps=10e-6)

    def calculate_gradients(self, outputs: Tensor, inputs: Tensor) -> Tensor:
        return autograd.grad(outputs, inputs,
                             grad_outputs=torch.ones_like(outputs),
                             retain_graph=True, create_graph=True)[0]

    def differentiate_field(self, points, ui: Tensor, i: int, j: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        d_ui = self.calculate_gradients(ui, points)
        d_ui_i, d_ui_j = d_ui[:, :, i:i + 1], d_ui[:, :, j:j + 1]
        dd_ui_i = self.calculate_gradients(d_ui_i, points)[:, :, i:i + 1]
        dd_ui_j = self.calculate_gradients(d_ui_j, points)[:, :, j:j + 1]
        return d_ui_i, d_ui_j, dd_ui_i, dd_ui_j

    def training_step(self, batch: list):
        in_data = FoamData(batch)
        in_data.points.requires_grad = True

        pred = self.forward(in_data.points, in_data.zones_ids)
        pred_data = PdeData(pred)

        # i=0 is x, j=1 is y
        d_ux_x, d_ux_y, dd_ux_x, dd_ux_y = self.differentiate_field(in_data.points, pred_data.ux, 0, 1)
        # i=1 is y, j=0 is x
        d_uy_y, d_uy_x, dd_uy_y, dd_uy_x = self.differentiate_field(in_data.points, pred_data.uy, 1, 0)

        d_p = self.calculate_gradients(pred_data.p, in_data.points)
        d_p_x, d_p_y = d_p[:, :, 0:1], d_p[:, :, 1:2]

        boundary_p_loss = self.boundary_loss(pred_data.p, in_data.pde.p)
        boundary_ux_loss = self.boundary_loss(pred_data.ux, in_data.pde.ux)
        boundary_uy_loss = self.boundary_loss(pred_data.uy, in_data.pde.uy)

        cont_loss = self.continuity_loss(d_ux_x, d_uy_y)
        mom_loss_x = self.momentum_loss(pred_data.ux, d_ux_x, d_ux_y, pred_data.uy, dd_ux_x, dd_ux_y, d_p_x,
                                        in_data.fx, in_data.zones_ids)
        mom_loss_y = self.momentum_loss(pred_data.uy, d_uy_y, d_uy_x, pred_data.ux, dd_uy_y, dd_uy_x, d_p_y,
                                        in_data.fy, in_data.zones_ids)

        loss = (boundary_p_loss + boundary_ux_loss + boundary_uy_loss + cont_loss + mom_loss_x + mom_loss_y) / 5.0

        self.training_loss_togger.log(loss,
                                      cont_loss,
                                      mom_loss_x,
                                      mom_loss_y,
                                      boundary_p_loss,
                                      boundary_ux_loss,
                                      boundary_uy_loss,
                                      l1_loss(pred_data.p, in_data.pde.p),
                                      l1_loss(pred_data.ux, in_data.pde.ux),
                                      l1_loss(pred_data.uy, in_data.pde.uy))

        return loss

    def validation_step(self, batch: list):
        batch_data = FoamData(batch)
        pred = self.forward(batch_data.points, batch_data.zones_ids)
        pred_data = PdeData(pred)

        p_error = l1_loss(pred_data.p, batch_data.pde.p)
        ux_error = l1_loss(pred_data.ux, batch_data.pde.ux)
        uy_error = l1_loss(pred_data.uy, batch_data.pde.uy)
        self.val_loss_logger.log(p_error, ux_error, uy_error)

    def predict_step(self, batch: Tensor) -> Tensor:
        batch_data = FoamData(batch)
        return self.forward(batch_data.points, batch_data.zones_ids)
