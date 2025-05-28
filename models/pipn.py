import torch
from torch import nn, Tensor, autograd
from torch.nn.functional import mse_loss, l1_loss
from torchinfo import summary
import lightning as L

from foam_dataset import PredictedDataBatch, FoamDataBatch


class Encoder(nn.Module):
    def __init__(self, n_points: int):
        super().__init__()
        self.n_points = n_points
        self.local_feature = nn.Sequential(
            nn.Conv1d(2, 64, 1),
            nn.Tanh(),
            nn.Conv1d(64, 64, 1),
            nn.Tanh(),
        )

        self.global_feature = nn.Sequential(
            nn.Conv1d(65, 64, 1),
            nn.Tanh(),
            nn.Conv1d(64, 128, 1),
            nn.Tanh(),
            nn.Conv1d(128, 1024, 1),
            nn.Tanh()
        )

    def forward(self, x: Tensor, porous: Tensor) -> tuple[Tensor, Tensor]:
        local_features = self.local_feature(x)
        local_features = torch.concatenate([local_features, porous], dim=1)
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
        self.n_points = n_internal + n_boundary
        self.encoder = Encoder(self.n_points)
        self.decoder = Decoder(3)

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(dim0=1, dim1=2)

        local_features, global_feature = self.encoder.forward(x, porous.transpose(dim0=1, dim1=2))

        # Expand global feature
        exp_global = global_feature.repeat(1, 1, self.n_points)

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

    def split_field(self, field: Tensor, region: str) -> Tensor:
        if region == 'internal':
            return field[:, 0:self.n_internal, :]
        elif region == 'boundary':
            return field[:, self.n_internal:, :]
        raise NotImplementedError(f'{region} is not supported!')

    def field_loss(self, pred_field: Tensor, tgt_field: Tensor):
        return mse_loss(self.split_field(pred_field, 'boundary'),
                        self.split_field(tgt_field, 'boundary'), reduction='sum')

    def continuity_loss(self, d_ux_x: Tensor, d_uy_y: Tensor) -> Tensor:
        pde = d_ux_x + d_uy_y
        pde = self.split_field(pde, 'internal')
        return mse_loss(pde, torch.zeros_like(pde), reduction='sum')

    def momentum_loss(self, ui, d_ui_i, d_ui_j, uj, dd_ui_i, dd_ui_j, d_p_i, f_i):
        pde = d_ui_i * ui + d_ui_j * uj - 0.01 * (dd_ui_i + dd_ui_j) + d_p_i - f_i
        pde = self.split_field(pde, 'internal')
        return mse_loss(pde, torch.zeros_like(pde), reduction='sum')

    def training_step(self, batch: list):
        batch_data = FoamDataBatch(batch)
        batch_data.points.requires_grad = True

        pred = self.forward(points)
        pred_ux, pred_uy, pred_p = pred[:, :, 0:1], pred[:, :, 1:2], pred[:, :, 2:3]

        # i=0 is x, j=1 is y
        d_ux_x, d_ux_y, dd_ux_x, dd_ux_y = self.differentiate_field(batch_data.points, pred.ux, 0, 1)
        # i=1 is y, j=0 is x
        d_uy_y, d_uy_x, dd_uy_y, dd_uy_x = self.differentiate_field(batch_data.points, pred.uy, 1, 0)

        d_p = self.calculate_gradients(pred.p, batch_data.points)
        d_p_x, d_p_y = d_p[:, :, 0:1], d_p[:, :, 1:2]

        p_loss = self.field_loss(pred.p, batch_data.p)
        uy_loss = self.field_loss(pred.uy, batch_data.uy)
        ux_loss = self.field_loss(pred.ux, batch_data.ux)

        cont_loss = self.continuity_loss(d_ux_x, d_uy_y)
        mom_loss_x = self.momentum_loss(pred.ux, d_ux_x, d_ux_y, pred.uy, dd_ux_x, dd_ux_y, d_p_x,
                                        batch_data.fx, batch_data.porous_zone)
        mom_loss_y = self.momentum_loss(pred.uy, d_uy_y, d_uy_x, pred.ux, dd_uy_y, dd_uy_x, d_p_y,
                                        batch_data.fy, batch_data.porous_zone)

        loss = (p_loss + ux_loss + uy_loss + cont_loss + mom_loss_x + mom_loss_y) / 5.0

        self.log("Train loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("Train loss p", p_loss, on_step=False, on_epoch=True)
        self.log("Train loss ux", ux_loss, on_step=False, on_epoch=True)
        self.log("Train loss uy", uy_loss, on_step=False, on_epoch=True)
        self.log("Train loss continuity", cont_loss, on_step=False, on_epoch=True)
        self.log("Train loss momentum x", mom_loss_x, on_step=False, on_epoch=True)
        self.log("Train loss momentum y", mom_loss_y, on_step=False, on_epoch=True)

        self.log("Train ux error", l1_loss(pred.ux, batch_data.ux), on_step=False, on_epoch=True)
        self.log("Train uy error", l1_loss(pred.uy, batch_data.uy), on_step=False, on_epoch=True)
        self.log("Train p error", l1_loss(pred.p, batch_data.p), on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: list):
        batch_data = FoamDataBatch(batch)
        pred = self.forward(batch_data.points, batch_data.porous_zone)

        p_loss = l1_loss(pred.p, batch_data.p)
        ux_loss = l1_loss(pred.ux, batch_data.ux)
        uy_loss = l1_loss(pred.uy, batch_data.uy)
        self.log("Val error p", p_loss, on_step=False, on_epoch=True)
        self.log("Val error ux", ux_loss, on_step=False, on_epoch=True)
        self.log("Val error uy", uy_loss, on_step=False, on_epoch=True)

    def predict_step(self, batch: Tensor) -> PredictedDataBatch:
        batch_data = FoamDataBatch(batch)
        return self.forward(batch_data.points, batch_data.porous_zone)
