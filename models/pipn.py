import torch
from torch import nn, Tensor, autograd
from torch.nn.functional import mse_loss, l1_loss
import lightning as L
from torch_cluster import fps, radius
from torch_geometric.nn import MLP, global_max_pool, PointNetConv
from torch_geometric.utils import unbatch

from foam_dataset import PdeData, FoamData
from models.losses import MomentumLoss, ContinuityLoss, LossLogger, BoundaryLoss


class SetAbstraction(torch.nn.Module):
    def __init__(self, ratio: float, r: float, mlp):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(mlp)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx])
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv((x, x[idx]), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSetAbstraction(torch.nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.nn = mlp

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 2))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.local_feature = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self.conv1 = SetAbstraction(0.5, 1, MLP([65 + 2, 64], act=nn.Tanh(), norm=None))
        self.conv2 = SetAbstraction(0.25, 0.4, MLP([64 + 2, 128], act=nn.Tanh(), norm=None))
        self.conv3 = GlobalSetAbstraction(MLP([128 + 2, 1024], act=nn.Tanh(), norm=None))

    def forward(self, x: Tensor, zones_ids: Tensor) -> tuple[Tensor, Tensor]:
        local_features = self.local_feature(x)
        global_in = torch.concatenate([local_features, zones_ids], dim=2)
        batch = torch.concatenate([torch.tensor([i] * x.shape[-2]) for i in range(len(x))]).to(device=x.device,
                                                                                               dtype=torch.int64)
        out = self.conv1(torch.concatenate([*global_in]), torch.concatenate([*x]), batch)
        out = self.conv2(*out)
        y, _, batch = self.conv3(*out)
        y = torch.stack(unbatch(y, batch))
        global_feature = torch.max(y, dim=1, keepdim=True)[0]
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


class Pipn(L.LightningModule):
    def __init__(self, n_internal: int, n_boundary: int):
        super().__init__()
        self.save_hyperparameters()
        self.n_internal = n_internal
        self.n_boundary = n_boundary
        self.encoder = Encoder()
        self.decoder = Decoder(3)
        self.mu = 0.01
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
        self.momentum_x_loss = MomentumLoss(self.mu, self.d, self.f, n_internal)
        self.momentum_y_loss = MomentumLoss(self.mu, self.d, self.f, n_internal)
        self.continuity_loss = ContinuityLoss(n_internal)
        self.boundary_loss = BoundaryLoss(n_internal)
        self.verbose_predict = False

    def forward(self, x: Tensor, zones_ids: Tensor) -> Tensor:
        local_features, global_feature = self.encoder.forward(x, zones_ids)
        exp_global = global_feature.repeat(1, local_features.shape[-2], 1)
        return self.decoder.forward(local_features, exp_global)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.002, eps=10e-6)

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

    def training_step(self, batch: list):
        in_data = FoamData(batch)
        in_data.points.requires_grad = True

        pred = self.forward(in_data.points, in_data.zones_ids)
        pred_data = PdeData(pred)

        # i=0 is x, j=1 is y
        d_ux_x, diff_x = self.differentiate_field(in_data.points, pred_data.ux, 0, 1)
        # i=1 is y, j=0 is x
        d_uy_y, diff_y = self.differentiate_field(in_data.points, pred_data.uy, 1, 0)

        d_p = self.calculate_gradients(pred_data.p, in_data.points)
        d_p_x, d_p_y = d_p[:, :, 0:1], d_p[:, :, 1:2]

        boundary_p_loss = self.boundary_loss(pred_data.p, in_data.pde.p)
        boundary_ux_loss = self.boundary_loss(pred_data.ux, in_data.pde.ux)
        boundary_uy_loss = self.boundary_loss(pred_data.uy, in_data.pde.uy)

        cont_loss = self.continuity_loss(d_ux_x, d_uy_y)
        mom_loss_x = self.momentum_x_loss(pred_data.ux, pred_data.uy, d_p_x, in_data.zones_ids, in_data.fx, d_ux_x,
                                          *diff_x)
        mom_loss_y = self.momentum_y_loss(pred_data.uy, pred_data.ux, d_p_y, in_data.zones_ids, in_data.fy, d_uy_y,
                                          *diff_y)

        loss = (cont_loss +
                mom_loss_x +
                mom_loss_y +
                boundary_p_loss +
                boundary_ux_loss +
                boundary_uy_loss)

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

    def predict_step(self, batch: Tensor) -> tuple[Tensor, Tensor] | Tensor:
        in_data = FoamData(batch)
        if self.verbose_predict:
            torch.set_grad_enabled(True)
            in_data.points.requires_grad = True
            pred = self.forward(in_data.points, in_data.zones_ids)
            pred_data = PdeData(pred)

            # i=0 is x, j=1 is y
            d_ux_x, diff_x = self.differentiate_field(in_data.points, pred_data.ux, 0, 1)
            # i=1 is y, j=0 is x
            d_uy_y, diff_y = self.differentiate_field(in_data.points, pred_data.uy, 1, 0)
            d_p = self.calculate_gradients(pred_data.p, in_data.points)
            d_p_x, d_p_y = d_p[:, :, 0:1], d_p[:, :, 1:2]

            cont = self.continuity_loss.func(d_ux_x, d_uy_y)
            momentum_x = self.momentum_x_loss.func(pred_data.ux, pred_data.uy, d_p_x, in_data.zones_ids, in_data.fx, d_ux_x,
                                              *diff_x)
            momentum_y = self.momentum_y_loss.func(pred_data.uy, pred_data.ux, d_p_y, in_data.zones_ids, in_data.fy, d_uy_y,
                                              *diff_y)

            return pred_data.data, torch.cat([momentum_x, momentum_y, cont], dim=2)
        else:
            return self.forward(in_data.points, in_data.zones_ids)
