import torch
from torch import nn, Tensor, autograd
from torch.nn.functional import mse_loss, l1_loss
import lightning as L
from torch_cluster import radius
from torch_geometric.nn import global_max_pool, PointNetConv, fps, knn_interpolate, MLP

from foam_dataset import PdeData, FoamData, StandardScaler, FoamDataset
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


class FeaturePropagation(torch.nn.Module):
    def __init__(self, k: int, mlp):
        super().__init__()
        self.k = k
        self.mlp = mlp

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.mlp(x)
        return x, pos_skip, batch_skip


class PointNetPP(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SetAbstraction(0.5, 0.2, MLP([8 + 2, 64, 128], act=nn.Tanh(), norm=None))
        self.conv2 = SetAbstraction(0.25, 0.4, MLP([128 + 2, 128, 256], act=nn.Tanh(), norm=None))
        self.conv3 = GlobalSetAbstraction(MLP([256 + 2, 256, 1024], act=nn.Tanh(), norm=None))

        self.propagate3 = FeaturePropagation(4, MLP([1024 + 256, 256], act=nn.Tanh(), norm=None))
        self.propagate2 = FeaturePropagation(8, MLP([256 + 128, 128], act=nn.Tanh(), norm=None))
        self.propagate1 = FeaturePropagation(16, MLP([128 + 8, 128, 128, 3], act=nn.Tanh(), norm=None))

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tensor:
        x = torch.cat([x, pos], dim=1)
        in_f = (x, pos, batch)
        abs1 = self.conv1(*in_f)
        abs2 = self.conv2(*abs1)
        abs3 = self.conv3(*abs2)

        prop3 = self.propagate3(*abs3, *abs2)
        prop2 = self.propagate2(*prop3, *abs1)
        out, _, _ = self.propagate1(*prop2, *in_f)
        return out


class Pipn(L.LightningModule):
    def __init__(self, n_internal: int, n_boundary: int, scalers: dict[str, StandardScaler]):
        super().__init__()
        self.save_hyperparameters()
        self.n_internal = n_internal
        self.n_boundary = n_boundary
        self.pointnet_pp = PointNetPP()
        self.mu = 0.01  # As rho=1 mu and nu are the same
        self.d = 1000
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

        self.momentum_x_loss = MomentumLoss(0, 1, self.mu, self.d, n_internal,
                                            self.u_scaler, self.points_scaler, self.p_scaler)
        self.momentum_y_loss = MomentumLoss(1, 0, self.mu, self.d, n_internal,
                                            self.u_scaler, self.points_scaler, self.p_scaler)
        self.continuity_loss = ContinuityLoss(n_internal, self.u_scaler, self.points_scaler)
        self.boundary_loss = BoundaryLoss(n_internal)
        self.verbose_predict = False

    def forward(self, x: Tensor, pos: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        return self.pointnet_pp(x, pos, batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001, eps=1e-4)

    def calculate_gradients(self, outputs: Tensor, inputs: Tensor) -> Tensor:
        return autograd.grad(outputs, inputs,
                             grad_outputs=torch.ones_like(outputs),
                             retain_graph=True, create_graph=True)[0]

    def differentiate_field(self, points, ui: Tensor, i: int, j: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        d_ui = self.calculate_gradients(ui, points)
        d_ui_i, d_ui_j = d_ui[..., i:i + 1], d_ui[..., j:j + 1]
        dd_ui_i = self.calculate_gradients(d_ui_i, points)[..., i:i + 1]
        dd_ui_j = self.calculate_gradients(d_ui_j, points)[..., j:j + 1]
        return d_ui_i, d_ui_j, dd_ui_i, dd_ui_j

    def training_step(self, in_data: FoamData, batch_idx: int):
        in_data.pos.requires_grad = True

        pred = self.forward(in_data.x, in_data.pos, in_data.edge_index, in_data.batch)
        pred_data = PdeData(pred)
        # i=0 is x, j=1 is y
        d_ux_x, d_ux_y, dd_ux_x, dd_ux_y = self.differentiate_field(in_data.pos, pred_data.ux, 0, 1)
        # i=1 is y, j=0 is x
        d_uy_y, d_uy_x, dd_uy_y, dd_uy_x = self.differentiate_field(in_data.pos, pred_data.uy, 1, 0)

        d_p = self.calculate_gradients(pred_data.p, in_data.pos)
        d_p_x, d_p_y = d_p[..., 0:1], d_p[..., 1:2]

        obs_ux_loss = mse_loss(pred_data.ux[in_data.obs_index, :], in_data.pde.ux[in_data.obs_index, :])
        obs_uy_loss = mse_loss(pred_data.uy[in_data.obs_index, :], in_data.pde.uy[in_data.obs_index, :])
        obs_p_loss = mse_loss(pred_data.p[in_data.obs_index, :], in_data.pde.p[in_data.obs_index, :])

        boundary_p_loss = self.boundary_loss(pred_data.p, in_data.pde.p)
        boundary_ux_loss = self.boundary_loss(pred_data.ux, in_data.pde.ux)
        boundary_uy_loss = self.boundary_loss(pred_data.uy, in_data.pde.uy)

        cont_loss = self.continuity_loss(d_ux_x, d_uy_y)
        mom_loss_x = self.momentum_x_loss(pred_data.ux, d_ux_x, d_ux_y, pred_data.uy, dd_ux_x, dd_ux_y, d_p_x,
                                          in_data.zones_ids[..., 1:2])

        mom_loss_y = self.momentum_y_loss(pred_data.uy, d_uy_y, d_uy_x, pred_data.ux, dd_uy_y, dd_uy_x, d_p_y,
                                          in_data.zones_ids[..., 1:2])

        loss = (cont_loss +
                mom_loss_x +
                mom_loss_y +
                boundary_p_loss +
                boundary_ux_loss +
                boundary_uy_loss +
                obs_p_loss * self.d +
                obs_ux_loss * self.d +
                obs_uy_loss * self.d)

        self.training_loss_togger.log(len(in_data.batch), loss,
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

    def validation_step(self, in_data: FoamData):
        pred = self.forward(in_data.x, in_data.pos, in_data.edge_index, in_data.batch)
        pred_data = PdeData(pred)
        p_error = l1_loss(self.p_scaler.inverse_transform(pred_data.p),
                          self.p_scaler.inverse_transform(in_data.pde.p))
        ux_error = l1_loss(self.u_scaler[0].inverse_transform(pred_data.ux),
                           self.u_scaler[0].inverse_transform(in_data.pde.ux))
        uy_error = l1_loss(self.u_scaler[1].inverse_transform(pred_data.uy),
                           self.u_scaler[1].inverse_transform(in_data.pde.uy))
        self.val_loss_logger.log(len(in_data.batch), p_error, ux_error, uy_error)

    def predict_step(self, in_data: FoamData) -> tuple[Tensor, Tensor] | Tensor:
        if self.verbose_predict:
            torch.set_grad_enabled(True)
            in_data.pos.requires_grad = True
            pred = self.forward(in_data.x, in_data.pos, in_data.edge_index, in_data.batch)
            pred_data = PdeData(pred)

            # i=0 is x, j=1 is y
            d_ux_x, d_ux_y, dd_ux_x, dd_ux_y = self.differentiate_field(in_data.points, pred_data.ux, 0, 1)
            # i=1 is y, j=0 is x
            d_uy_y, d_uy_x, dd_uy_y, dd_uy_x = self.differentiate_field(in_data.points, pred_data.uy, 1, 0)
            d_p = self.calculate_gradients(pred_data.p, in_data.points)
            d_p_x, d_p_y = d_p[..., 0:1], d_p[..., 1:2]

            momentum_x = self.momentum_x_loss.f(pred_data.ux, d_ux_x, d_ux_y, pred_data.uy, dd_ux_x, dd_ux_y, d_p_x,
                                                in_data.zones_ids)
            momentum_y = self.momentum_y_loss.f(pred_data.uy, d_uy_y, d_uy_x, pred_data.ux, dd_uy_y, dd_uy_x, d_p_y,
                                                in_data.zones_ids)
            cont = self.continuity_loss.f(d_ux_x, d_uy_y)

            return pred_data.data, torch.cat([momentum_x, momentum_y, cont], dim=2)
        else:
            return self.forward(in_data.x, in_data.pos, in_data.edge_index, in_data.batch)
