import torch
from sympy.physics.units import momentum
from torch import nn, Tensor, autograd
from torch.nn.functional import mse_loss, l1_loss
import lightning as L
from torch.optim.lr_scheduler import ExponentialLR
from torch_cluster import radius
from torch_geometric.nn import global_max_pool, PointNetConv, fps, knn_interpolate, MLP
from torch_geometric.utils import unbatch

from foam_dataset import PdeData, FoamData, StandardScaler, Normalizer
from models.losses import MomentumLoss, ContinuityLoss, LossLogger


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


class EncoderPp(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SetAbstraction(0.5, 0.2, MLP([3 + 2, 64, 64], act=nn.Tanh(), norm=None))
        self.conv2 = SetAbstraction(0.25, 0.4, MLP([64 + 2, 128, 128], act=nn.Tanh(), norm=None))
        self.conv3 = GlobalSetAbstraction(MLP([128 + 2, 256, 256], act=nn.Tanh(), norm=None))

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor):
        x = torch.cat([x, pos], dim=1)
        in_f = (x, pos, batch)
        abs1 = self.conv1(*in_f)
        abs2 = self.conv2(*abs1)
        abs3 = self.conv3(*abs2)
        return abs3, (in_f, abs1, abs2)


class DecoderPp(nn.Module):
    def __init__(self):
        super().__init__()
        self.propagate3 = FeaturePropagation(4, MLP([128 + 256, 128, 128], act=nn.Tanh(), norm=None))
        self.propagate2 = FeaturePropagation(8, MLP([64 + 128, 64, 64], act=nn.Tanh(), norm=None))
        self.propagate1 = FeaturePropagation(16, MLP([64 + 3, 64, 64, 3], act=nn.Tanh(), norm=None))

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor, skip) -> Tensor:
        prop3 = self.propagate3(x, pos, batch, *skip[2])
        prop2 = self.propagate2(*prop3, *skip[1])
        out, _, _ = self.propagate1(*prop2, *skip[0])
        return out


class Branch(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = MLP([7, 256, 256, 256], act=nn.Tanh(), norm=None)

    def forward(self, ceof_points: Tensor, d: Tensor, f: Tensor, inlet_points: Tensor, inlet_ux: Tensor):
        """
        :param ceof_points: Coordinates of darcy boundary points(B, M, 2)
        :param d: Darcy coefficients (B, M, 2)
        :param f: Forchheimer coefficients (B, M, 2)
        :param inlet_points: Coordinates of inlet boundary points(B, N, 2)
        :param inlet_ux: inlet velocity along x (B, N, 1)
        :return: Parameter embedding (B, 1, 64)
        """
        points = torch.cat([ceof_points, inlet_points], dim=-2)
        par_dim = d.shape[-1] + f.shape[-1] + inlet_ux.shape[-1]
        x = torch.zeros((points.shape[0], points.shape[1], par_dim), device=points.device)
        x[..., 0:ceof_points.shape[-2], 0:2] = d
        x[..., 0:ceof_points.shape[-2], 2:4] = f
        x[..., ceof_points.shape[-2]:, 4:5] = inlet_ux
        x = torch.cat([points, x], dim=-1)
        y = self.linear(x)
        return torch.max(y, dim=1, keepdim=True)[0].squeeze()


class NeuralOperator(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.Tanh()
        )
        if dropout:
            self.linear.append(nn.Dropout(0.05))

    def forward(self, x: Tensor, par_embedding: Tensor):
        return self.linear(x) * par_embedding


class PiGanoPP(L.LightningModule):
    def __init__(self, domain_dict: dict, scalers: dict[str, StandardScaler | Normalizer]):
        super().__init__()
        self.save_hyperparameters()
        self.domain_dict = domain_dict
        self.encoder = EncoderPp()
        self.decoder = DecoderPp()
        self.branch = Branch()
        self.neural_op1 = NeuralOperator(256, 256)
        self.neural_op2 = NeuralOperator(256, 256)
        self.neural_op3 = NeuralOperator(256, 256)

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

    def forward(self, x: Tensor,
                pos: Tensor,
                edge_index: Tensor,
                batch: Tensor,
                par_points: Tensor,
                par_d: Tensor,
                par_f: Tensor,
                par_inlet_points: Tensor,
                par_inlet_ux: Tensor) -> Tensor:
        par_embedding = self.branch(par_points, par_d, par_f, par_inlet_points, par_inlet_ux)
        par_embedding = torch.stack([*par_embedding])

        out, skip = self.encoder(x, pos, batch)
        out_x, out_pos, out_batch = out

        out = self.neural_op1(out_x, par_embedding)
        out = self.neural_op2(out, par_embedding)
        out = self.neural_op3(out, par_embedding)

        return self.decoder(out, out_pos, out_batch, skip)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = ExponentialLR(optimizer, 0.999)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def calculate_gradients(self, outputs: Tensor, inputs: Tensor) -> Tensor:
        return autograd.grad(outputs, inputs,
                             grad_outputs=torch.ones_like(outputs),
                             retain_graph=True, create_graph=True)[0]

    def differentiate_field(self, points, ui: Tensor, i: int, j: int) -> tuple:
        d_ui = self.calculate_gradients(ui, points)
        d_ui_i, d_ui_j = d_ui[..., i:i + 1], d_ui[..., j:j + 1]
        dd_ui_i = self.calculate_gradients(d_ui_i, points)[..., i:i + 1]
        dd_ui_j = self.calculate_gradients(d_ui_j, points)[..., j:j + 1]
        return d_ui_i, (d_ui_j, dd_ui_i, dd_ui_j)

    def training_step(self, in_data: FoamData, batch_idx: int):
        in_data.domain_dict = self.domain_dict

        internal_batch = in_data.slice('internal').batch
        internal_pos = in_data.slice('internal').pos
        internal_pos.requires_grad = True
        unb_pos = torch.stack(unbatch(internal_pos, internal_batch))
        boundary_pos = torch.stack(unbatch(in_data.slice('boundary').pos, in_data.slice('boundary').batch))
        in_pos = torch.cat([unb_pos, boundary_pos], dim=-2)
        in_pos = torch.cat([*in_pos])

        pred = self.forward(in_data.zones_ids,
                            in_pos,
                            in_data.edge_index,
                            in_data.batch,
                            torch.stack(unbatch(in_data.slice('internal').pos, internal_batch)),
                            torch.stack(unbatch(in_data.slice('internal').d, internal_batch)),
                            torch.stack(unbatch(in_data.slice('internal').f, internal_batch)),
                            torch.stack(unbatch(in_data.slice('inlet').pos, in_data.slice('inlet').batch)),
                            torch.stack(unbatch(in_data.slice('inlet').inlet_ux, in_data.slice('inlet').batch)))

        pred_data = PdeData(pred, in_data.batch, self.domain_dict)
        # i=0 is x, j=1 is y
        d_ux_x, x_diff = self.differentiate_field(internal_pos, pred_data.slice('internal').ux, 0, 1)
        # i=1 is y, j=0 is x
        d_uy_y, y_diff = self.differentiate_field(internal_pos, pred_data.slice('internal').uy, 1, 0)

        d_p = self.calculate_gradients(pred_data.slice('internal').p, internal_pos)
        d_p_x, d_p_y = d_p[..., 0:1], d_p[..., 1:2]

        obs_ux_loss = mse_loss(pred_data.ux[in_data.obs_index, :], in_data.obs.pde.ux)
        obs_uy_loss = mse_loss(pred_data.uy[in_data.obs_index, :], in_data.obs.pde.uy)
        obs_p_loss = mse_loss(pred_data.p[in_data.obs_index, :], in_data.obs.pde.p)

        boundary_p_loss = mse_loss(pred_data.slice('boundary').p, in_data.slice('boundary').pde.p)
        boundary_ux_loss = mse_loss(pred_data.slice('boundary').ux, in_data.slice('boundary').pde.ux)
        boundary_uy_loss = mse_loss(pred_data.slice('boundary').uy, in_data.slice('boundary').pde.uy)

        cont_loss = self.continuity_loss(d_ux_x, d_uy_y)
        mom_loss_x = self.momentum_x_loss(pred_data.slice('internal').ux,
                                          pred_data.slice('internal').uy,
                                          d_p_x,
                                          in_data.slice('internal').zones_ids,
                                          in_data.slice('internal').d,
                                          in_data.slice('internal').f,
                                          d_ux_x,
                                          *x_diff)

        mom_loss_y = self.momentum_y_loss(pred_data.slice('internal').uy,
                                          pred_data.slice('internal').ux,
                                          d_p_y,
                                          in_data.slice('internal').zones_ids,
                                          in_data.slice('internal').d,
                                          in_data.slice('internal').f,
                                          d_uy_y,
                                          *y_diff)

        loss = (cont_loss +
                mom_loss_x +
                mom_loss_y +
                boundary_p_loss +
                boundary_ux_loss +
                boundary_uy_loss +
                obs_p_loss * 1000 +
                obs_ux_loss * 1000 +
                obs_uy_loss * 1000)

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
        in_data.domain_dict = self.domain_dict

        internal_batch = in_data.slice('internal').batch
        pred = self.forward(in_data.zones_ids,
                            in_data.pos,
                            in_data.edge_index,
                            in_data.batch,
                            torch.stack(unbatch(in_data.slice('internal').pos, internal_batch)),
                            torch.stack(unbatch(in_data.slice('internal').d, internal_batch)),
                            torch.stack(unbatch(in_data.slice('internal').f, internal_batch)),
                            torch.stack(unbatch(in_data.slice('inlet').pos, in_data.slice('inlet').batch)),
                            torch.stack(unbatch(in_data.slice('inlet').inlet_ux, in_data.slice('inlet').batch)))

        pred_data = PdeData(pred, in_data.batch, self.domain_dict)
        p_error = l1_loss(self.p_scaler.inverse_transform(pred_data.p),
                          self.p_scaler.inverse_transform(in_data.pde.p))
        ux_error = l1_loss(self.u_scaler[0].inverse_transform(pred_data.ux),
                           self.u_scaler[0].inverse_transform(in_data.pde.ux))
        uy_error = l1_loss(self.u_scaler[1].inverse_transform(pred_data.uy),
                           self.u_scaler[1].inverse_transform(in_data.pde.uy))
        self.val_loss_logger.log(len(in_data.batch), p_error, ux_error, uy_error)

    def predict_step(self, in_data: FoamData) -> tuple[Tensor, Tensor] | Tensor:
        in_data.domain_dict = self.domain_dict
        internal_batch = in_data.slice('internal').batch

        if self.verbose_predict:
            torch.set_grad_enabled(True)

            internal_pos = in_data.slice('internal').pos
            internal_pos.requires_grad = True
            unb_pos = torch.stack(unbatch(internal_pos, in_data.slice('internal').batch))
            boundary_pos = torch.stack(unbatch(in_data.slice('boundary').pos, in_data.slice('boundary').batch))
            in_pos = torch.cat([unb_pos, boundary_pos], dim=-2)
            in_pos = torch.concatenate([*in_pos])

            pred = self.forward(in_data.zones_ids,
                                in_pos,
                                in_data.edge_index,
                                in_data.batch,
                                torch.stack(unbatch(in_data.slice('internal').pos, internal_batch)),
                                torch.stack(unbatch(in_data.slice('internal').d, internal_batch)),
                                torch.stack(unbatch(in_data.slice('internal').f, internal_batch)),
                                torch.stack(unbatch(in_data.slice('inlet').pos, in_data.slice('inlet').batch)),
                                torch.stack(unbatch(in_data.slice('inlet').inlet_ux, in_data.slice('inlet').batch)))
            pred_data = PdeData(pred, in_data.batch, self.domain_dict)

            # i=0 is x, j=1 is y
            d_ux_x, x_diff = self.differentiate_field(internal_pos, pred_data.slice('internal').ux, 0, 1)
            # i=1 is y, j=0 is x
            d_uy_y, y_diff = self.differentiate_field(internal_pos, pred_data.slice('internal').uy, 1, 0)

            d_p = self.calculate_gradients(pred_data.slice('internal').p, internal_pos)
            d_p_x, d_p_y = d_p[..., 0:1], d_p[..., 1:2]

            cont = self.continuity_loss.f(d_ux_x, d_uy_y)
            momentum_x = self.momentum_x_loss.func(pred_data.slice('internal').ux,
                                                   pred_data.slice('internal').uy,
                                                   d_p_x,
                                                   in_data.slice('internal').zones_ids,
                                                   in_data.slice('internal').d,
                                                   in_data.slice('internal').f,
                                                   d_ux_x,
                                                   *x_diff)

            momentum_y = self.momentum_y_loss.func(pred_data.slice('internal').uy,
                                                   pred_data.slice('internal').ux,
                                                   d_p_y,
                                                   in_data.slice('internal').zones_ids,
                                                   in_data.slice('internal').d,
                                                   in_data.slice('internal').f,
                                                   d_uy_y,
                                                   *y_diff)

            return pred_data.data, torch.cat([momentum_x, momentum_y, cont], dim=-1)
        else:
            return self.forward(in_data.zones_ids,
                                in_data.pos,
                                in_data.edge_index,
                                in_data.batch,
                                torch.stack(unbatch(in_data.slice('internal').pos, internal_batch)),
                                torch.stack(unbatch(in_data.slice('internal').d, internal_batch)),
                                torch.stack(unbatch(in_data.slice('internal').f, internal_batch)),
                                torch.stack(unbatch(in_data.slice('inlet').pos, in_data.slice('inlet').batch)),
                                torch.stack(unbatch(in_data.slice('inlet').inlet_ux, in_data.slice('inlet').batch)))
