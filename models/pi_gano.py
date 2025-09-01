import torch
from torch import nn, autograd, Tensor
import lightning as L
from torch.nn.functional import l1_loss, mse_loss
from torch.optim.lr_scheduler import ExponentialLR
from torchvision.ops import MLP

from foam_dataset import StandardScaler, FoamData, PdeData, Normalizer
from models.losses import MomentumLoss, ContinuityLoss, LossLogger


class Branch(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = MLP(10, [256, 256, 512], activation_layer=nn.Tanh)

    def forward(self, ceof_points: Tensor, d: Tensor, f: Tensor, inlet_points: Tensor, inlet_ux: Tensor):
        """
        :param ceof_points: Coordinates of darcy boundary points(B, M, 3)
        :param d: Darcy coefficients (B, M, 3)
        :param f: Forchheimer coefficients (B, M, 3)
        :param inlet_points: Coordinates of inlet boundary points(B, N, 3)
        :param inlet_ux: inlet velocity along x (B, N, 1)
        :return: Parameter embedding (B, 1, 256)
        """
        points = torch.cat([ceof_points, inlet_points], dim=-2)
        par_dim = d.shape[-1] + f.shape[-1] + inlet_ux.shape[-1]
        x = torch.zeros((points.shape[0], points.shape[1], par_dim), device=points.device)
        x[..., 0:ceof_points.shape[-2], 0:3] = d
        x[..., 0:ceof_points.shape[-2], 3:6] = f
        x[..., ceof_points.shape[-2]:, 6:7] = inlet_ux
        x = torch.cat([points, x], dim=-1)
        y = self.linear(x)
        return torch.max(y, dim=1, keepdim=True)[0]


class GeometryEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = MLP(4, [256, 256, 256], activation_layer=nn.Tanh)

    def forward(self, points: Tensor, zones_ids: Tensor) -> Tensor:
        """
        :param points: Coordinates (B, N, 2)
        :param zones_ids: Porous zone index (B, M, 1)
        :return: Embedding (B, 1, 64)
        """
        x = torch.cat([points, zones_ids], dim=-1)
        y = self.linear(x)
        return torch.max(y, dim=1, keepdim=True)[0]


class NeuralOperator(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.Tanh()
        )
        if dropout:
            self.linear.append(nn.Dropout(0.15))

    def forward(self, x: Tensor, par_embedding: Tensor):
        return self.linear(x) * par_embedding


class PiGano(L.LightningModule):
    def __init__(self, domain_dict: dict[str, slice], scalers: dict[str, StandardScaler | Normalizer]):
        super().__init__()

        self.branch = Branch()
        self.geometry_encoder = GeometryEncoder()
        self.points_encoder = MLP(4, [256, 256, 256], activation_layer=nn.Tanh)
        self.neural_op1 = NeuralOperator(512, 512)
        self.neural_op2 = NeuralOperator(512, 512, True)
        self.neural_op3 = NeuralOperator(512, 512, True)
        self.neural_op4 = NeuralOperator(512, 512)
        self.reduction = nn.Linear(512, 4)

        self.domain_dict = domain_dict
        self.mu = 14.61e-6

        self.u_scaler = scalers['U']
        self.p_scaler = scalers['p']
        self.points_scaler = scalers['Points']
        self.d_scaler = scalers['d']
        self.f_scaler = scalers['f']

        self.momentum_x_loss = MomentumLoss(0, 1, 2, self.mu, self.u_scaler, self.points_scaler, self.p_scaler,
                                            self.d_scaler, self.f_scaler)
        self.momentum_y_loss = MomentumLoss(1, 0, 2, self.mu, self.u_scaler, self.points_scaler, self.p_scaler,
                                            self.d_scaler, self.f_scaler)
        self.momentum_z_loss = MomentumLoss(2, 0, 1, self.mu, self.u_scaler, self.points_scaler, self.p_scaler,
                                            self.d_scaler, self.f_scaler)
        self.continuity_loss = ContinuityLoss(self.u_scaler, self.points_scaler)

        self.verbose_predict = False

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
        self.save_hyperparameters()

    def to(self, *args, **kwargs):
        res = super().to(*args, *kwargs)
        self.u_scaler = self.u_scaler.to(*args, *kwargs)
        self.p_scaler = self.p_scaler.to(*args, *kwargs)
        self.points_scaler = self.points_scaler.to(*args, *kwargs)
        return res

    def forward(self,
                pred_points: Tensor,
                zones_ids: Tensor,
                par_points: Tensor,
                par_d: Tensor,
                par_f: Tensor,
                par_inlet_points: Tensor,
                par_inlet_ux: Tensor) -> Tensor:
        geom_embedding = self.geometry_encoder.forward(pred_points.detach(), zones_ids)
        par_embedding = self.branch.forward(par_points, par_d, par_f, par_inlet_points, par_inlet_ux)
        local_embedding = self.points_encoder.forward(torch.cat([pred_points, zones_ids], dim=-1))

        geom_embedding = geom_embedding.repeat((1, local_embedding.shape[-2], 1))
        local_embedding = torch.cat([local_embedding, geom_embedding], dim=-1)

        y = self.neural_op1.forward(local_embedding, par_embedding)
        y = self.neural_op2.forward(y, par_embedding)
        y = self.neural_op3.forward(y, par_embedding)
        y = self.neural_op4.forward(y, par_embedding)
        return self.reduction.forward(y)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, eps=1e-6)
        scheduler = ExponentialLR(optimizer, 0.999)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def differentiate_field(self, points, ui: Tensor, i: int, j: int, k: int) -> tuple:
        d_ui = self.calculate_gradients(ui, points)
        d_ui_i, d_ui_j, d_ui_k = d_ui[:, :, i:i + 1], d_ui[:, :, j:j + 1], d_ui[:, :, k:k + 1]
        dd_ui_i = self.calculate_gradients(d_ui_i, points)[:, :, i:i + 1]
        dd_ui_j = self.calculate_gradients(d_ui_j, points)[:, :, j:j + 1]
        dd_ui_k = self.calculate_gradients(d_ui_j, points)[:, :, k:k + 1]
        return d_ui_i, (d_ui_j, d_ui_k, dd_ui_i, dd_ui_j, dd_ui_k)

    def calculate_gradients(self, outputs: Tensor, inputs: Tensor) -> Tensor:
        return autograd.grad(outputs, inputs,
                             grad_outputs=torch.ones_like(outputs),
                             retain_graph=True, create_graph=True)[0]

    def training_step(self, batch: list, batch_idx: int):
        in_data = FoamData(batch, self.domain_dict)

        internal_points = in_data['internal'].points
        internal_points.requires_grad = True
        in_points = torch.cat([internal_points, in_data['boundary'].points], dim=-2)

        pred = self.forward(in_points,
                            in_data.zones_ids,
                            in_data['internal'].points,
                            in_data['internal'].d,
                            in_data['internal'].f,
                            in_data['inlet'].points,
                            in_data['inlet'].inlet_ux)

        pred_data = PdeData(pred, self.domain_dict)

        # i=0 is x, j=1 is y
        d_ux_x, x_diff = self.differentiate_field(internal_points, pred_data['internal'].ux, 0, 1, 2)
        # i=1 is y, j=0 is x
        d_uy_y, y_diff = self.differentiate_field(internal_points, pred_data['internal'].uy, 1, 0, 2)
        # i=1 is y, j=0 is x
        d_uz_z, z_diff = self.differentiate_field(internal_points, pred_data['internal'].uz, 2, 0, 1)

        d_p = self.calculate_gradients(pred_data['internal'].p, internal_points)
        d_p_x, d_p_y, d_p_z = d_p[..., 0:1], d_p[..., 1:2], d_p[..., 2:3]

        obs_ux_loss = mse_loss(pred_data.ux.gather(1, in_data.obs_samples[..., 0:1]), in_data.obs.pde.ux)
        obs_uy_loss = mse_loss(pred_data.uy.gather(1, in_data.obs_samples[..., 0:1]), in_data.obs.pde.uy)
        obs_uz_loss = mse_loss(pred_data.uz.gather(1, in_data.obs_samples[..., 0:1]), in_data.obs.pde.uz)
        obs_p_loss = mse_loss(pred_data.p.gather(1, in_data.obs_samples[..., 0:1]), in_data.obs.pde.p)

        boundary_p_loss = mse_loss(pred_data['boundary'].p, in_data['boundary'].pde.p)
        boundary_ux_loss = mse_loss(pred_data['boundary'].ux, in_data['boundary'].pde.ux)
        boundary_uy_loss = mse_loss(pred_data['boundary'].uy, in_data['boundary'].pde.uy)
        boundary_uz_loss = mse_loss(pred_data['boundary'].uz, in_data['boundary'].pde.uz)

        cont_loss = self.continuity_loss(d_ux_x, d_uy_y, d_uz_z)
        mom_loss_x = self.momentum_x_loss(pred_data['internal'].ux,
                                          pred_data['internal'].uy,
                                          pred_data['internal'].uz,
                                          d_p_x,
                                          in_data['internal'].zones_ids,
                                          in_data['internal'].d,
                                          in_data['internal'].f,
                                          d_ux_x,
                                          *x_diff)

        mom_loss_y = self.momentum_y_loss(pred_data['internal'].uy,
                                          pred_data['internal'].ux,
                                          pred_data['internal'].uz,
                                          d_p_y,
                                          in_data['internal'].zones_ids,
                                          in_data['internal'].d,
                                          in_data['internal'].f,
                                          d_uy_y,
                                          *y_diff)

        mom_loss_z = self.momentum_y_loss(pred_data['internal'].uz,
                                          pred_data['internal'].ux,
                                          pred_data['internal'].uy,
                                          d_p_z,
                                          in_data['internal'].zones_ids,
                                          in_data['internal'].d,
                                          in_data['internal'].f,
                                          d_uz_z,
                                          *z_diff)

        loss = (cont_loss +
                mom_loss_x +
                mom_loss_y +
                mom_loss_z +
                boundary_p_loss +
                boundary_ux_loss +
                boundary_uy_loss +
                boundary_uz_loss +
                obs_p_loss * 100 +
                obs_ux_loss * 100 +
                obs_uy_loss * 100 +
                obs_uz_loss * 100)

        self.training_loss_togger.log(loss,
                                      cont_loss,
                                      mom_loss_x,
                                      mom_loss_y,
                                      mom_loss_z,
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
        batch_data = FoamData(batch, self.domain_dict)
        pred = self.forward(batch_data.points,
                            batch_data.zones_ids,
                            batch_data['internal'].points,
                            batch_data['internal'].d,
                            batch_data['internal'].f,
                            batch_data['inlet'].points,
                            batch_data['inlet'].inlet_ux)
        pred_data = PdeData(pred)
        p_error = l1_loss(self.p_scaler.inverse_transform(pred_data.p),
                          self.p_scaler.inverse_transform(batch_data.pde.p))
        ux_error = l1_loss(self.u_scaler[0].inverse_transform(pred_data.ux),
                           self.u_scaler[0].inverse_transform(batch_data.pde.ux))
        uy_error = l1_loss(self.u_scaler[1].inverse_transform(pred_data.uy),
                           self.u_scaler[1].inverse_transform(batch_data.pde.uy))
        uz_error = l1_loss(self.u_scaler[2].inverse_transform(pred_data.uz),
                           self.u_scaler[2].inverse_transform(batch_data.pde.uz))
        self.val_loss_logger.log(p_error, ux_error, uy_error, uz_error)

    def predict_step(self, batch: Tensor) -> tuple[Tensor, Tensor] | Tensor:
        in_data = FoamData(batch, self.domain_dict)
        if self.verbose_predict:
            torch.set_grad_enabled(True)
            internal_points = in_data['internal'].points
            internal_points.requires_grad = True
            in_points = torch.cat([internal_points, in_data['boundary'].points], dim=-2)

            pred = self.forward(in_points,
                                in_data.zones_ids,
                                in_data['internal'].points,
                                in_data['internal'].d,
                                in_data['internal'].f,
                                in_data['inlet'].points,
                                in_data['inlet'].inlet_ux)
            pred_data = PdeData(pred, self.domain_dict)

            # i=0 is x, j=1 is y
            d_ux_x, x_diff = self.differentiate_field(internal_points, pred_data['internal'].ux, 0, 1, 2)
            # i=1 is y, j=0 is x
            d_uy_y, y_diff = self.differentiate_field(internal_points, pred_data['internal'].uy, 1, 0, 2)
            # i=1 is y, j=0 is x
            d_uz_z, z_diff = self.differentiate_field(internal_points, pred_data['internal'].uz, 2, 0, 1)

            d_p = self.calculate_gradients(pred_data['internal'].p, internal_points)
            d_p_x, d_p_y, d_p_z = d_p[..., 0:1], d_p[..., 1:2], d_p[..., 2:3]

            cont = self.continuity_loss.func(d_ux_x, d_uy_y, d_uz_z)
            momentum_x = self.momentum_x_loss.func(pred_data['internal'].ux,
                                                   pred_data['internal'].uy,
                                                   pred_data['internal'].uz,
                                                   d_p_x,
                                                   in_data['internal'].zones_ids,
                                                   in_data['internal'].d,
                                                   in_data['internal'].f,
                                                   d_ux_x,
                                                   *x_diff)

            momentum_y = self.momentum_y_loss.func(pred_data['internal'].uy,
                                                   pred_data['internal'].ux,
                                                   pred_data['internal'].uz,
                                                   d_p_y,
                                                   in_data['internal'].zones_ids,
                                                   in_data['internal'].d,
                                                   in_data['internal'].f,
                                                   d_uy_y,
                                                   *y_diff)

            momentum_z = self.momentum_y_loss.func(pred_data['internal'].uz,
                                                   pred_data['internal'].ux,
                                                   pred_data['internal'].uy,
                                                   d_p_z,
                                                   in_data['internal'].zones_ids,
                                                   in_data['internal'].d,
                                                   in_data['internal'].f,
                                                   d_uz_z,
                                                   *z_diff)
            torch.set_grad_enabled(False)
            return pred_data.data, torch.cat([momentum_x, momentum_y, momentum_z, cont], dim=-1)
        else:
            return self.forward(in_data.points,
                                in_data.zones_ids,
                                in_data['internal'].points,
                                in_data['internal'].d,
                                in_data['internal'].f,
                                in_data['inlet'].points,
                                in_data['inlet'].inlet_ux)
