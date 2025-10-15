from numpy.random import default_rng

from common.training import train, build_arg_parser
from dataset.foam_dataset import FoamDataset
from models.losses import FixedLossScaler
from models.pipn.pipn_foam import PipnFoamPpFull, PipnFoamPpMrg, PipnFoamPp, PipnFoam


def get_model(name, normalizers):
    n_dims = 3
    loss_scaler = FixedLossScaler({'continuity': [1],
                                   'momentum': [1] * n_dims,
                                   'boundary': [1] * (n_dims + 1),
                                   'observations': [100] * (n_dims + 1)})
    nu, d, f = 1489.4e-6, 30000, 79.731
    n_boundary_ids = 4
    match name:
        case 'pipn':
            return PipnFoam(nu=nu,
                            d=d,
                            f=f,
                            fe_local_layers=[n_dims, 64, 64],
                            fe_global_layers=[64 + n_boundary_ids + 1, 96, 128, 1024],
                            seg_layers=[1024 + 64, 512, 256, 128, n_dims + 1],
                            seg_dropout=[0.03, 0.02, 0, 0],
                            scalers=normalizers,
                            loss_scaler=loss_scaler)
        case 'pipn-pp':
            return PipnFoamPp(nu=nu,
                              d=d,
                              f=f,
                              fe_local_layers=[n_dims, 64, 64],
                              seg_layers=[1024 + 64, 384, 128, n_dims + 1],
                              seg_dropout=[0.03, 0, 0],
                              fe_radius=[0.5, 1],
                              fe_fraction=[0.5, 0.25],
                              fe_global_layers=[[n_dims + n_boundary_ids + n_dims, 64, 128],
                                                [128 + n_dims, 128, 256],
                                                [256 + n_dims, 256, 1024]],
                              scalers=normalizers,
                              loss_scaler=loss_scaler,
                              max_neighbors=16)
        case 'pipn-pp-mrg':
            return PipnFoamPpMrg(nu=nu,
                                 d=d,
                                 f=f,
                                 fe_local_layers=[n_dims, 64, 64],
                                 seg_layers=[1024 + 64, 384, 128, n_dims + 1],
                                 seg_dropout=[0.03, 0, 0],
                                 scalers=normalizers,
                                 loss_scaler=loss_scaler,
                                 n_dims=n_dims,
                                 mrg_in_features=n_boundary_ids + n_dims,
                                 max_neighbors=16)
        case 'pipn-pp-full':
            return PipnFoamPpFull(nu=nu,
                                  d=d,
                                  f=f,
                                  enc_layers=[[n_dims + n_boundary_ids + 1 + n_dims, 64, 64, 128],
                                              [128 + n_dims, 128, 128, 256],
                                              [256 + n_dims, 1024]],
                                  enc_radius=[0.4, 0.8],
                                  enc_fraction=[0.5, 0.25],
                                  dec_layers=[[1024 + 256, 256, 256],
                                              [128 + 256, 128, 128],
                                              [128 + n_dims + n_boundary_ids + 1, 128, 128, 128, n_dims + 1]],
                                  dec_k=[3, 3, 3],
                                  last_dec_dropout=[0., 0., [0., 0.2, 0.2, 0.]],
                                  scalers=normalizers,
                                  loss_scaler=loss_scaler,
                                  max_neighbors=16)
        case _:
            raise NotImplementedError


def run():
    args = build_arg_parser().parse_args()

    n_internal = args.n_internal
    n_boundary = args.n_boundary
    n_obs = args.n_observations

    rng = default_rng(8421)
    train_data = FoamDataset(args.train_dir, n_internal, n_boundary, n_obs, rng=rng)
    val_data = FoamDataset(args.val_dir, n_internal, n_boundary, n_obs, rng=rng, meta_dir=args.train_dir)

    model = get_model(args.model, train_data.normalizers)

    train(args, model, train_data, val_data)


if __name__ == '__main__':
    run()
