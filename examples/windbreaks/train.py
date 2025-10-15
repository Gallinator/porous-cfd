from numpy.random import default_rng

from common.training import train, build_arg_parser
from dataset.foam_dataset import FoamDataset
from models.losses import FixedLossScaler, RelobraloScaler
from models.pi_gano.pi_gano import PiGano
from models.pi_gano.pi_gano_pp import PiGanoPp
from models.pi_gano.pi_gano_pp_full import PiGanoPpFull


def get_model(args, normalizers):
    if args.loss_scaler == 'relobralo':
        loss_scaler = RelobraloScaler(12, alpha=1 - 0.995)
    else:
        loss_scaler = FixedLossScaler({'continuity': [10],
                                       'momentum': [10] * 3,
                                       'boundary': [1] * 4,
                                       'observations': [1] * 4})
    variable_boundaries = {'Subdomains': ['inlet', 'internal'], 'Features': ['Ux-inlet', 'd', 'f']}
    n_dim = 3
    n_boundary_id = 5
    match args.model:
        case 'pi-gano':
            return PiGano(nu=14.61e-6,
                          out_features=n_dim + 1,
                          branch_layers=[10, 256, 256, 512],
                          geometry_layers=[n_boundary_id + n_dim + 1, 256, 256, 256],
                          local_layers=[n_dim, 256, 256, 256],
                          n_operators=4,
                          operator_dropout=[0, 0.15, 0.15, 0],
                          scalers=normalizers,
                          variable_boundaries=variable_boundaries,
                          loss_scaler=loss_scaler)
        case 'pi-gano-pp':
            return PiGanoPp(nu=14.61e-6,
                            out_features=n_dim + 1,
                            branch_layers=[10, 256, 256, 512],
                            geometry_layers=[[n_dim * 2 + n_boundary_id, 64, 128],
                                             [128 + n_dim, 128],
                                             [128 + n_dim, 256, 256]],
                            geometry_radius=[0.5, 1],
                            geometry_fraction=[0.5, 0.25],
                            local_layers=[n_dim, 256, 256, 256],
                            n_operators=4,
                            operator_dropout=[0, 0.15, 0.15, 0],
                            scalers=normalizers,
                            variable_boundaries=variable_boundaries,
                            loss_scaler=loss_scaler)
        case 'pi-gano-pp-full':
            return PiGanoPpFull(nu=14.61e-6,
                                out_features=4,
                                branch_layers=[10, 256, 256, 256],
                                enc_layers=[[n_dim * 2 + 1 + n_boundary_id, 64, 64, 128],
                                            [128 + n_dim, 128, 128, 256],
                                            [256 + n_dim, 512, 1024]],
                                enc_radius=[0.5, 1],
                                enc_fraction=[0.5, 0.25],
                                dec_layers=[[1024 + 256, 256, 256],
                                            [128 + 256, 128, 128],
                                            [128 + n_dim + 1 + n_boundary_id, 128, 128, 128, 4]],
                                dec_k=[3, 3, 3],
                                fp_dropout=[0., 0., [0., 0.2, 0.2, 0.]],
                                scalers=normalizers,
                                loss_scaler=loss_scaler,
                                variable_boundaries=variable_boundaries)
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

    model = get_model(args, train_data.normalizers)

    train(args, model, train_data, val_data)


if __name__ == '__main__':
    run()
