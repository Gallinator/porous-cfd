from numpy.random import default_rng

from common.training import train, build_arg_parser
from dataset.foam_dataset import FoamDataset
from models.losses import FixedLossScaler
from models.pi_gano.pi_gano import PiGano
from models.pi_gano.pi_gano_pp import PiGanoPp
from models.pi_gano.pi_gano_pp_full import PiGanoPpFull


def get_model(name, normalizers):
    loss_scaler = FixedLossScaler({'continuity': [100],
                                   'momentum': [100] * 3,
                                   'boundary': [1] * 4,
                                   'observations': [1] * 4})
    variable_boundaries = {'Subdomains': ['inlet', 'internal'], 'Features': ['Ux-inlet', 'd', 'f']}
    match name:
        case 'pi-gano':
            return PiGano(14.61e-6,
                          4,
                          [10, 256, 256, 512],
                          [4, 256, 256, 256],
                          [3, 256, 256, 256],
                          4,
                          [0, 0.15, 0.15, 0, 0],
                          normalizers,
                          variable_boundaries,
                          loss_scaler)
        case 'pi-gano-pp':
            return PiGanoPp(14.61e-6,
                            4,
                            [10, 256, 256, 512],
                            [[3 + 1 + 3, 256], [256 + 3, 256], [256 + 3, 256]],
                            [0.2, 0.5, 1],
                            [0.7, 0.5, 0.25],
                            [3, 256, 256, 256],
                            4,
                            [0, 0.1, 0.1, 0, 0],
                            normalizers,
                            variable_boundaries,
                            loss_scaler)
        case 'pi-gano-pp-full':
            return PiGanoPpFull(nu=14.61e-6,
                                out_features=4,
                                branch_layers=[10, 256, 256, 256],
                                enc_layers=[[3 + 1 + 3, 64, 64, 128], [128 + 3, 128, 128, 256], [256 + 3, 512, 1024]],
                                enc_radius=[0.2, 0.5, 1],
                                enc_fraction=[0.7, 0.5, 0.25],
                                dec_layers=[[1024 + 256, 256, 256], [128 + 256, 128, 128], [128 + 4, 128, 128, 128, 4]],
                                dec_k=[6, 6, 6],
                                fp_dropout=[0., 0., [0., 0.2, 0.2, 0.]],
                                scalers=normalizers,
                                loss_scaler=loss_scaler,
                                variable_boundaries=variable_boundaries)
        case _:
            raise NotImplementedError


if __name__ == '__main__':
    args = build_arg_parser().parse_args()

    n_internal = args.n_internal
    n_boundary = args.n_boundary
    n_obs = args.n_observations

    rng = default_rng(8421)
    train_data = FoamDataset(args.val_dir, n_internal, n_boundary, n_obs, rng=rng)
    val_data = FoamDataset(args.val_dir, n_internal, n_boundary, n_obs, rng=rng, meta_dir=args.train_dir)

    model = get_model(args.model, train_data.normalizers)

    train(args, model, train_data, val_data)
