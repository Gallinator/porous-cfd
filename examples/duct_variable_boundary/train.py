from numpy.random import default_rng

from common.training import train, build_arg_parser
from dataset.foam_dataset import FoamDataset
from models.losses import FixedLossScaler
from models.pi_gano.pi_gano import PiGano
from models.pi_gano.pi_gano_pp import PiGanoPp
from models.pi_gano.pi_gano_pp_full import PiGanoPpFull


def get_model(name, normalizers):
    loss_scaler = FixedLossScaler({'continuity': [1],
                                   'momentum': [1] * 2,
                                   'boundary': [1] * 3,
                                   'observations': [100] * 3})
    variable_boundaries = {'Subdomains': ['inlet', 'internal'], 'Features': ['U-inlet', 'd', 'f']}
    n_dim = 2
    n_boundary_id = 4
    match name:
        case 'pi-gano':
            return PiGano(1489.4e-6,
                          3,
                          [8, 128, 352, 352, 352],
                          [n_dim + n_boundary_id + 1, 64, 176, 176, 176],
                          [n_dim, 64, 176, 176, 176],
                          4,
                          [0, 0.1, 0.1, 0, 0],
                          normalizers,
                          variable_boundaries,
                          loss_scaler)
        case 'pi-gano-pp':
            return PiGanoPp(1489.4e-6,
                            3,
                            [8, 128, 352, 352, 352],
                            [[n_dim * 2 + n_boundary_id, 64],
                             [64 + n_dim, 176],
                             [176 + n_dim, 176],
                             [176 + n_dim, 176]],
                            [0.2, 0.5, 1],
                            [0.7, 0.5, 0.25],
                            [n_dim, 64, 176, 176, 176],
                            4,
                            [0, 0.1, 0.1, 0, 0],
                            normalizers,
                            variable_boundaries,
                            loss_scaler)
        case 'pi-gano-pp-full':
            return PiGanoPpFull(nu=1489.4e-6,
                                out_features=3,
                                branch_layers=[8, 128, 256, 256, 256],
                                enc_layers=[[n_dim * 2 + n_boundary_id + 1, 64, 64, 128],
                                            [128 + n_dim, 128, 128, 256],
                                            [256 + n_dim, 512]],
                                enc_radius=[0.2, 0.5, 1],
                                enc_fraction=[0.7, 0.5, 0.25],
                                dec_layers=[[512 + 256, 256, 256],
                                            [128 + 256, 128, 128],
                                            [128 + n_dim + n_boundary_id + 1, 128, 128, 128, 4]],
                                dec_k=[6, 6, 6],
                                fp_dropout=[0., 0., [0., 0.2, 0.2, 0.]],
                                scalers=normalizers,
                                loss_scaler=loss_scaler,
                                variable_boundaries=variable_boundaries)
        case _:
            raise NotImplementedError


def train_model():
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
    train_model()
