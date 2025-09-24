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
    match name:
        case 'pi-gano':
            return PiGano(1489.4e-6,
                          3,
                          [8, 128, 352, 352, 352],
                          [3, 64, 176, 176, 176],
                          [2, 64, 176, 176, 176],
                          4,
                          [0, 0.1, 0.1, 0, 0],
                          normalizers,
                          variable_boundaries,
                          loss_scaler)
        case 'pi-gano-pp':
            return PiGanoPp(1489.4e-6,
                            3,
                            [8, 128, 352, 352, 352],
                            [[2 + 1 + 2, 64], [64 + 2, 176], [176 + 2, 176], [176 + 2, 176]],
                            [0.2, 0.5, 1],
                            [0.7, 0.5, 0.25],
                            [2, 64, 176, 176, 176],
                            4,
                            [0, 0.1, 0.1, 0, 0],
                            normalizers,
                            variable_boundaries,
                            loss_scaler)
        case _:
            raise NotImplementedError


if __name__ == '__main__':
    args = build_arg_parser().parse_args()

    n_internal = args.n_internal
    n_boundary = args.n_boundary
    n_obs = args.n_observations

    rng = default_rng(8421)
    train_data = FoamDataset(args.train_dir, n_internal, n_boundary, n_obs, rng=rng)
    val_data = FoamDataset(args.val_dir, n_internal, n_boundary, n_obs, rng=rng, meta_dir=args.train_dir)

    model = get_model(args.model, train_data.normalizers)

    train(args, model, train_data, val_data)
