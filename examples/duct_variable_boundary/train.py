from numpy.random import default_rng

from common.training import train, build_arg_parser
from dataset.foam_dataset import FoamDataset
from models.losses import FixedLossScaler
from models.pi_gano import PiGano

if __name__ == '__main__':
    args = build_arg_parser().parse_args()

    n_internal = args.n_internal
    n_boundary = args.n_boundary
    n_obs = args.n_observations

    rng = default_rng(8421)
    train_data = FoamDataset(args.train_dir, n_internal, n_boundary, n_obs, rng=rng)
    val_data = FoamDataset(args.val_dir, n_internal, n_boundary, n_obs, rng=rng, meta_dir=args.train_dir)

    loss_scaler = FixedLossScaler({'continuity': [1],
                                   'momentum': [1] * 2,
                                   'boundary': [1] * 3,
                                   'observations': [100] * 3})
    model = PiGano(1489.4e-6,
                   2,
                   3,
                   8,
                   [128, 352, 352, 352],
                   [64, 176, 176, 176],
                   [64, 176, 176, 176],
                   4,
                   [0, 0.1, 0.1, 0, 0],
                   train_data.normalizers,
                   {'Subdomains': ['inlet', 'internal'], 'Features': ['U-inlet', 'd', 'f']},
                   loss_scaler)

    train(args, model, train_data, val_data)
