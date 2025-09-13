from numpy.random import default_rng

from common.training import train, build_arg_parser
from dataset.foam_dataset import FoamDataset
from models.pi_gano_2d import PiGano2d

if __name__ == '__main__':
    args = build_arg_parser().parse_args()

    n_internal = args.n_internal
    n_boundary = args.n_boundary
    n_obs = args.n_observations

    rng = default_rng(8421)
    train_data = FoamDataset(args.train_dir, n_internal, n_boundary, n_obs, rng=rng)
    val_data = FoamDataset(args.val_dir, n_internal, n_boundary, n_obs, rng=rng, meta_dir=args.train_dir)

    model = PiGano2d(train_data.normalizers)

    train(args, model, train_data, val_data)
