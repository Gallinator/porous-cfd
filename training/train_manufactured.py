from numpy.random import default_rng
from dataset.manufactured_dataset import ManufacturedDataset
from models.pipn_baseline import PipnPorous
from training.common import train, build_arg_parser

if __name__ == '__main__':
    args = build_arg_parser().parse_args()

    n_internal = args.n_internal
    n_boundary = args.n_boundary
    n_obs = args.n_observations

    rng = default_rng(8421)
    train_data = ManufacturedDataset(args.train_dir, n_internal, n_boundary, 50, 1, rng=rng)
    val_data = ManufacturedDataset(args.val_dir, n_internal, n_boundary, 50, 1, rng=rng, meta_dir=args.train_dir)

    model = PipnPorous()

    train(args, model, train_data, val_data)
