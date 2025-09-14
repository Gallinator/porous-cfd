from numpy.random import default_rng

from common.training import build_arg_parser, train
from manufactured_dataset import ManufacturedDataset
from models.pipn_baseline import PipnPorous

if __name__ == '__main__':
    args = build_arg_parser().parse_args()

    n_internal = args.n_internal
    n_boundary = args.n_boundary
    n_obs = args.n_observations

    rng = default_rng(8421)
    d = 50
    f = 1
    train_data = ManufacturedDataset(args.train_dir, n_internal, n_boundary, d, f, rng=rng)
    val_data = ManufacturedDataset(args.val_dir, n_internal, n_boundary, d, f, rng=rng, meta_dir=args.train_dir)

    model = PipnPorous(nu=0.01, d=d, f=f, in_dim=2, out_features=3)

    train(args, model, train_data, val_data)
