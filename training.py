from numpy.random import default_rng

from common.training import build_arg_parser, train
from dataset.fluid_dataset import FluidDataset

from models.pipn_fluid import PipnFluid
from visualization.visualization_2d import plot_fields

if __name__ == '__main__':
    args = build_arg_parser().parse_args()

    n_internal = args.n_internal
    n_boundary = args.n_boundary
    n_obs = args.n_observations

    rng = default_rng(8421)
    train_data = FluidDataset(args.train_dir, n_internal, n_boundary, n_obs, rng=rng)
    val_data = FluidDataset(args.val_dir, n_internal, n_boundary, n_obs, rng=rng, meta_dir=args.train_dir)

    # Inspect a sample
    test_sample = train_data[0]
    plot_fields('Test plot', test_sample['C'], test_sample['U'], test_sample['p'])

    model = PipnFluid(n_features=2, n_pde=3, nu=0.001)

    train(args, model, train_data, val_data)
