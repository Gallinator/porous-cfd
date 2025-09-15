from numpy.random import default_rng

from common.training import build_arg_parser, train
from dataset.foam_dataset import FoamDataset
from models.losses import FixedLossScaler
from models.pipn_foam import PipnFoam

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
    model = PipnFoam(nu=1489.4e-6,
                     d=14000,
                     f=17.11,
                     in_dim=2,
                     out_features=3,
                     scalers=train_data.normalizers,
                     loss_scaler=loss_scaler)

    train(args, model, train_data, val_data)
