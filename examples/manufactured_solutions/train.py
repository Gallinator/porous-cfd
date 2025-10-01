from numpy.random import default_rng

from common.training import build_arg_parser, train
from manufactured_dataset import ManufacturedDataset
from models.pipn.pipn_baseline import PipnManufactured, PipnManufacturedPorousPp


def get_model(name, d, f):
    n_dim = 2
    n_boundary_ids = 2
    if name == 'pipn':
        return PipnManufactured(nu=0.01, d=d, f=f,
                                fe_local_layers=[n_dim, 64, 64],
                                fe_global_layers=[64 + n_boundary_ids + 1, 64, 128, 1024],
                                seg_layers=[1024 + 64, 512, 256, 128, 3])
    elif name == 'pipn-pp':
        return PipnManufacturedPorousPp(nu=0.01, d=d, f=f,
                                        fe_local_layers=[n_dim, 64, 64],
                                        fe_global_layers=[[n_dim * 2 + n_boundary_ids, 64],
                                                          [64 + n_dim, 128],
                                                          [128 + n_dim, 1024]],
                                        fe_global_radius=[0.5, 0.25],
                                        fe_global_fraction=[0.6, 1.2],
                                        seg_layers=[1024 + 64, 512, 256, 128, 3])
    else:
        raise NotImplementedError()


def train_model():
    args = build_arg_parser().parse_args()

    n_internal = args.n_internal
    n_boundary = args.n_boundary

    rng = default_rng(8421)
    d = 50
    f = 1
    train_data = ManufacturedDataset(args.train_dir, n_internal, n_boundary, d, f, rng=rng)
    val_data = ManufacturedDataset(args.val_dir, n_internal, n_boundary, d, f, rng=rng, meta_dir=args.train_dir)

    model = get_model(args.model, d, f)

    train(args, model, train_data, val_data)


if __name__ == '__main__':
    train_model()
