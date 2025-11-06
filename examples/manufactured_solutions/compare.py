from numpy.random import default_rng

from common.compare import compare, build_arg_parser
from dataset.data_parser import parse_model_type
from examples.manufactured_solutions.manufactured_dataset import ManufacturedDataset
from models.pipn.pipn_baseline import PipnManufactured, PipnManufacturedPorousPp


def get_model(checkpoint):
    model_type = parse_model_type(checkpoint)
    match model_type:
        case 'pipn':
            return PipnManufactured.load_from_checkpoint(checkpoint)
        case 'pipn-pp':
            return PipnManufacturedPorousPp.load_from_checkpoint(checkpoint)
        case _:
            raise NotImplementedError


if __name__ == '__main__':
    args = build_arg_parser().parse_args()

    model1 = get_model(args.checkpoint)
    model2 = get_model(args.checkpoint_other)

    rng = default_rng(8421)
    data = ManufacturedDataset(args.data_dir, args.n_internal, args.n_boundary, 50, 1, rng, args.meta_dir)

    compare(args, model1, model2, data)
