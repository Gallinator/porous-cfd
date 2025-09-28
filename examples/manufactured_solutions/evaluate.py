from pathlib import Path
from numpy.random import default_rng
from common.evaluation import build_arg_parser, evaluate
from dataset.data_parser import parse_model_type
from dataset.foam_data import FoamData
from dataset.foam_dataset import FoamDataset
from models.pipn.pipn_baseline import PipnManufactured, PipnManufacturedPorousPp
from manufactured_dataset import ManufacturedDataset


def get_model(checkpoint):
    model_type = parse_model_type(checkpoint)
    match model_type:
        case 'pipn':
            return PipnManufactured.load_from_checkpoint(checkpoint)
        case 'pipn-pp':
            return PipnManufacturedPorousPp.load_from_checkpoint(checkpoint)
        case _:
            raise NotImplementedError


def sample_process(data: FoamDataset, predicted: FoamData, target: FoamData, extras: FoamData) -> tuple:
    pass


def postprocess_fn(data: FoamDataset, results: tuple, plots_path: Path):
    pass


if __name__ == '__main__':
    args = build_arg_parser().parse_args()

    model = get_model(args.checkpoint)

    rng = default_rng(8421)
    data = ManufacturedDataset(args.data_dir, args.n_internal, args.n_boundary, 50, 1, rng, args.meta_dir)

    evaluate(args, model, data, True, sample_process, postprocess_fn)
