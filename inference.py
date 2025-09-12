from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

from common.inference import create_case_plot_dir, build_arg_parser, predict, create_plots_root
from dataset.foam_data import FoamData
from dataset.foam_dataset import FoamDataset
from models.pipn_fluid import PipnFluid
from visualization.visualization_2d import plot_fields


def sample_process_fn(data: FoamDataset, target: FoamData, predicted: FoamData, case_path: Path):
    """
    This function is executed for each sample predicted by the model. Useful to plot results.
    :param data:
    :param target:
    :param predicted:
    :param case_path: the path to the sample data. Useful to create a plot directory with the case name.
    :return:
    """
    case_plot_path = create_case_plot_dir(plots_path, case_path.name)

    plt.interactive(case_plot_path is None)

    plot_fields('Predicted', target['C'], predicted['U'], predicted['p'], save_path=case_plot_path)
    plot_fields('Ground truth', target['C'], target['U'], target['p'], save_path=case_plot_path)

    plt.interactive(False)

    u_error = np.abs(predicted['U'] - target['U'])
    p_error = np.abs(predicted['p'] - target['p'])
    plot_fields('Absolute error', target['C'], u_error, p_error, False, save_path=case_plot_path)


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    rng = np.random.default_rng(8421)
    plots_path = create_plots_root(args)
    model = PipnFluid.load_from_checkpoint(args.checkpoint)
    val_data = FoamDataset(args.data_dir, args.n_internal, args.n_boundary, args.n_observations, rng, args.meta_dir)
    predict(args, model, val_data, sample_process_fn)
