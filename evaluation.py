from pathlib import Path
import numpy as np
import torch
from numpy.random import default_rng
from torch.nn.functional import l1_loss
from common.evaluation import save_mae_to_csv, build_arg_parser, evaluate
from dataset.foam_data import FoamData
from dataset.foam_dataset import FoamDataset
from models.pipn_fluid import PipnFluid
from visualization.common import plot_data_dist, plot_errors


def sample_process(data: FoamDataset, predicted: FoamData, target: FoamData) -> tuple:
    """
    This function is execute for each predicted sample. Useful to calculate metrics on the results (errors, etc...).
    :param data:
    :param predicted:
    :param target:
    :return: a tuple of data
    """
    u_scaler = data.normalizers['U'].to(model.device)
    p_scaler = data.normalizers['p'].to(model.device)

    u_error = l1_loss(u_scaler.inverse_transform(predicted['U']),
                      u_scaler.inverse_transform(target['U']), reduction='none')
    p_error = l1_loss(p_scaler.inverse_transform(predicted['p']),
                      p_scaler.inverse_transform(target['p']), reduction='none')
    # Those are the errors at each point
    error = torch.cat([u_error, p_error], dim=-1)

    return error,


def postprocess_fn(data: FoamDataset, results: list, plots_path: Path):
    """
    Executed after all the samples have been processed. Useful to calculate metrics.
    :param data:
    :param results: The data from sample_process_fn. Each item contains the data for all samples
    :param plots_path: The base directory for plots
    :return:
    """

    # In this case errors contains the error for each point for each sample
    errors = results

    errors = np.concatenate(errors)

    plot_data_dist('Absolute error distribution',
                   errors[..., :data.n_dims],
                   errors[..., data.n_dims:],
                   save_path=plots_path)
    mae = np.average(errors, axis=0).tolist()
    plot_errors('Average relative error', mae, save_path=plots_path)

    if args.save_plots:
        errors_dict = {'Total': mae}
        save_mae_to_csv(errors_dict, ['Ux', 'Uy', 'p'], plots_path)


if __name__ == '__main__':
    args = build_arg_parser().parse_args()

    model = PipnFluid.load_from_checkpoint(args.checkpoint)

    rng = default_rng(8421)
    data = FoamDataset(args.data_dir, args.n_internal, args.n_boundary, args.n_observations, rng, args.meta_dir)

    evaluate(args, model, data, False, sample_process, postprocess_fn)
