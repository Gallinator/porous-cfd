from argparse import ArgumentParser, Namespace
from pathlib import Path
import numpy as np
import pandas
import torch
from pandas import DataFrame
from scipy.stats import kruskal, mannwhitneyu, f_oneway, shapiro, levene
from common import evaluation
from common.evaluation import evaluate
from dataset.foam_dataset import FoamDataset
from visualization.common import plot_multi_bar, get_fields_names, plot_per_case


def build_arg_parser() -> ArgumentParser:
    arg_parser = evaluation.build_arg_parser()
    arg_parser.add_argument('--checkpoint-other', type=str)
    return arg_parser


def switch_active_checkpoint(args):
    args_dict = vars(args)
    old_active = args_dict['checkpoint']
    args_dict['checkpoint'] = args_dict['checkpoint_other']
    args_dict['checkpoint_other'] = old_active
    return Namespace(**args_dict)


def plot_error_comparison(name_1, name_2, errors_1: DataFrame, errors_2: DataFrame, plots_path):
    metrics = set(errors_1.index).intersection(errors_2.index)

    for m in metrics:
        values_1 = errors_1.loc[m].values
        values_2 = errors_2.loc[m].values

        plot_multi_bar(m,
                       {name_1: values_1.tolist(), name_2: values_2.tolist()},
                       get_fields_names(values_1),
                       plots_path)


def get_name_from_checkpoint(checkpoint):
    name = Path(checkpoint).parent.name.replace('-', ' ')
    if not name[0].isupper():
        name = name.capitalize()
    return name


def plot_max_difference(title, errors_1, errors_2, reduction_f, plots_path, data):
    max_1, max_2 = reduction_f(errors_1, axis=-2), reduction_f(errors_2, axis=-2)
    delta = max_1 - max_2
    plot_per_case(title, delta, plots_path)


def compare(args, model1, model2, data: FoamDataset):
    results = {}
    eval_data_path = []
    torch.manual_seed(8421)

    def postprocess_fn(dataset, partial_results, plots_path):
        results[active_model] = partial_results
        eval_data_path.append(plots_path)

    name_1 = get_name_from_checkpoint(args.checkpoint)
    name_2 = get_name_from_checkpoint(args.checkpoint_other)

    active_model = name_1

    evaluate(args, model1, data, False, None, postprocess_fn)

    active_model = name_2
    args = switch_active_checkpoint(args)
    evaluate(args, model2, data, False, None, postprocess_fn)

    plots_dir = Path(args.checkpoint).parent.parent / 'comparisons' / f'{name_1} vs {name_2}' / Path(data.data_dir).name
    plots_dir.mkdir(exist_ok=True, parents=True)

    errors_1 = np.concatenate([results[name_1]['U error'], results[name_1]['p error']], axis=-1)
    errors_2 = np.concatenate([results[name_2]['U error'], results[name_2]['p error']], axis=-1)

    plot_max_difference('Max error difference', errors_1, errors_2, np.max, plots_dir, data)
    plot_max_difference('Average error difference', errors_1, errors_2, np.mean, plots_dir, data)

    errors_1 = np.concatenate(errors_1)
    errors_2 = np.concatenate(errors_2)

    index = ['Ux', 'Uy', 'Uz'][:errors_2.shape[-1] - 1] + ['p']
    results_df = DataFrame(index=index, columns=['Kruskal-Wallis', 'Mann-Whitney U', 'ANOVA'])

    kruskal_results = kruskal(errors_1, errors_2, axis=0, keepdims=True)
    results_df['Kruskal-Wallis'] = kruskal_results[-1].flatten()

    mannwhitneyu_results = mannwhitneyu(errors_1, errors_2, axis=0, keepdims=True)
    results_df['Mann-Whitney U'] = mannwhitneyu_results[-1].flatten()

    shapiro_df = DataFrame(index=index, columns=[name_1, name_2])
    transf_1, transf_2 = np.log(errors_1), np.log(errors_2)
    shapiro_1 = shapiro(transf_1, axis=0, keepdims=True)
    shapiro_2 = shapiro(transf_2, axis=0, keepdims=True)
    shapiro_df[name_1] = shapiro_1[-1].flatten()
    shapiro_df[name_2] = shapiro_2[-1].flatten()

    levene_results = levene(transf_1, transf_2, center='mean')[-1]
    print('Homoscedasticity transformed p-values')
    print(DataFrame(data=[levene_results], columns=index))
    print('\n')

    anova_results = f_oneway(transf_1, transf_2, axis=0, keepdims=True)
    results_df['ANOVA'] = anova_results[-1].flatten()

    print('Log transformed errors normality test p-values')
    print(shapiro_df)
    print('\n')

    print('Statistical tests p-values')
    print(results_df)

    eval1 = pandas.read_csv(f'{eval_data_path[0]}/Errors.csv', index_col=0)
    eval2 = pandas.read_csv(f'{eval_data_path[1]}/Errors.csv', index_col=0)

    plot_error_comparison(name_1, name_2, eval1, eval2, plots_dir)
    shapiro_df.to_csv(plots_dir / 'Shapiro.csv')
    results_df.to_csv(plots_dir / 'Test.csv')
