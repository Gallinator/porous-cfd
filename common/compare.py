import numpy as np
from pandas import DataFrame
from scipy.stats import kruskal, mannwhitneyu, f_oneway, shapiro

from common.evaluation import evaluate
from dataset.foam_dataset import FoamDataset


def compare(args, model1, model2, data: FoamDataset):
    results = {}

    def postprocess_fn(dataset, partial_results, plots_path):
        results[active_model] = partial_results

    active_model = 'Model1'
    evaluate(args, model1, data, False, None, postprocess_fn)

    active_model = 'Model2'
    evaluate(args, model2, data, False, None, postprocess_fn)

    u_1 = np.concatenate(results['Model1']['U error'])
    p_1 = np.concatenate(results['Model1']['p error'])
    errors_1 = np.concatenate([u_1, p_1], axis=-1)

    u_2 = np.concatenate(results['Model2']['U error'])
    p_2 = np.concatenate(results['Model2']['p error'])
    errors_2 = np.concatenate([u_2, p_2], axis=-1)

    index = ['Ux', 'Uy', 'Uz'][:errors_2.shape[-1] - 1] + ['p']
    results_df = DataFrame(index=index, columns=['Kruskal-Wallis', 'Mann-Whitney U', 'ANOVA'])

    kruskal_results = kruskal(errors_1, errors_2, axis=0, keepdims=True)
    results_df['Kruskal-Wallis'] = kruskal_results[-1].flatten()

    mannwhitneyu_results = mannwhitneyu(errors_1, errors_2, axis=0, keepdims=True)
    results_df['Mann-Whitney U'] = mannwhitneyu_results[-1].flatten()

    shapiro_df = DataFrame(index=index, columns=['Model 1', 'Model 2'])
    transf_1, transf_2 = np.log(errors_1), np.log(errors_2)
    shapiro_1 = shapiro(transf_1, axis=0, keepdims=True)
    shapiro_2 = shapiro(transf_2, axis=0, keepdims=True)
    shapiro_df['Model 1'] = shapiro_1[-1].flatten()
    shapiro_df['Model 2'] = shapiro_2[-1].flatten()

    anova_results = f_oneway(transf_1, transf_2, axis=0, keepdims=True)
    results_df['ANOVA'] = anova_results[-1].flatten()

    print('Log transformed errors normality test p-values')
    print(shapiro_df)
    print('\n')

    print('Statistical tests p-values')
    print(results_df)
