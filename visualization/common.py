import glob
from functools import partial
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pandas
from matplotlib import pyplot as plt
from numpy.linalg import norm
from rich.progress import track
from scipy.interpolate import make_smoothing_spline

from dataset import data_parser
from dataset.data_parser import parse_internal_fields

M_S = '\left[ \\frac{m}{s} \\right]'
M2_S2 = '\left[ \\frac{m^2}{s^2} \\right]'

LIGHT_COLORS = ['lightblue', 'lightcoral', 'bisque',
                'lightgreen', 'lightgrey', 'lightsalmon',
                'moccasin', 'powderblue', 'lavander',
                'thistle', 'lightpink']


def plot_or_save(fig, save_path):
    if fig._suptitle is not None:
        file_name = fig._suptitle.get_text()
    else:
        file_name = fig.axes[0].get_title()

    if save_path is not None:
        plt.savefig(f'{save_path}/{file_name}.png', transparent=True, dpi=300)
        plt.close(fig)
    else:
        plt.show()


def plot_histogram(ax, data, color: str, title: str, bins='doane'):
    ax.set_title(title, pad=10)
    ax.hist(data, bins=bins, color=color, edgecolor='black')


def plot_dataset_dist(path: str, save_path=None):
    data = []
    for case in track(glob.glob(f"{path}/*/"), description="Reading data"):
        case_data = data_parser.parse_case_fields(case, 'U', 'p', 'cellToRegion')
        data.append(case_data)

    data = pandas.concat(data)
    plot_data_dist(f'{Path(path).name} distribution', data['U'].values, data['p'].values, data['cellToRegion'].values,
                   save_path)
    box_plot('Fields boxplot',
             [*np.hsplit(data['U'].values, 3), np.vstack(data['p'].values)],
             ['$U_x$', '$U_y$', '$U_z$', '$p$'],
             save_path)


def plot_data_dist(title, u, p, zones_ids=None, save_path=None):
    fig = plt.figure(layout='constrained')
    fig.suptitle(title, fontsize=20)
    ax_ux, ax_uy, ax_uz, ax_p, ax_zones, _ = fig.subplots(ncols=3, nrows=2).flatten()

    plot_histogram(ax_ux, u[..., 0], 'lightsteelblue', '$U_x$')
    plot_histogram(ax_uy, u[..., 1], 'lemonchiffon', '$U_y$')
    if u.shape[-1] > 2:
        plot_histogram(ax_uz, u[..., 2], 'thistle', '$U_z$')
    plot_histogram(ax_p, p, 'lightsalmon', '$p$')
    if zones_ids is not None:
        plot_histogram(ax_zones, zones_ids, 'palegreen', 'Material zones', 2)
    else:
        plot_histogram(ax_zones, norm(u, axis=1), 'palegreen', '$U$')
    plot_or_save(fig, save_path)


def plot_barh(ax, title, values, labels, colors, spacing=0.01, offset=0.0):
    ax.set_title(title, pad=10)
    ax.set_xlim(right=max(values) * 1.3)
    w = 0.01
    x = np.arange(0, spacing * len(values), step=w)
    rects = ax.barh(x + offset, values, w, color=colors, label=labels)
    ax.bar_label(rects, fmt='%.2e', padding=10)
    ax.set_yticks([])
    ax.legend(ncols=2)


def plot_timing(total: list, average: list, save_path=None):
    fig = plt.figure()
    ax_total, ax_avg = fig.subplots(2)
    colors = ['salmon', 'lightblue']
    labels = ['PINN', 'OpenFoam']

    plot_barh(ax_total, 'Total simulation time [s]', total, labels, colors)
    plot_barh(ax_avg, 'Average simulation time [s/case]', average, labels, colors)

    fig.tight_layout()
    plot_or_save(fig, save_path)


def plot_errors(title, *args, save_path=None):
    fig, ax = plt.subplots()
    colors = ['salmon', 'lightblue', 'palegreen']
    labels = [f'$U_x {M_S}$', f'$U_y {M_S}$', f'$p {M2_S2}$']
    if len(*args) > 3:
        colors.append('moccasin')
        labels.insert(-1, f'$U_z {M_S}$', )
    plot_barh(ax, title, *args, labels, colors)

    fig.tight_layout()
    plot_or_save(fig, save_path)


def plot_multi_bar(title, values: dict, values_labels, save_path=None):
    fig, ax = plt.subplots()
    ax.set_title(title, pad=10)
    w = 0.01
    x = np.array([x * 0.03 for x in range(len(values_labels))])
    colors = LIGHT_COLORS[:len(values)]

    for i, (k, v) in enumerate(values.items()):
        rects = ax.bar(x + i * w, v, w, label=k, color=colors[i])
        ax.bar_label(rects, fmt='%.2e', padding=10)

    ax.legend()
    ax.set_ylim(0, max([max(d) for d in values.values()]) * 1.1)
    ax.set_xticks(x + w / 2, values_labels)
    fig.tight_layout()
    plot_or_save(fig, save_path)


def plot_u_direction_change(data_dir, save_path=None):
    diff = []
    for c in list(set(glob.glob(f'{data_dir}/*')) - set(glob.glob(f'{data_dir}/*.json'))):
        data = parse_internal_fields(c, 'mag(grad(Unorm))')
        diff.append(data.to_numpy())

    unorm_means = [np.mean(d) for d in diff]
    fig = plt.figure(layout='constrained')
    ax_1, ax_2 = fig.subplots(2, 1).flatten()
    ax_1.bar(np.arange(0, len(unorm_means)), unorm_means, color='lightblue')
    ax_1.set_title('Average U direction change per case')
    ax_1.set_xticks([])
    ax_1.set_ylabel('U direction change')

    plot_histogram(ax_2, unorm_means, 'salmon', 'Average U direction change distribution', 20)
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax_2.text(0.985, 0.94, f'Mean: {mean(unorm_means):.2f}\nStd: {stdev(unorm_means):.2f}',
              transform=ax_2.transAxes,
              fontsize=8,
              verticalalignment='top',
              horizontalalignment='right',
              bbox=props)
    ax_2.set_xlabel('U direction change')
    ax_2.set_ylabel('Frequency')
    plot_or_save(fig, save_path)


def box_plot(title: str, values, labels, save_path=None):
    fig, axs = plt.subplots(nrows=1, ncols=len(values))
    fig.suptitle(title)

    for a, v, l in zip(axs, values, labels):
        a.boxplot(v, tick_labels=[l])
    plot_or_save(fig, save_path)


def get_fields_names(f: np.ndarray):
    return ['$U_x$', '$U_y$', '$U_z$'][:f.shape[-1] - 1] + ['$p$']


def plot_errors_vs_var(title, errors, var, labels, save_path=None):
    n_errors = errors.shape[-1]
    fig, axs = plt.subplots(ncols=1, nrows=n_errors)
    fig.suptitle(title)

    fields_names = get_fields_names(errors)

    for i in range(n_errors):
        axs[i].scatter(var, errors[:, i], label='Raw')
        axs[i].set_xlabel(labels[0])
        axs[i].set_ylabel(labels[1])

        if len(var) > 5:
            interp = make_smoothing_spline(var, errors[..., i])
            x = np.linspace(min(var), max(var), 100)
            axs[i].plot(x, interp(x), c='red', label='Interpolated')
        axs[i].legend()
        axs[i].set_title(fields_names[i])

    plt.tight_layout()
    plot_or_save(fig, save_path)


def get_heatmap(mae: np.array, x: np.array, y: np.array) -> tuple[np.array, np.array, np.array]:
    """
    Creates a 2D heatmap matrix.
    :param field_mae: N
    :param x: N
    :param y: N
    :return a tuple with the matrix and the sorted x and y
    """
    x_unique = np.unique(x)
    y_unique = np.unique(y)[::-1]

    heatmap = np.ones((len(y_unique), len(x_unique))) * np.nan
    for f, x, y in zip(mae, x, y):
        j = (x_unique == x).nonzero()[0]
        i = (y_unique == y).nonzero()[0]
        heatmap[i, j] = f

    return heatmap, x_unique, y_unique


def plot_errors_vs_multi_vars(title, errors, x, y, labels, save_path=None):
    fig = plt.figure(figsize=(16, 9))
    axs = fig.subplots(nrows=1, ncols=errors.shape[-1], )
    fig.suptitle(title)
    fields_names = get_fields_names(errors)

    for ax, e, f_name in zip(axs, np.hsplit(errors, errors.shape[-1]), fields_names):
        matrix, label_x, label_y = get_heatmap(e, x, y)
        plot_heatmap(ax, matrix, label_x, label_y, labels)
        ax.set_title(f_name)

    plt.tight_layout()
    plot_or_save(fig, save_path)


def plot_heatmap(ax, matrix, x, y, labels):
    def tick_fmt(i, pos, l):
        if isinstance(l[0], np.int64):
            return f'{l[i]:d}'
        else:
            if l[i] < 1e-3:
                return f'{l[i]:.2e}'
            else:
                return f'{l[i]:.3f}'

    ax.set_xticks(range(len(x)), labels=x, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(y)), labels=y)
    ax.xaxis.set_major_formatter(partial(tick_fmt, l=x))
    ax.yaxis.set_major_formatter(partial(tick_fmt, l=y))
    ax.imshow(matrix, cmap='Wistia')
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])

    for i in range(len(y)):
        for j in range(len(x)):
            value = matrix[i][j]
            if value >= 0:
                ax.text(j, i, f'{value:.2e}', ha="center", va="center", color="black")
