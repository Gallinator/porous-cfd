import glob
from functools import partial
from pathlib import Path
from statistics import mean, stdev

import matplotlib
import numpy as np
import pandas
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
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


def plot_or_save(fig: Figure, save_path: str | Path | None):
    """
    Shows the plot or saves it to a .png file.

    The filename is taken from the plot title. Pass a None path to show the plot without saving.
    """
    if fig._suptitle is not None:
        file_name = fig._suptitle.get_text()
    else:
        file_name = fig.axes[0].get_title()

    if save_path is not None:
        plt.savefig(f'{save_path}/{file_name}.png', transparent=True, dpi=300)
        plt.close(fig)
    else:
        plt.show()


def plot_histogram(ax: Axes, data, color: str, title: str, bins: str | int = 'doane'):
    """
    Plots a histogram on ax.
    :param ax: The axis to use for the plot.
    :param data: The data to plot of shape (N).
    :param color: Bars color.
    :param title: Title of the histogram.
    :param bins: Bins argument passed to the hist() function. Can be a number or a string for automatic binning.
    """
    ax.set_title(title, pad=10)
    ax.hist(data, bins=bins, color=color, edgecolor='black')


def plot_dataset_dist(path: str, save_path=None):
    """
    Plots the distribution of the velocity, pressure and number of porous points of a whole OpenFOAM dataset.
    :param path: Path to the data. The folder must contain a child folder for each OpenFOAM case.
    :param save_path: Pass None to show the plot without saving.
    """
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


def plot_data_dist(title: str,
                   u: np.ndarray,
                   p: np.ndarray,
                   zones_ids: np.ndarray | None = None,
                   save_path=None):
    """
    Plots the distribution of a single case.
    :param title: The title of the plot.
    :param u: The velocity data of shape (N,D).
    :param p: The pressure data of shape (N,1).
    :param zones_ids: The binary porous media indicator of shape (N,1). If not provided the distribution of the velocity magnitude is used.
    :param save_path: Pass None to show the plot without saving.
    """
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


def plot_barh(ax: Axes, title: str, values: list, labels: list[str], colors: list[str], spacing=0.01, offset=0.0):
    """
    Plots a horizontal bar graph on ax. Numerical values are formatted with the scientific notation.
    :param ax: The axis to plot on.
    :param title: The title of the plot.
    :param values: List of values to plot.
    :param labels: The label of each bar.
    :param colors: The color of each bar.
    :param spacing:
    :param offset:
    """
    ax.set_title(title, pad=10)
    ax.set_xlim(right=max(values) * 1.3)
    w = 0.01
    x = np.arange(0, spacing * len(values), step=w)
    rects = ax.barh(x + offset, values, w, color=colors, label=labels)
    ax.bar_label(rects, fmt='%.2e', padding=10)
    ax.set_yticks([])
    ax.legend(ncols=2)


def plot_timing(total: list, average: list, save_path=None):
    """
    Plots the average and total simulation times of both the OpenFOAM and PINN solvers.

    The PINN data must be the first in each list.
    :param total: Total simulation times over the whole dataset.
    :param average: Average per case simulation time.
    :param save_path: Pass None to show the plot without saving.
    """
    fig = plt.figure()
    ax_total, ax_avg = fig.subplots(2)
    colors = ['salmon', 'lightblue']
    labels = ['PINN', 'OpenFoam']

    plot_barh(ax_total, 'Total simulation time [s]', total, labels, colors)
    plot_barh(ax_avg, 'Average simulation time [s/case]', average, labels, colors)

    fig.tight_layout()
    plot_or_save(fig, save_path)


def plot_errors(title: str, *args, save_path=None):
    """
    Plots errors with horizontal bars.
    :param title: The title of the plot.
    :param args: The values to plot. Must be provided as Ux, Uy, Uz, p. Uz is optional.
    :param save_path: Pass None to show the plot without saving.
    """
    fig, ax = plt.subplots()
    colors = ['salmon', 'lightblue', 'palegreen']
    labels = [f'$U_x {M_S}$', f'$U_y {M_S}$', f'$p {M2_S2}$']
    if len(*args) > 3:
        colors.append('moccasin')
        labels.insert(-1, f'$U_z {M_S}$', )
    plot_barh(ax, title, *args, labels, colors)

    fig.tight_layout()
    plot_or_save(fig, save_path)


def plot_multi_bar(title: str, values: dict[str:list], values_labels: list[str], save_path=None):
    """
    Plots comparison bar graphs for multiple variables.
    :param title: The title of the plot.
    :param values: The values to plot. Must contain a vector of values for each key. The key names will be added to the legend.
    :param values_labels: The labels to add to the x-axis.
    :param save_path: Pass None to show the plot without saving.
    """
    fig, ax = plt.subplots(figsize=(len(values_labels) * len(values), 5))
    ax.set_title(title, pad=10)
    n_groups = len(values)
    w = 0.01
    x = np.array([x * w * (n_groups + 1) for x in range(len(values_labels))])
    colors = LIGHT_COLORS[:len(values)]

    for i, (k, v) in enumerate(values.items()):
        rects = ax.bar(x + w * i, v, w, label=k, color=colors[i])
        ax.bar_label(rects, fmt='%.2e', padding=10)

    ax.legend()
    ax.set_ylim(0, max([max(d) for d in values.values()]) * 1.1)
    ax.set_xticks(x + w / 2 * (len(values) - 1), values_labels)
    fig.tight_layout()
    plot_or_save(fig, save_path)


def plot_u_direction_change(data_dir: str, save_path=None):
    """
    Plots the distribution and per case average velocity direction change at each point.
    :param data_dir: The dataset directory. Must contain a subfolder for each OpenFOAM case.
    :param save_path: Pass None to show the plot without saving.
    """
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
    """
    Box plot of multi variable data.
    :param title: Title of the plot.
    :param values: The data to plot of shape (N,D).
    :param labels: Labels of shape (D).
    :param save_path: Pass None to show the plot without saving.
    """
    fig, axs = plt.subplots(nrows=1, ncols=len(values))
    fig.suptitle(title)

    for a, v, l in zip(axs, values, labels):
        a.boxplot(v, tick_labels=[l])
    plot_or_save(fig, save_path)


def get_fields_names(f: np.ndarray):
    """Extracts the field names from data of shape (N,D). Assumes velocities before pressure ordering."""
    return ['$U_x$', '$U_y$', '$U_z$'][:f.shape[-1] - 1] + ['$p$']


def plot_errors_vs_var(title: str,
                       errors: np.ndarray,
                       var: np.ndarray,
                       labels: list[str],
                       save_path=None):
    """
    Plots the value of multidimensional errors with respect to a variable using a scatter plot and a smoothed trand line.

    The trend is added only if N > 5.
    :param title: The title of the plot.
    :param errors: Error data of shape (N,D).
    :param var: Value of the target variable at each sample, shape (N).
    :param labels: Label of each variable, shape (D).
    :param save_path: Pass None to show the plot without saving.
    """
    n_errors = errors.shape[-1]
    fig, axs = plt.subplots(ncols=1, nrows=n_errors, figsize=(8, 10))
    fig.suptitle(title)

    fields_names = get_fields_names(errors)
    cmap = matplotlib.colormaps['Set2']

    for i in range(n_errors):
        axs[i].scatter(var, errors[:, i], label='Raw', color=cmap(2), s=15)
        axs[i].set_xlabel(labels[0])
        axs[i].set_ylabel(labels[1])

        if len(var) > 5:
            interp = make_smoothing_spline(var, errors[..., i])
            x = np.linspace(min(var), max(var), 100)
            axs[i].plot(x, interp(x), color=cmap(1), label='Interpolated')
        axs[i].legend()
        axs[i].set_title(fields_names[i])

    plt.tight_layout()
    plot_or_save(fig, save_path)


def get_heatmap(mae: np.array, x: np.array, y: np.array) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates a 2D heatmap matrix of mae with respect to two variables x and y.
    :param mae: Errors vector of shape (N).
    :param x: First variable values of shape (N).
    :param y: Second variable values of shape (N).
    :return: a tuple with the matrix and the sorted x and y variables.
    """
    x_unique = np.unique(x)
    y_unique = np.unique(y)[::-1]

    heatmap = np.ones((len(y_unique), len(x_unique))) * np.nan
    for f, x, y in zip(mae, x, y):
        j = (x_unique == x).nonzero()[0]
        i = (y_unique == y).nonzero()[0]
        heatmap[i, j] = f

    return heatmap, x_unique, y_unique


def plot_errors_vs_multi_vars(title: str,
                              errors: np.array,
                              x: np.array,
                              y: np.array,
                              labels: list[str],
                              save_path=None):
    """
    Plots the value of multidimensional errors with respect to a multidimensional variable with a heatmap.

    :param title: The title of the plot.
    :param errors: Errors vector of shape (N).
    :param x: First variable values of shape (N).
    :param y: Second variable values of shape (N).
    :param labels: The variables label.
    :param save_path: Pass None to show the plot without saving.
    """
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


def plot_heatmap(ax: Axes, matrix: np.ndarray, x: np.ndarray, y: np.ndarray, labels: list[str]):
    """
    Plots a heatmap with custom number formatting on ax.
    :param ax: The axis to use for plotting.
    :param matrix: The heatmap matrix created with get_heatmap()-
    :param x: The sorted x-axis variable.
    :param y: The sorted y-axis variable.
    :param labels: The variables label.
    """

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


def plot_per_case(title: str, values, save_path=None):
    """
    Plots multiple variables per case bar plots.
    :param title: The title of the plot.
    :param values: The values to plot of shape (C,D).
    :param save_path: Pass None to show the plot without saving.
    """
    fig = plt.figure(layout='constrained')
    fig.suptitle(title)
    axs = fig.subplots(nrows=values.shape[-1], ncols=1).flatten()
    cmap = plt.get_cmap('Set2')
    labels = get_fields_names(values)
    for i, (ax, f, fname) in enumerate(zip(axs, np.hsplit(values, len(labels)), labels)):
        if min(f) < 0:
            ax.axhline(0, 0, 1, linestyle='--', color='black')
        ax.bar(np.arange(len(f)), f.flatten(), color=cmap(i))
        ax.set_xticks([])
        ax.set_ylabel(f'{fname} MAE')
    plot_or_save(fig, save_path)
