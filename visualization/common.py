import glob
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pandas
from matplotlib import pyplot as plt
from numpy.linalg import norm
from rich.progress import track

from dataset import data_parser
from dataset.data_parser import parse_internal_fields

M_S = '\left[ \\frac{m}{s} \\right]'
M2_S2 = '\left[ \\frac{m^2}{s^2} \\right]'


def plot_fields(title: str, points: np.array, u: np.array, p: np.array, porous: np.array or None, plot_streams=True,
                save_path=None):
    return NotImplemented


def plot_or_save(fig, save_path):
    if fig._suptitle is not None:
        file_name = fig._suptitle.get_text()
    else:
        file_name = fig.axes[0].get_title()

    if save_path is not None:
        plt.savefig(f'{save_path}/{file_name}.png', transparent=True)
        plt.close(fig)
    else:
        plt.show()


def plot_case(path: str):
    fields = data_parser.parse_case_fields(path, 'C', 'U', 'p', 'cellToRegion')
    plot_fields(Path(path).stem,
                fields['C'].to_numpy(),
                fields['U'].to_numpy(),
                fields['p'].to_numpy(),
                fields['cellToRegion'])


def plot_histogram(ax, data, color: str, title: str, bins=100):
    ax.set_title(title, pad=10)
    ax.hist(data, bins=bins, color=color, edgecolor='black')


def plot_dataset_dist(path: str, save_path=None):
    data = []
    for case in track(list(set(glob.glob(f"{path}/*/")) - set(glob.glob(f'{path}/meta.json'))),
                      description="Reading data"):
        case_data = data_parser.parse_case_fields(case, 'U', 'p', 'cellToRegion')
        data.append(case_data)

    data = pandas.concat(data)
    plot_data_dist(f'{path} distribution', data['U'], data['p'], data['cellToRegion'], save_path)


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


def plot_residuals(*args, trim, save_path=None):
    fig, ax = plt.subplots()
    colors = ['salmon', 'lightblue']
    labels = ['PINN', 'OpenFoam']
    u_dims = len(args[0]) - 1
    ax.set_title(f'Absolute average residuals (trimmed {trim})', pad=10)
    w = 0.01
    x = np.array([x * 0.03 for x in range(len(args[0]))])

    for i, d in enumerate(args):
        rects = ax.bar(x + i * w, d, w, color=colors[i], label=labels[i])
        ax.bar_label(rects, fmt='%.2e', padding=10)

    ax.legend()
    ax.set_ylim(0, max([max(d) for d in args]) * 1.1)
    labels = ['Momentum x', 'Momentum y', 'Momentum z'][:u_dims] + ['Continuity']
    ax.set_xticks(x + w / 2, labels)
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
