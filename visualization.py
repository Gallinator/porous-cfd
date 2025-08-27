import glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.linalg import norm
from rich.progress import track
from scipy.interpolate import griddata

import data_parser
from data_parser import parse_internal_mesh

M_S = '$\left[ \\frac{m}{s} \\right]$'
M2_S2 = '$\left[ \\frac{m^2}{s^2} \\right]$'


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


def add_colorbar(fig, ax, plot):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    fig.colorbar(plot, cax=cax)


def plot_scalar_field(title: str, points: np.array, value: np.array, porous: np.array or None, fig, ax):
    ax.set_title(title, pad=20)
    porous_zone = np.nonzero(porous > 0)[0]
    ax.scatter(points[porous_zone, 0], points[porous_zone, 1], points[porous_zone, 2], marker='o', s=50, zorder=-1,
               c='#ffffffff',
               label='Porous', edgecolors='black')
    plot = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=value, s=2, cmap='turbo')

    ax.set_ymargin(0.025)
    ax.set_xmargin(0.02)
    fig.colorbar(plot, ax=ax)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')


def plot_uneven_stream(title: str, points: np.array, field: np.array, fig, ax):
    ax.set_title(title, pad=20)
    x = points[:, 0].flatten()
    y = points[:, 1].flatten()
    xx = np.linspace(x.min(), x.max(), 50)
    yy = np.linspace(y.min(), y.max(), 50)

    xi, yi = np.meshgrid(xx, yy)
    field_x = field[:, 0].flatten()
    field_y = field[:, 1].flatten()
    field_s = norm(field, axis=1).flatten()
    g_x = griddata(points, field_x, (xi, yi), method='nearest')
    g_y = griddata(points, field_y, (xi, yi), method='nearest')
    g_s = griddata(points, field_s, (xi, yi), method='nearest')

    plot = ax.streamplot(xx, yy, g_x, g_y, color=g_s, density=2, cmap='turbo')
    add_colorbar(fig, ax, plot.lines)
    ax.set_ymargin(0)
    ax.set_aspect('equal')


def plot_fields(title: str, points: np.array, u: np.array, p: np.array, porous: np.array or None, plot_streams=True,
                save_path=None):
    fig = plt.figure(figsize=(16, 9), layout='constrained')
    fig.suptitle(title, fontsize=20)
    ax_u_x, ax_u_y, ax_u_z, ax_p = (fig.add_subplot(2, 2, 1, projection='3d'),
                                    fig.add_subplot(2, 2, 2, projection='3d'),
                                    fig.add_subplot(2, 2, 3, projection='3d'),
                                    fig.add_subplot(2, 2, 4, projection='3d'))
    # Pressure
    plot_scalar_field(f'$p$ {M2_S2}', points, p, porous, fig, ax_p)

    # Velocity
    plot_scalar_field(f'$u_x$ {M_S}', points, u[:, 0], porous, fig, ax_u_x)

    plot_scalar_field(f'$u_y$ {M_S}', points, u[:, 1], porous, fig, ax_u_y)

    plot_scalar_field(f'$u_z$ {M_S}', points, u[:, 2], porous, fig, ax_u_z)

    plot_or_save(fig, save_path)


def plot_case(path: str):
    b_data = data_parser.parse_boundary(path, ['U'], ['p'])
    b_data = np.concatenate(list(b_data.values()))
    i_data = parse_internal_mesh(path, "U", "p")
    data = np.vstack([b_data, i_data])

    plot_fields(Path(path).stem,
                data[..., 0:3],
                data[..., 3:6],
                data[..., 6:7],
                data[..., 7:8])


def plot_histogram(ax, data, color: str, title: str, bins=100):
    ax.set_title(title, pad=10)
    ax.hist(data, bins=bins, color=color, edgecolor='black')


def plot_dataset_dist(path: str, save_path=None):
    data = []
    for case in track(list(set(glob.glob(f"{path}/*")) - set(glob.glob(f'{path}/meta.json'))),
                      description="Reading data"):
        b_data = data_parser.parse_boundary(case, ['U'], ['p'])
        b_data = np.concatenate(list(b_data.values()))
        i_data = parse_internal_mesh(case, "U", "p")
        data.extend(b_data)
        data.extend(i_data)

    data = np.array(data)
    plot_data_dist(f'{path} distribution', data[..., 3:6], data[..., 6:7], data[..., -1:], save_path)


def plot_data_dist(title, u, p, zones_ids=None, save_path=None):
    ux, uy, uz = u[..., 0], u[..., 1], u[..., 2]
    fig = plt.figure(layout='constrained')
    fig.suptitle(title, fontsize=20)
    ax_ux, ax_uy, ax_uz, ax_p, ax_zones, _ = fig.subplots(ncols=3, nrows=2).flatten()

    plot_histogram(ax_ux, ux, 'lightsteelblue', '$U_x$')
    plot_histogram(ax_uy, uy, 'lemonchiffon', '$U_y$')
    plot_histogram(ax_uz, uz, 'thistle', '$U_z$')
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


def plot_errors(*args, save_path=None):
    fig, ax = plt.subplots()
    colors = ['salmon', 'lightblue', 'palegreen', 'moccasin']
    labels = [f'$U_x$ {M_S}', f'$U_y$ {M_S}', f'$U_z$ {M_S}', f'$p$ {M2_S2}']

    plot_barh(ax, 'Average relative error', *args, labels, colors)

    fig.tight_layout()
    plot_or_save(fig, save_path)


def plot_residuals(*args, trim, save_path=None):
    fig, ax = plt.subplots()
    colors = ['salmon', 'lightblue']
    labels = ['PINN', 'OpenFoam']
    ax.set_title(f'Absolute average residuals (trimmed {trim})', pad=10)
    w = 0.01
    x = np.array([x * 0.03 for x in range(len(args[0]))])

    for i, d in enumerate(args):
        rects = ax.bar(x + i * w, d, w, color=colors[i], label=labels[i])
        ax.bar_label(rects, fmt='%.2e', padding=10)

    ax.legend()
    ax.set_ylim(0, max([max(d) for d in args]) * 1.1)
    ax.set_xticks(x + w / 2, ['Momentum x', 'Momentum y', 'Momentum z', 'Continuity'])
    fig.tight_layout()
    plot_or_save(fig, save_path)
