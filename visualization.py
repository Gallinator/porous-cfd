import glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import tri
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
    fluid_zone = np.nonzero(porous == 0)[0]
    ax.scatter(points[porous_zone, 0], points[porous_zone, 1], marker='o', s=25, zorder=1, c='#00000000',
               label='Porous', edgecolors='black')
    ax.scatter(points[..., 0], points[..., 1], s=5, zorder=1, c='black',
               label='Collocation')

    triangulation = tri.Triangulation(points[..., 0], points[..., 1])
    refiner = tri.UniformTriRefiner(triangulation)
    tri_points, tri_field = refiner.refine_field(value.flatten(), subdiv=3)

    plot = ax.tricontourf(tri_points, tri_field, levels=100, zorder=-1, cmap='turbo')

    ax.set_ymargin(0.025)
    ax.set_xmargin(0.02)
    add_colorbar(fig, ax, plot)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')


def plot_uneven_stream(title: str, points: np.array, field: np.array, fig, ax, plot_streamlines=True):
    ax.set_title(title, pad=20)

    triangulation = tri.Triangulation(points[..., 0], points[..., 1])
    refiner = tri.UniformTriRefiner(triangulation)
    tri_points, tri_field = refiner.refine_field(np.linalg.norm(field, axis=1).flatten())
    plot = ax.tricontourf(tri_points, tri_field, levels=100, zorder=-1, cmap='turbo')
    x = points[:, 0].flatten()
    y = points[:, 1].flatten()
    xx = np.linspace(x.min(), x.max(), 50)
    yy = np.linspace(y.min(), y.max(), 50)

    xi, yi = np.meshgrid(xx, yy)
    field_x = field[:, 0].flatten()
    field_y = field[:, 1].flatten()

    g_x = griddata(points, field_x, (xi, yi), method='nearest')
    g_y = griddata(points, field_y, (xi, yi), method='nearest')

    ax.streamplot(xx, yy, g_x, g_y, color='black', density=2, zorder=1)

    add_colorbar(fig, ax, plot)
    ax.set_aspect('equal')


def plot_fields(title: str, points: np.array, u: np.array, p: np.array, porous: np.array or None, plot_streams=True,
                save_path=None):
    fig = plt.figure(figsize=(16, 9), layout='constrained')
    fig.suptitle(title, fontsize=20)
    ax_u_x, ax_u_y, ax_p, ax_u = fig.subplots(ncols=2, nrows=2).flatten()
    # Pressure
    plot_scalar_field(f'$p$ {M2_S2}', points, p, porous, fig, ax_p)

    # Velocity
    plot_scalar_field(f'$u_x$ {M_S}', points, u[:, 0], porous, fig, ax_u_x)

    plot_scalar_field(f'$u_y$ {M_S}', points, u[:, 1], porous, fig, ax_u_y)
    if plot_streams:
        plot_uneven_stream(f'$U$ {M_S}', points, u, fig, ax_u)
    else:
        plot_scalar_field(f'$u_y$ {M_S}', points, np.linalg.norm(u, axis=1), porous, fig, ax_u)

    plot_or_save(fig, save_path)


def plot_case(path: str):
    b_data = data_parser.parse_boundary(path, ['U'], ['p'])
    i_data = parse_internal_mesh(path, "U", "p")
    data = np.vstack([b_data, i_data])

    plot_fields(Path(path).stem,
                data[..., 0:2],
                data[..., 2:4],
                data[..., 4:5],
                data[..., 5:6])


def plot_histogram(ax, data, color: str, title: str, bins=100):
    ax.set_title(title, pad=10)
    ax.hist(data, bins=bins, color=color, edgecolor='black')


def plot_dataset_dist(path: str, save_path=None):
    data = []
    for case in track(list(set(glob.glob(f"{path}/*")) - set(glob.glob(f'{path}/meta.json'))),
                      description="Reading data"):
        b_data = data_parser.parse_boundary(case, ['U'], ['p'])
        i_data = parse_internal_mesh(case, "U", "p")
        data.extend(b_data)
        data.extend(i_data)

    data = np.array(data)
    plot_data_dist(f'{path} distribution', data[..., 2:4], data[..., 4:5], data[..., -1:], save_path)


def plot_data_dist(title, u, p, zones_ids=None, save_path=None):
    ux, uy = u[..., 0], u[..., 1]
    fig = plt.figure(layout='constrained')
    fig.suptitle(title, fontsize=20)
    ax_ux, ax_uy, ax_p, ax_zones = fig.subplots(ncols=2, nrows=2).flatten()

    plot_histogram(ax_ux, ux, 'lightsteelblue', '$U_x$')
    plot_histogram(ax_uy, uy, 'lemonchiffon', '$U_y$')
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
    colors = ['salmon', 'lightblue', 'palegreen']
    labels = [f'$U_x$ {M_S}', f'$U_y$ {M_S}', f'$p$ {M2_S2}']

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
    ax.set_xticks(x + w / 2, ['Momentum x', 'Momentum y', 'Continuity'])
    fig.tight_layout()
    plot_or_save(fig, save_path)


def plot_u_direction_change(data_dir):
    diff = []
    for c in list(set(glob.glob(f'{data_dir}/*')) - set(glob.glob(f'{data_dir}/*.json'))):
        data = parse_internal_mesh(c, 'mag(grad(Unorm))', 'mag(grad(p))')
        diff.append(data[..., 2:4])

    unorm_mean = [np.mean(d[..., 0], axis=0) for d in diff]
    fig = plt.figure(layout='constrained')
    ax_1, ax_2 = fig.subplots(2, 1).flatten()
    ax_1.bar(np.arange(0, len(unorm_mean)), unorm_mean, color='lightblue')
    ax_1.set_title('Average U direction change per case')
    plot_histogram(ax_2, unorm_mean, 'salmon', 'Average U direction change distribution', 20)
    plt.show()
