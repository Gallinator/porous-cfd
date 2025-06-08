import glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from rich.progress import track
from scipy.interpolate import griddata

import data_parser
from data_parser import parse_internal_mesh

plt.style.use('dark_background')


def plot_scalar_field(title: str, points: np.array, value: np.array, fig, ax):
    ax.set_title(title, pad=20)
    plot = ax.scatter(points[:, 0], points[:, 1], s=2, c=value, cmap='turbo')
    fig.colorbar(plot, ax=ax)
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
    fig.colorbar(plot.lines, ax=ax)
    ax.set_aspect('equal')


def plot_fields(title: str, points: np.array, u: np.array, p: np.array, plot_streams=True):
    fig = plt.figure(figsize=(12, 10), layout='constrained')
    fig.suptitle(title, fontsize=20)
    ax_u_x, ax_u_y, ax_p, ax_u = fig.subplots(ncols=2, nrows=2).flatten()
    # Pressure
    plot_scalar_field('$p$ $\left[ \\frac{m^2}{s^2} \\right]$', points, p, fig, ax_p)

    # Velocity
    plot_scalar_field('$u_x$ $\left[ \\frac{m}{s} \\right]$', points, u[:, 0], fig, ax_u_x)

    plot_scalar_field('$u_y$ $\left[ \\frac{m}{s} \\right]$', points, u[:, 1], fig, ax_u_y)

    if plot_streams:
        plot_uneven_stream('$U$ $\left[ \\frac{m}{s} \\right]$', points, u, fig, ax_u)
    else:
        plot_scalar_field('$U$ $\left[ \\frac{m}{s} \\right]$', points, norm(u, axis=1), fig, ax_u)

    plt.show()


def plot_case(path: str):
    boundary_c, boundary_u, boundary_p = data_parser.parse_boundary(path)
    mesh_c, mesh_p, mesh_u = parse_internal_mesh(path, "p", "U")

    plot_fields(Path(path).stem,
                np.vstack([boundary_c, mesh_c]),
                np.vstack([boundary_u, mesh_u]),
                np.vstack([boundary_p, mesh_p]))


def plot_dataset_dist(path: str):
    ux, uy, p, zones = [], [], [], []
    for case in track(list(set(glob.glob(f"{path}/*")) - set(glob.glob(f'{path}/meta.json'))),
                      description="Reading data"):
        b_points, b_u, b_p, b_porous_idx = data_parser.parse_boundary(case)
        i_points, i_u, i_p, i_porous_idx = parse_internal_mesh(case, "U", "p")
        ux += b_u[:, 0].flatten().tolist()
        ux += i_u[:, 0].flatten().tolist()
        uy += b_u[:, 1].flatten().tolist()
        uy += i_u[:, 1].flatten().tolist()
        p += i_p.flatten().tolist()
        p += b_p.flatten().tolist()
        zones += b_porous_idx.flatten().tolist()
        zones += i_porous_idx.flatten().tolist()

    def plot_histogram(ax, data, color: str, title: str, bins=100):
        ax.set_title(title, pad=10)
        ax.hist(data, bins=bins, color=color, edgecolor='black')

    fig = plt.figure(layout='constrained')
    ax_ux, ax_uy, ax_p, ax_zones = fig.subplots(ncols=2, nrows=2).flatten()

    plot_histogram(ax_ux, ux, 'lightsteelblue', '$U_x$')
    plot_histogram(ax_uy, uy, 'lemonchiffon', '$U_y$')
    plot_histogram(ax_p, p, 'lightsalmon', '$p$')
    plot_histogram(ax_zones, zones, 'palegreen', 'Material zones', 2)

    plt.show()
