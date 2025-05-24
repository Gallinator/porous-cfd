from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from scipy.interpolate import griddata

import data_parser
from data_parser import parse_internal_mesh

plt.style.use('dark_background')


def plot_scalar_field(title: str, points: np.array, value: np.array, porous: np.array or None, fig, ax):
    ax.set_title(title, pad=20)
    plot = ax.scatter(points[:, 0], points[:, 1], s=porous + 1 * 0.5, c=value, cmap='turbo')
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


def plot_fields(title: str, points: np.array, u: np.array, p: np.array, porous: np.array or None):
    fig = plt.figure(figsize=(12, 10), layout='constrained')
    fig.suptitle(title, fontsize=20)
    ax_u_x, ax_u_y, ax_p, ax_u = fig.subplots(ncols=2, nrows=2).flatten()
    # Pressure
    plot_scalar_field('$p$ $\left[ \\frac{m^2}{s^2} \\right]$', points, p, porous, fig, ax_p)

    # Velocity
    plot_scalar_field('$u_x$ $\left[ \\frac{m}{s} \\right]$', points, u[:, 0], porous, fig, ax_u_x)

    plot_scalar_field('$u_y$ $\left[ \\frac{m}{s} \\right]$', points, u[:, 1], porous, fig, ax_u_y)

    plot_uneven_stream('$U$ $\left[ \\frac{m}{s} \\right]$', points, u, fig, ax_u)

    plt.show()


def plot_case(path: str):
    boundary_c, boundary_u, boundary_p = data_parser.parse_boundary(path)
    mesh_c, mesh_p, mesh_u, po = parse_internal_mesh(path, "p", "U")
    porous = np.vstack([np.zeros(boundary_p.shape), po])

    plot_fields(Path(path).stem,
                np.vstack([boundary_c, mesh_c]),
                np.vstack([boundary_u, mesh_u]),
                np.vstack([boundary_p, mesh_p]),
                porous)

plot_case('data/train/s1-1_r180_semi_circle')
