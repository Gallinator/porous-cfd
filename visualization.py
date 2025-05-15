import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
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


def plot_case(path: str):
    boundary_c, boundary_u, boundary_p = data_parser.parse_boundary(path)
    mesh_c, mesh_p, mesh_u = parse_internal_mesh(path, "p", "U")

    fig = plt.figure(figsize=(12, 10), layout='constrained')
    ax_u_x, ax_u_y, ax_p, ax_u = fig.subplots(ncols=2, nrows=2).flatten()
    mesh = np.vstack([boundary_c, mesh_c])
    # Pressure
    p = np.concatenate([boundary_p.flatten(), mesh_p.flatten()])
    plot_scalar_field('$p$ $\left[ \\frac{m^2}{s^2} \\right]$', mesh, p, fig, ax_p)

    # Velocity
    u_x = np.concatenate([boundary_u[:, 0], mesh_u[:, 0]])
    plot_scalar_field('$u_x$ $\left[ \\frac{m}{s} \\right]$', mesh, u_x, fig, ax_u_x)

    u_y = np.concatenate([boundary_u[:, 1], mesh_u[:, 1]])
    plot_scalar_field('$u_y$ $\left[ \\frac{m}{s} \\right]$', mesh, u_y, fig, ax_u_y)

    plot_uneven_stream('$U$ $\left[ \\frac{m}{s} \\right]$', mesh, np.vstack([boundary_u, mesh_u]), fig, ax_u)

    plt.show()
