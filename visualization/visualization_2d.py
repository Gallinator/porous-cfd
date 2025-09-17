import matplotlib.pyplot as plt
import numpy as np
from matplotlib import tri
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata

from visualization.common import M2_S2, M_S, plot_or_save


def add_colorbar(fig, ax, plot):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    fig.colorbar(plot, cax=cax)


def plot_scalar_field(title: str, points: np.array, value: np.array, porous: np.array or None, fig, ax):
    ax.set_title(title, pad=20)
    porous_zone = np.nonzero(porous > 0)[0]
    ax.scatter(points[porous_zone, 0], points[porous_zone, 1], marker='o', s=25, zorder=1, c='#00000000',
               label='Porous', edgecolors='black')
    ax.scatter(points[..., 0], points[..., 1], s=5, zorder=1, c='black',
               label='Collocation')

    triangulation = tri.Triangulation(points[..., 0], points[..., 1])
    refiner = tri.UniformTriRefiner(triangulation)
    tri_points, tri_field = refiner.refine_field(value.flatten(), subdiv=3)

    plot = ax.tricontourf(tri_points, tri_field, levels=100, zorder=-1, cmap='coolwarm')

    ax.set_ymargin(0.025)
    ax.set_xmargin(0.02)
    add_colorbar(fig, ax, plot)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')


def plot_uneven_stream(title: str, points: np.array, field: np.array, fig, ax):
    ax.set_title(title, pad=20)

    triangulation = tri.Triangulation(points[..., 0], points[..., 1])
    refiner = tri.UniformTriRefiner(triangulation)
    tri_points, tri_field = refiner.refine_field(np.linalg.norm(field, axis=1).flatten())
    plot = ax.tricontourf(tri_points, tri_field, levels=100, zorder=-1, cmap='coolwarm')
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
    ax.set_ymargin(0)
    add_colorbar(fig, ax, plot)
    ax.set_aspect('equal')


def plot_fields(title: str, points: np.array, u: np.array, p: np.array, porous: np.array or None, plot_streams=True,
                save_path=None):
    fig = plt.figure(figsize=(16, 9), layout='constrained')
    fig.suptitle(title, fontsize=20)
    ax_u_x, ax_u_y, ax_p, ax_u = fig.subplots(ncols=2, nrows=2).flatten()
    # Pressure
    plot_scalar_field(f'$p {M2_S2}$', points, p, porous, fig, ax_p)

    # Velocity
    plot_scalar_field(f'$u_x {M_S}$', points, u[:, 0], porous, fig, ax_u_x)

    plot_scalar_field(f'$u_y {M_S}$', points, u[:, 1], porous, fig, ax_u_y)
    if plot_streams:
        plot_uneven_stream(f'$U {M_S}$', points, u, fig, ax_u)
    else:
        plot_scalar_field(f'$U {M_S}$', points, np.linalg.norm(u, axis=1), porous, fig, ax_u)

    plot_or_save(fig, save_path)
