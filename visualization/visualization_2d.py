from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import tri
from matplotlib.axes import Axes
from matplotlib.colorizer import _ScalarMappable, ColorizingArtist
from matplotlib.figure import Figure
from matplotlib.tri import Triangulation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata

from dataset import data_parser
from visualization.common import M2_S2, M_S, plot_or_save


def add_colorbar(fig: Figure, ax: Axes, plot: _ScalarMappable | ColorizingArtist):
    """
    Adds a colorbar to fig on a separate axis.
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    fig.colorbar(plot, cax=cax)


def mask_triangulation(triangulation: Triangulation, mask: list[list[float]], points: np.ndarray):
    """
    Masks triangulations required by tricontour plots in-place.

    Supports only rectangular masks encoded as bounding boxes [(bottom, left), (top, right)].
    :param triangulation: The triangulation to mask.
    :param mask: A list of rectangular bounding boxes.
    :param points: The source points used for the triangulation, shape (N,2).
    """
    mask_full = np.full((len(triangulation.triangles),), False)
    for m in mask:
        tri_centers = points[triangulation.triangles].mean(axis=1)
        inside = np.logical_and(tri_centers > np.array(m[0]), tri_centers < np.array(m[1]))
        inside = np.all(inside, axis=-1)
        mask_full = np.logical_or(mask_full, inside)
    triangulation.set_mask(mask_full)


def plot_scalar_field(title: str,
                      points: np.ndarray,
                      value: np.ndarray,
                      porous_id: np.ndarray,
                      fig: Figure, ax: Axes,
                      mask: list[list[float]] = None):
    """
    Creates a contour plot from a scalar field.

    Might produce artifacts due to interpolation. Supports masking of bounding boxes.
    The porous region is highlighted using the porous_id array, whose values are zero (fluid point) or one (porous point).
    :param title: The title of the subplot.
    :param points: Array of points of shape (N,2).
    :param value: Array of scalar values of shape (N,1).
    :param porous_id: The porous indicator variable.
    :param fig: The matplotlib Figure.
    :param ax: The axis to add the plot to.
    :param mask: Optional bounding box mask.
    """
    ax.set_title(title, pad=20)
    porous_zone = np.nonzero(porous_id > 0)[0]
    ax.scatter(points[porous_zone, 0], points[porous_zone, 1], marker='o', s=25, zorder=1, c='#00000000',
               label='Porous', edgecolors='black')
    ax.scatter(points[..., 0], points[..., 1], s=5, zorder=1, c='black',
               label='Collocation')

    triangulation = tri.Triangulation(points[..., 0], points[..., 1])
    if mask:
        mask_triangulation(triangulation, mask, points)

    refiner = tri.UniformTriRefiner(triangulation)
    tri_points, tri_field = refiner.refine_field(value.flatten(), subdiv=3)

    plot = ax.tricontourf(tri_points, tri_field, levels=100, zorder=-1, cmap='coolwarm')

    ax.set_ymargin(0.025)
    ax.set_xmargin(0.02)
    add_colorbar(fig, ax, plot)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')


def plot_uneven_stream(title: str,
                       points: np.ndarray,
                       field: np.ndarray,
                       fig: Figure,
                       ax: Axes,
                       mask: list[list[float]] = None):
    """
    Adds a stream plot to ax from an uneven grid of values.
    :param title: The title of the subplot.
    :param points: Array of points, shape (N,2).
    :param field: The values of the field, shape (N,2).
    :param fig: The matplotlib figure.
    :param ax: The axis to add the plot to.
    :param mask: Optional bounding box mask.
    """
    ax.set_title(title, pad=20)

    triangulation = tri.Triangulation(points[..., 0], points[..., 1])
    if mask:
        mask_triangulation(triangulation, mask, points)
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

    if mask:
        p_x, p_y = np.vstack(xi.flatten()), np.vstack(yi.flatten())
        p = np.concatenate([p_x, p_y], axis=-1)
        mask_full = np.full((len(p_x),), False)
        for m in mask:
            inside = np.logical_and(p > np.array(m[0]), p < np.array(m[1]))
            inside = np.all(inside, axis=-1)
            mask_full = np.logical_or(mask_full, inside)
        mask_full = mask_full.reshape(xi.shape)
        g_x[mask_full] = np.nan
        g_y[mask_full] = np.nan

    ax.streamplot(xx, yy, g_x, g_y, color='black', density=2, zorder=1)
    ax.set_ymargin(0)
    add_colorbar(fig, ax, plot)
    ax.set_aspect('equal')


def plot_fields(title: str,
                points: np.ndarray,
                u: np.ndarray,
                p: np.ndarray,
                porous_id: np.ndarray,
                plot_streams=True,
                save_path=None,
                mask=None):
    """
    Plots or saves a velocity and pressure vector field using contour plots and streamplots.

    Allows to replace the streamplots with a velocity magnitude contour plot, useful for plotting errors.
    Supports bounding box masking. Pass None to save_path to show the plot instead of saving it to a .png image.
    The porous region is highlighted using the porous_id array, whose values are zero (fluid point) or one (porous point).
    Supports only rectangular masks encoded as bounding boxes [(bottom, left), (top, right)].
    The filename is taken from the plot title.
    :param title: The main title of the plot.
    :param points: Array of points of shape (N,2).
    :param u: Array of velocity vectors, shape (N,2).
    :param p: Array of pressure values, shape (N,1).
    :param porous_id: Array of porous ids, shape (N,1).
    :param plot_streams: Set to Fals to plot the velocity magnitude instead of the streamlines.
    :param save_path: Parent folder to save the plot image into. Pass None to show the path without saving.
    :param mask: Optional bounding box mask.
    """
    domain_size = [max(points[:, 0]) - min(points[:, 0]), max(points[:, 1]) - min(points[:, 1])]
    domain_max_size = max(domain_size)
    domain_size_normalized = [domain_size[0] / domain_max_size, domain_size[1] / domain_max_size]
    fig = plt.figure(figsize=(16 * domain_size_normalized[0] * 1.1, 16 * domain_size_normalized[1]),
                     layout='constrained')
    fig.suptitle(title, fontsize=20)
    ax_u_x, ax_u_y, ax_p, ax_u = fig.subplots(ncols=2, nrows=2).flatten()
    # Pressure
    plot_scalar_field(f'$p {M2_S2}$', points, p, porous_id, fig, ax_p, mask)

    # Velocity
    plot_scalar_field(f'$u_x {M_S}$', points, u[:, 0], porous_id, fig, ax_u_x, mask)

    plot_scalar_field(f'$u_y {M_S}$', points, u[:, 1], porous_id, fig, ax_u_y, mask)
    if plot_streams:
        plot_uneven_stream(f'$U {M_S}$', points, u, fig, ax_u, mask)
    else:
        plot_scalar_field(f'$U {M_S}$', points, np.linalg.norm(u, axis=1), porous_id, fig, ax_u, mask)

    plot_or_save(fig, save_path)


def plot_case(path: str, save_path=None):
    """
    Creates contour and streamlines plots of an OpenFOAM case, with optional saving to a .png image.

    The filename of the saved plot is taken from the title.
    :param path: The OpenFOAM case path.
    :param save_path: directory to save the plot into. Pass None to show the plot instead of saving.
    """
    fields = data_parser.parse_case_fields(path, 'C', 'U', 'p', 'cellToRegion')
    plot_fields(Path(path).stem,
                fields['C'].to_numpy()[..., 0:2],
                fields['U'].to_numpy()[..., 0:2],
                fields['p'].to_numpy(),
                fields['cellToRegion'],
                save_path=save_path)
