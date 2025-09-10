import glob
import os
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas
from numpy.linalg import norm
import pyvista as pv
from pyvista import Plotter, PolyData, OpenFOAMReader, PointSet
from rich.progress import track
from dataset import data_parser

M_S = '\left[ \\frac{m}{s} \\right]'
M2_S2 = '\left[ \\frac{m^2}{s^2} \\right]'


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


def plot_scalar_field(title, points: np.array, value: np.array, zones_ids, plotter):
    points = PolyData(points)
    colorbar = {'title': title, 'vertical': True, 'position_y': 0.25, 'height': 0.5}
    plotter.add_mesh(points, scalars=value, scalar_bar_args=colorbar, point_size=5.0, cmap='coolwarm')
    plotter.show_grid(all_edges=True)
    plotter.camera.position = (-80, -100, 50)
    plotter.camera.zoom(0.75)


def plot_2d_slice(mesh, tree, solid, normal, origin, plotter):
    mesh_slice = mesh.slice(normal=normal, origin=origin)
    u_slice = np.copy(mesh_slice['Uinterp'])
    u_slice[..., -1 if normal == 'z' else -2] = 0
    mesh_slice['Uslice'] = u_slice
    plane = 'xy' if normal == 'z' else 'xz'
    colorbar = {'title': f'$U {plane} {M_S}$', 'position_x': 0.25, 'height': 0.05, 'width': 0.5}
    plotter.add_mesh(mesh_slice,
                     cmap='coolwarm',
                     scalars='Uslice',
                     scalar_bar_args=colorbar)

    sliced_tree = tree.slice(normal=normal, origin=origin)
    plotter.add_mesh(sliced_tree, color='black', line_width=5)

    sliced_solid = solid.slice(normal=normal, origin=origin)
    plotter.add_mesh(sliced_solid, color='black', line_width=5)
    plotter.enable_parallel_projection()
    match plane:
        case 'xy':
            plotter.view_xy()
        case 'xz':
            plotter.view_xz()
    plotter.show_bounds(location='outer', xtitle='X', ytitle='Y', ztitle='z')


def plot_3d_streamlines(interp_mesh, inlet_mesh, tree, solid, plotter):
    stream_start_points = np.array(inlet_mesh.points)
    stream_start_points = stream_start_points[stream_start_points[..., 0] == -20]
    stream_start_points = PointSet(random.choices(stream_start_points, k=250))
    colorbar = {'title': f'$U {M_S}$', 'position_x': 0.25, 'height': 0.05, 'width': 0.5}
    streamlines = interp_mesh.streamlines_from_source(stream_start_points, vectors='Uinterp')
    plotter.add_mesh(streamlines, render_lines_as_tubes=False,
                     scalar_bar_args=colorbar,
                     lighting=False,
                     scalars='Uinterp',
                     line_width=1,
                     cmap='coolwarm')

    # Add solid meshes
    plotter.add_mesh(tree, color='mediumseagreen')
    plotter.add_mesh(solid, color='oldlace')
    plotter.camera.position = (-80, -100, 50)
    plotter.camera.zoom(0.5)
    plotter.show_bounds(location='outer', xtitle='X', ytitle='Y', ztitle='z')


def plot_streamlines(title, case_dir, points: np.array, u: np.array, save_path=None):
    empty_foam = f'{case_dir}/empty.foam'
    open(empty_foam, 'w').close()

    foam_reader = OpenFOAMReader(empty_foam)
    foam_reader.set_active_time_value(foam_reader.time_values[-1])
    foam_reader.cell_to_point_creation = True

    mesh = foam_reader.read()
    tree = pv.get_reader(f'{case_dir}/constant/triSurface/mesh.obj').read()
    solid = pv.get_reader(f'{case_dir}/constant/triSurface/solid.obj').read()

    data_points = PolyData(points)
    data_points['Uinterp'] = u
    internal_mesh = mesh['internalMesh']
    interp_mesh = internal_mesh.interpolate(data_points, radius=5)

    plotter = Plotter(shape=(1, 3), off_screen=save_path is not None, window_size=[3840, 1440])

    plotter.subplot(0, 0)
    plot_3d_streamlines(interp_mesh, mesh['boundary']['inlet'], tree, solid, plotter)

    plotter.subplot(0, 1)
    plot_2d_slice(interp_mesh, tree, solid, 'z', (0, 0, solid.center[2]), plotter)

    plotter.subplot(0, 2)
    plot_2d_slice(interp_mesh, tree, solid, 'y', (0, solid.center[1], 0), plotter)

    plotter.show(screenshot=f'{save_path}/{title}.png' if save_path else False)
    os.remove(empty_foam)


def plot_houses(title, points: np.ndarray, u: np.ndarray, p: np.ndarray, house_mesh_path, save_path=None):
    house = pv.get_reader(house_mesh_path).read()
    data = PolyData(points)
    data['Uinterp'] = u
    data['pinterp'] = p

    plotter = Plotter(shape=(1, 2), off_screen=save_path is not None, window_size=[3840, 1440])

    colorbar = {'title': title, 'vertical': True, 'position_y': 0.25, 'height': 0.5}

    plotter.subplot(0, 0)
    plotter.add_mesh(house, scalar_bar_args=colorbar, color='oldlace')
    plot_scalar_field(f'U error ${M_S}$', points, np.linalg.norm(u, axis=1), None, plotter)

    plotter.subplot(0, 1)
    plotter.add_mesh(house, scalar_bar_args=colorbar, color='oldlace')
    plot_scalar_field(f'p error ${M2_S2}$', points, p, None, plotter)

    plotter.show(screenshot=f'{save_path}/{title}.png' if save_path else False)


def plot_fields(title, points: np.array, u: np.array, p: np.array, porous: np.array or None, save_path=None):
    plotter = Plotter(shape=(2, 2), off_screen=save_path is not None, window_size=[2500, 1080])

    # Pressure
    plotter.subplot(1, 1)
    plot_scalar_field(rf'$p {M2_S2}$', points, p, porous, plotter)
    # Velocity
    plotter.subplot(0, 0)
    plot_scalar_field(rf'$u_x {M_S}$', points, u[:, 0], porous, plotter)
    plotter.subplot(0, 1)
    plot_scalar_field(rf'$u_y {M_S}$', points, u[:, 1], porous, plotter)
    plotter.subplot(1, 0)
    plot_scalar_field(rf'$u_z {M_S}$', points, u[:, 2], porous, plotter)

    plotter.show(screenshot=f'{save_path}/{title}.png' if save_path else False)


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
