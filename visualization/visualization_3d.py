import os
import random
from pathlib import Path

import numpy as np
import vtk
import pyvista as pv
from pyvista import Plotter, PolyData, OpenFOAMReader, PointSet

from dataset import data_parser
from visualization.common import M_S, M2_S2


def plot_scalar_field(title, points: np.array, value: np.array, zones_ids, plotter):
    poly_points = PolyData(points)
    colorbar = {'title': title, 'vertical': True, 'position_y': 0.25, 'height': 0.5}
    plotter.add_mesh(poly_points, scalars=value, scalar_bar_args=colorbar, point_size=5.0, cmap='coolwarm')
    plotter.show_grid(all_edges=True)

    plotter.camera.position = np.array((-0.8, -1, 0.5)) * np.max(np.linalg.norm(points, axis=-1)) * 2.5
    plotter.camera.zoom(0.75)


def plot_2d_slice(mesh, normal, origin, plotter, *additional_meshes: tuple):
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

    for m, _ in additional_meshes:
        sliced_mesh = m.slice(normal=normal, origin=origin)
        plotter.add_mesh(sliced_mesh, color='black', line_width=5)

    plotter.enable_parallel_projection()
    match plane:
        case 'xy':
            plotter.view_xy()
        case 'xz':
            plotter.view_xz()
    plotter.show_bounds(location='outer', xtitle='X', ytitle='Y', ztitle='z')


def plot_3d_streamlines(interp_mesh, inlet_mesh, plotter, additional_meshes: dict):
    stream_start_points = np.array(inlet_mesh.points)
    min_x = np.min(inlet_mesh.points, axis=0)[0]
    stream_start_points = stream_start_points[stream_start_points[..., 0] == min_x]
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
    for m, c in additional_meshes:
        plotter.add_mesh(m, color=c)

    plotter.camera.position = np.array((-0.8, -1, 0.5)) * np.max(np.linalg.norm(interp_mesh.points, axis=-1)) * 2.5
    plotter.camera.zoom(0.5)
    plotter.show_bounds(location='outer', xtitle='X', ytitle='Y', ztitle='z')


def plot_streamlines(title, case_dir, points: np.array, u: np.array, additional_meshes: dict[str, str], save_path=None):
    empty_foam = f'{case_dir}/empty.foam'
    open(empty_foam, 'w').close()

    foam_reader = OpenFOAMReader(empty_foam)
    foam_reader.set_active_time_value(foam_reader.time_values[-1])
    foam_reader.cell_to_point_creation = True

    mesh = foam_reader.read()
    add_objects = [pv.get_reader(f'{case_dir}/constant/triSurface/{m}.obj').read() for m in additional_meshes.keys()]
    add_objects = list(zip(add_objects, additional_meshes.values()))

    data_points = PolyData(points)
    data_points['Uinterp'] = u
    internal_mesh = mesh['internalMesh']
    interp_mesh = internal_mesh.interpolate(data_points, radius=5)

    plotter = Plotter(shape=(1, 3), off_screen=save_path is not None, window_size=[3840, 1440])

    plotter.subplot(0, 0)
    plot_3d_streamlines(interp_mesh, mesh['boundary']['inlet'], plotter, add_objects)

    plotter.subplot(0, 1)
    center = (0, 0, add_objects[0][0].center[2] if len(add_objects) > 0 else 1)
    plot_2d_slice(interp_mesh, 'z', center, plotter, *add_objects)

    plotter.subplot(0, 2)
    center = (0, 0, add_objects[0][0].center[1] if len(add_objects) > 0 else 1)
    plot_2d_slice(interp_mesh, 'y', center, plotter, *add_objects)

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
