import os
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import vtk
import pyvista as pv
from pyvista import Plotter, PolyData, OpenFOAMReader, PointSet, DataSet, UnstructuredGrid
from dataset import data_parser
from visualization.common import M_S, M2_S2

pv.global_theme.transparent_background = True


def plot_scalar_field(title: str, points: np.ndarray, value: np.ndarray, porous_id: np.ndarray, plotter: Plotter):
    """
    Creates a 3D scatter plot from a scalar field.
    :param title: The title of the subplot.
    :param points: Array of points of shape (N,3).
    :param value: Array of scalar values of shape (N,1).
    :param porous_id: Binary vector of porous indicator values, shape (N,1). Reserved for future use.
    :param plotter: The PyVista plotter to add this plot to.
    """
    poly_points = PolyData(points)
    colorbar = {'title': title, 'vertical': True, 'position_y': 0.25, 'height': 0.5}
    plotter.add_mesh(poly_points, scalars=value, scalar_bar_args=colorbar, point_size=5.0, cmap='coolwarm')
    bar = plotter.scalar_bars[title]
    bar.GetTitleTextProperty().SetLineSpacing(1.5)
    plotter.show_grid(all_edges=True)

    plotter.camera.position = np.array((-0.8, -1, 0.5)) * np.max(np.linalg.norm(points, axis=-1)) * 2.5
    plotter.camera.zoom(0.75)
    plotter.disable_shadows()


def plot_2d_slice(mesh: UnstructuredGrid,
                  field: str,
                  label: str,
                  origin: Sequence[float],
                  plotter: Plotter,
                  cur_pos: list[int],
                  *additional_meshes: dict[DataSet:str]):
    """
    Plots 2D slices of a 3D field.

    Supports outlines of additional 3D objects specified as a dictionary of PyVista datasets and colors.
    :param mesh: The mesh to plot.
    :param field: The name of the field to plot..
    :param label: Label of the subplot.
    :param origin: TOrigin point of the slicing planes.
    :param plotter: PyVist plotter to add the subplot to.
    :param cur_pos: Position of the subplot in the parent plot grid.
    :param additional_meshes: Additional meshes to add to the plot.
    """
    slices = mesh.slice_orthogonal(x=origin[0], y=origin[1], z=origin[2])
    planes = ['yz', 'xz', 'xy']
    sliced_meshes = []
    for m, _ in additional_meshes:
        sliced_m = m.slice_orthogonal(x=origin[0], y=origin[1], z=origin[2])
        sliced_meshes.append(sliced_m)

    for i, (s, p) in enumerate(zip(slices, planes)):
        plotter.subplot(cur_pos[0], i + cur_pos[1])
        title = f'${label}_{{{planes[i]}}} \\quad {M_S}$'
        colorbar = {'title': title, 'position_x': 0.25, 'height': 0.05, 'width': 0.5}
        plotter.add_mesh(s, cmap='coolwarm', scalars=field, scalar_bar_args=colorbar, lighting=False)
        bar = plotter.scalar_bars[title]
        bar.GetTitleTextProperty().SetLineSpacing(1.5)

        for m in sliced_meshes:
            if len(m[i].points > 0):
                plotter.add_mesh(m[i], color='black', line_width=5)

        plotter.enable_parallel_projection()
        match p:
            case 'xy':
                plotter.view_xy()
            case 'xz':
                plotter.view_xz()
            case 'yz':
                plotter.view_yz()
        plotter.show_bounds(location='outer', xtitle='X', ytitle='Y', ztitle='z')
        plotter.disable_shadows()


def plot_3d_streamlines(interp_mesh: UnstructuredGrid,
                        inlet_mesh: PolyData,
                        plotter: Plotter,
                        additional_meshes: dict[DataSet:str]):
    """
    Plots the 3D velocity streamlines.

    Supports additional 3D objects specified as a dictionary of PyVista datasets and colors.
    :param interp_mesh: The interpolated domain mesh to use for plotting.
    :param inlet_mesh: The inlet mesh used to generate the streamlines starting points
    :param plotter: The PyVista plotter to add the subplot to.
    :param additional_meshes: Additional meshes to add to the plot.
    """
    stream_start_points = np.array(inlet_mesh.points)
    min_x = np.min(inlet_mesh.points, axis=0)[0]
    stream_start_points = stream_start_points[stream_start_points[..., 0] == min_x]
    stream_start_points = PointSet(random.choices(stream_start_points, k=250))
    colorbar = {'title': f'$U \quad {M_S}$', 'position_x': 0.25, 'height': 0.05, 'width': 0.5}
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


def plot_streamlines(title: str,
                     case_dir: str,
                     points: np.ndarray,
                     u: np.ndarray,
                     p: np.ndarray,
                     additional_meshes: dict[str, str],
                     save_path=None,
                     interp_radius=0.1):
    """
    Plots or saves the velocity streamlines and sliced velocity and pressure fields.

    It is possible to add 3D objects to the plots by passing the names of the .obj files located in the constant/triSurface case directory and the color of each object.
    For a list of available color see the `PyVista colors`_.
    The sliced are created on the xy, xz and yz planes starting from the center of the first additional mesh.
    The filename is taken from the plot title.

    .. _PyVista colors: https://docs.pyvista.org/api/utilities/named_colors.html
    :param title: The main title of the plot.
    :param case_dir: The directory of the OpenFOAM case to plot.
    :param points: Array of domain points, shape (N,3).
    :param u: Array of velocity values, shape (N,3).
    :param p: Array of pressure values, shape (N,1).
    :param additional_meshes: Additional objects to add to the plots.
    :param save_path: The directory to save the plot image into. Pass None to show the plot instead of saving.
    :param interp_radius: Radius used to interpolate the sampled points onto the full OpenFOAM mesh.
    """
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
    data_points['pinterp'] = p
    internal_mesh = mesh['internalMesh']
    interp_mesh = internal_mesh.interpolate(data_points, radius=interp_radius)

    plotter = Plotter(shape=(2, 4), off_screen=save_path is not None, window_size=[4096, 3000])

    plotter.subplot(0, 0)
    plot_3d_streamlines(interp_mesh, mesh['boundary']['inlet'], plotter, add_objects)

    center = (0., 0., add_objects[0][0].center[2] if len(add_objects) > 0 else 1.)
    plot_2d_slice(interp_mesh, 'Uinterp', 'U', center, plotter, (0, 1), *add_objects)
    plot_2d_slice(interp_mesh, 'pinterp', 'p', center, plotter, (1, 0), *add_objects)

    plotter.show(screenshot=f'{save_path}/{title}.png' if save_path else False)
    os.remove(empty_foam)


def plot_houses(title: str, points: np.ndarray, u: np.ndarray, p: np.ndarray, house_mesh_path: str, save_path=None):
    """
    Scatter of plot of houses surface points values.

    The filename of the saved plot is taken from the title.
    :param title: The title of the plot.
    :param points: The points to plot, shape (N,3).
    :param u: The velocity at points, shape (N,3).
    :param p: The pressure at points, shape (N,1).
    :param house_mesh_path: path to the house mesh .obj file.
    :param save_path: directory to save the plot into. Pass None to show the plot instead fo saving.
    """
    house = pv.get_reader(house_mesh_path).read()
    data = PolyData(points)
    data['Uinterp'] = u
    data['pinterp'] = p

    plotter = Plotter(shape=(1, 2), off_screen=save_path is not None, window_size=[3840, 1440])

    colorbar = {'title': title, 'vertical': True, 'position_y': 0.25, 'height': 0.5}

    plotter.subplot(0, 0)
    plotter.add_mesh(house, scalar_bar_args=colorbar, color='oldlace')
    plotter.camera.zoom(5)
    plot_scalar_field(f'U error ${M_S}$', points, np.linalg.norm(u, axis=1), None, plotter)

    plotter.subplot(0, 1)
    plotter.add_mesh(house, scalar_bar_args=colorbar, color='oldlace')
    plotter.camera.zoom(5)
    plot_scalar_field(f'p error ${M2_S2}$', points, p, None, plotter)

    plotter.show(screenshot=f'{save_path}/{title}.png' if save_path else False)


def plot_fields(title: str, points: np.array, u: np.array, p: np.array, porous: np.array or None, save_path=None):
    """
    Scatter plot of a 3D field.

    Porous points marking is currently disabled. The filename of the saved plot is taken from the title.
    :param title: Title of the plot.
    :param points: The points to plot, shape (N,3).
    :param u: The velocity at points, shape (N,3).
    :param p: The pressure at points, shape (N,1).
    :param porous: Array of porous points binary indicator. Reserved for future use.
    :param save_path: directory to save the plot into. Pass None to show the plot instead of saving.
    """
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


def plot_case(path: str, save_path=None):
    """
    Scatter plot of a 3D OpenFOAM case.

    The filename of the saved plot is taken from the title.
    :param path: Path to the OpenFOAM case.
    :param save_path: directory to save the plot into. Pass None to show the plot instead of saving.
    """
    fields = data_parser.parse_case_fields(path, 'C', 'U', 'p', 'cellToRegion')
    plot_fields(Path(path).stem,
                fields['C'].to_numpy(),
                fields['U'].to_numpy(),
                fields['p'].to_numpy(),
                fields['cellToRegion'],
                save_path=save_path)
