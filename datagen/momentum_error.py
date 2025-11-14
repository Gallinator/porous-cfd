import os
import re

import numpy as np
import torch
from foamlib import FoamFile, FoamCase, FoamFieldFile
from pandas import DataFrame
from torch import Tensor

from dataset.data_parser import parse_case_fields


def momentum_error(nu: float, d: Tensor, f: Tensor, u: Tensor, u_jac: Tensor, u_laplace: Tensor, p_grad: Tensor,
                   zone_id: Tensor) -> Tensor:
    """
    Calculates the momentum error. All tensors have shape (n_points,d)
    :param nu: kinematic viscosity.
    :param d: Darcy coefficients.
    :param f: Forchheimer coefficients.
    :param u: Velocity.
    :param u_jac: Velocity Jacobian.
    :param u_laplace: Velocity laplace operator.
    :param p_grad: Pressure gradients.
    :param zone_id: Zero if fluid, one if porous.
    :return: The momentum error.
    """
    source = u * (d * nu + 1 / 2 * torch.norm(u, dim=-1, keepdim=True) * f)
    return (torch.matmul(u_jac, u.unsqueeze(-1)).squeeze() -
            nu * torch.matmul(u_laplace, torch.ones_like(u).unsqueeze(-1)).squeeze() +
            p_grad +
            source * zone_id)


def write_momentum_error(case_path: str):
    """
    Write the momentum error field.
    This is necessary as the openfoam momentum calculation seems to not take into account the porous material.
    """
    # Set labels
    jac_labels = ['grad(U)xx', 'grad(U)xy', 'grad(U)xz',
                  'grad(U)yx', 'grad(U)yy', 'grad(U)yz',
                  'grad(U)zx', 'grad(U)zy', 'grad(U)zz']
    lap_labels = ['grad(grad(U)xx)', 'grad(grad(U)xy)', 'grad(grad(U)xz)',
                  'grad(grad(U)yx)', 'grad(grad(U)yy)', 'grad(grad(U)yz)',
                  'grad(grad(U)zx)', 'grad(grad(U)zy)', 'grad(grad(U)zz)']

    data = parse_case_fields(case_path, 'U', 'grad(p)', *jac_labels, *lap_labels, 'd', 'f', 'cellToRegion', max_dim=3)
    # Vector values
    grad_p = torch.tensor(data['grad(p)'].values)
    u = torch.tensor(data['U'].values)
    d = torch.tensor(data['d'].values)
    f = torch.tensor(data['f'].values)
    id = torch.tensor(data['cellToRegion'].values).unsqueeze(-1)

    # Extract jacobian and laplacian
    jacobian = data[jac_labels].values.reshape(-1, 3, 3)
    jacobian = torch.tensor(jacobian)
    laplacian = data[lap_labels].values.reshape(-1, 3, 3, 3)
    # Use diagonal as only ii values are required
    laplacian = np.diagonal(laplacian, axis1=-2, axis2=-1)
    laplacian = torch.tensor(laplacian)

    nu = FoamFile(f'{case_path}/constant/transportProperties')['nu'].value

    error = momentum_error(nu, d, f, u, jacobian, laplacian, grad_p, id)
    error_df = DataFrame(error, index=data.index, columns=['x', 'y', 'z'])

    case = FoamCase(case_path)
    last_time = int(case[-1].time)

    # Set internal field
    internal_moment_field = FoamFieldFile(f'{case_path}/{last_time}/momentError')
    internal_moment_field.internal_field = error_df.loc['internal'].values

    # Set postprocess and boundary fields
    internal_moment_field['boundaryField'] = {}
    for b in error_df.index.unique():
        if b != 'internal':
            field_values = error_df.loc[b].values
            internal_moment_field.boundary_field[b] = {'type': 'extrapolatedCalculated',
                                                       'values': field_values}
            postprocess_path = f"{case_path}/postProcessing"
            last_step_dir = f'{postprocess_path}/{b}/surface/{last_time}'
            patch_dir = list(os.listdir(f"{last_step_dir}"))[0]
            momentum_file_path = f'{last_step_dir}/{patch_dir}/vectorField/momentError'

            with FoamFile(momentum_file_path) as f:
                f[None] = field_values

            # Reformat
            with open(momentum_file_path, 'r') as f:
                data = f.read()
            with open(momentum_file_path, 'w') as f:
                data = re.sub('FoamFile.+{.+}\n*', '', data, flags=re.DOTALL)
                data = data.replace(') (', ')\n(')
                data = data.replace('((', '(\n(')
                data = data.replace('))', ')\n)')
                f.write(data)

    # Add empty patches if 2D
    empty_patches = [b for b, v in case[0]['U'].boundary_field.items() if v['type'] == 'empty']
    for p in empty_patches:
        internal_moment_field.boundary_field[p] = {'type': 'empty'}
