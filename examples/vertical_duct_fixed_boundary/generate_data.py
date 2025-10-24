import glob
from pathlib import Path

from datagen.data_generator import build_arg_parser
from examples.duct_fixed_boundary_hard.generator_2d_fixed import Generator2DFixedHard
from examples.vertical_duct_fixed_boundary.generator_2d_fixed import Generator2DFixedHardTop
from visualization.common import plot_dataset_dist, plot_u_direction_change
from visualization.visualization_2d import plot_case


def generate():
    args = build_arg_parser().parse_args()
    openfoam_cmd = f'{args.openfoam_dir}/etc/openfoam'
    generator = Generator2DFixedHardTop('assets', openfoam_cmd, args.openfoam_procs, 0.5, args.meta_only)
    generator.generate(args.data_root_dir,0)


if __name__ == '__main__':
    generate()
