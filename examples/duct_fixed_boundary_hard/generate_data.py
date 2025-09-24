import glob
from pathlib import Path

from datagen.data_generator import build_arg_parser
from examples.duct_fixed_boundary_hard.generator_2d_fixed import Generator2DFixedHard
from visualization.common import plot_dataset_dist, plot_u_direction_change
from visualization.visualization_2d import plot_case

if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    openfoam_cmd = f'{args.openfoam_dir}/etc/openfoam'
    generator = Generator2DFixedHard('assets', openfoam_cmd, args.openfoam_procs, 0.5, args.meta_only)
    generator.generate(args.data_root_dir)

    splits = list(glob.glob(f'{args.data_root_dir}/*/'))
    cases = glob.glob(f'{splits[0]}/*/')
    for c in cases:
        plot_case(c)
        break

    plots_path = Path(args.data_root_dir) / 'plots'
    for s in splits:
        split_plot_path = plots_path / Path(s).name
        split_plot_path.mkdir(exist_ok=True, parents=True)
        plot_dataset_dist(s, split_plot_path)
        plot_u_direction_change(s, split_plot_path)
