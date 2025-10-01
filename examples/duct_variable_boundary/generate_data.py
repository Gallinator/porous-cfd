from datagen.data_generator import build_arg_parser
from examples.duct_variable_boundary.generator_2d_variable import Generator2DVariable


def run():
    args = build_arg_parser().parse_args()
    openfoam_cmd = f'{args.openfoam_dir}/etc/openfoam'
    generator = Generator2DVariable('assets', openfoam_cmd, args.openfoam_procs, 0.5, args.meta_only)
    generator.generate(args.data_root_dir)


if __name__ == '__main__':
    run()
