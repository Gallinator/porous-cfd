from datagen.data_generator import build_arg_parser
from examples.windbreaks.windbreak_generator import WindbreakGenerator

if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    openfoam_cmd = f'{args.openfoam_dir}/etc/openfoam'
    generator = WindbreakGenerator('assets', openfoam_cmd, args.openfoam_procs, 0.5, args.meta_only)
    generator.generate(args.data_root_dir)
