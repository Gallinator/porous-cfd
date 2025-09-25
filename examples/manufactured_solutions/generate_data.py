import os

from datagen.data_generator import build_arg_parser
from examples.manufactured_solutions.manufactured_generator import GeneratorManufactured

if __name__ == '__main__':
    print(os.environ)
    print(os.getcwd())
    args = build_arg_parser().parse_args()
    openfoam_cmd = f'{args.openfoam_dir}/etc/openfoam'
    generator = GeneratorManufactured('assets', openfoam_cmd, args.openfoam_procs, meta_only=args.meta_only)
    generator.generate(args.data_root_dir)
