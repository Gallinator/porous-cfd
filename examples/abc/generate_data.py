from pathlib import Path
from datagen.data_generator import build_arg_parser
from examples.abc.abc_generator import AbcGenerator
from examples.abc.data_preprocess import extract, download_from_gdrive


def run():
    args = build_arg_parser().parse_args()
    data_root = Path(args.data_root_dir)
    data_root.mkdir(exist_ok=True, parents=True)
    if not args.meta_only:
        download_from_gdrive('1pOGB9vO_Jf3YeRemJSs0yC5nBBKZcRTH&confirm', 'assets/Abc.tar.gz')
        extract('assets/Abc.tar.gz', 'assets/meshes/standard')

    openfoam_cmd = f'{args.openfoam_dir}/etc/openfoam'
    generator = AbcGenerator('assets', openfoam_cmd, args.openfoam_procs, 0.5, args.meta_only)
    generator.generate(args.data_root_dir)


if __name__ == '__main__':
    run()
