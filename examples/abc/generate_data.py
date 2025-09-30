from pathlib import Path
from datagen.data_generator import build_arg_parser
from examples.abc.abc_generator import AbcGenerator
from examples.abc.data_preprocess import move_to_meshes, extract

if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    data_root = Path(args.data_root_dir)
    data_root.mkdir(exist_ok=True, parents=True)
    # download('https://archive.nyu.edu/retrieve/89085/abc_0000_obj_v00.7z', data_root / 'Abc.7z')
    extract('assets/abc.tar.gz', 'assets/meshes/standard')

    openfoam_cmd = f'{args.openfoam_dir}/etc/openfoam'
    generator = AbcGenerator('assets', openfoam_cmd, args.openfoam_procs, 0.5, args.meta_only)
    generator.generate(args.data_root_dir)
