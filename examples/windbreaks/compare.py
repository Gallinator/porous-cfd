from numpy.random import default_rng

from common.compare import compare, build_arg_parser
from dataset.data_parser import parse_model_type
from dataset.foam_dataset import FoamDataset
from models.pi_gano.pi_gano import PiGano, PiGanoFull
from models.pi_gano.pi_gano_pp import PiGanoPp
from models.pipn.pipn_foam import PipnFoam, PipnFoamPp, PipnFoamPpMrg, PipnFoamPpFull





if __name__ == '__main__':
    args = build_arg_parser()
    args = args.parse_args()

    model1 = get_model(args.checkpoint)
    model2 = get_model(args.checkpoint_other)

    rng = default_rng(8421)
    data = FoamDataset(args.data_dir, args.n_internal, args.n_boundary, args.n_observations, rng, args.meta_dir,
                       extra_fields=['momentError', 'div(phi)'])

    compare(args, model1, model2, data)
