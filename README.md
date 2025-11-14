# Modeling diffusion in complex media through Physically Informed Neural Networks
Official implementation of the paper *Geometry-Aware Physics-Informed PointNets
for Modeling Flows Across Porous Structures* accepted at the MEDES 25 conference. 

It is possible to simulate mixed fluid-porous fluid domains using PIPN (Physics Informed Pointnet), PIPN++, PI-GANO (Physics Informed Geometry Aware Neural Operator) and PI-GANO++ architectures.

The models are trained with losses enforcing the Navier-Stokes-Darcy equations coupled with the penalization method.

A selection of 2D and 3D experiments can be run both locally and on SLURM clusters with Singularity.


 ## Quick start
- #### Install [OpenFOAM 2412](https://develop.openfoam.com/Development/openfoam/-/wikis/precompiled/)
- #### Create a conda environment and activate it

    ```conda create -n porous-cfd python=3.11```

    ```conda activate porous-cfd```
- #### Install [PyTorch]() and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/2.4.0/install/installation.html)
- #### Install the dependencies:

    ```pip install -r requirements```
- #### Choose an example from the *examples* folder (for instance ABC)

    ```cd examples/abc```
- #### Generate the data
    Pass the OpenFOAM binary directory (usually */usr/lib/openfoam/openfoam2412*) and the number of processors

    ```python generate_data.py --openfoam-dir "/usr/lib/openfoam/openfoam2412" --openfoam-procs 8```
- #### Train the model 
    Choose the number of internal, boundary and observation points

    ```python train.py --model pipn-pp --name pipn-pp-test --n-internal 1500 --n-boundary 1000 --n-observations 700```

    The model weights will be save in *abc/lightning_logs/pipn-pp-test*
- #### Show predictions

    ```python inference.py --n-internal 1500 --n-boundary 1000 --n-observations 700 --data-dir data/val```
    
    To save the plots as images pass  the ```--save-plots``` argument. The results will be saved in *abc/lightning_logs/pipn-pp-test/plots/val*.

- #### Evaluate the model

    ```python evaluate.py --n-internal 1500 --n-boundary 1000 --n-observations 700```

    To save the plots as images pass  the ```--save-plots``` argument. The results will be saved in *abc/lightning_logs/pipn-pp-test/plots/stats*.

## Run on SLURM HPC with Singularity
- #### Build the container

    ```singularity build --fakeroot container.sif  singularity/container.def```
- #### Run the sbatch script on the cluster

    ```sbatch --job-name=abc-test sbatch.sh -c container.sif -x abc -g -i 1500 -b 1000 -o 700 -m pipn-pp -n pipn-pp-test```

    The model weights and plots will be saved in ```examples/abc/lightning_logs/pipn-pp-test```.

## Use your own data
The project was designed to allow customization of both the data and pipeline. The data generation can be customized to include porous and solid geometries and custom OpenFOAM simulation templates.

### Directory setup
Create a new directory inside the *examples* folder. This will contain the experiment. Add your own case template or copy one of the OpenFOAM case template from the other examples and customize its settings. Depending on the base experiment some parameters will be overridden when generating the data such as the ones found in *decomposeParDict*, *snappyHexMeshDict* and *fvOptions*. Those can be customized with the data generator class. The SLURM uses the ```run()``` functions inside each script.

### Create your own meshes
Save the base objects as *.obj* and place them inside the *examples/your_example/assets/meshes/split_name* directory. It is recommended to create a *config.json* which includes the data splitting and optional variable boundary parameters:

```json
{
  "cfd params": {
    "coeffs": [
      {"d": [5000, 5000, 0], "f": [16.381, 16.381, 0]},
      {"d": [7000, 7000, 0], "f": [20.783, 20.783, 0]},
    ],
    "inlet": [0.1, 0.125],
    "angle": [-30, 30]
  },
  "splits": {
    "train": 0.6,
    "val": 0.2,
    "test": 0.2
  }
}
```

The parameters can be customized and will be available when generating the data.
Define a *transforms.json* file containing the data augmentation transforms. It is also possible to use the configurations from other examples as a base.
Define a *data_config.json* which defines the fields, normalization and variable boundary conditions. For example:

```json
{
  "Fields": ["C", "U", "p", "cellToRegion", "d", "f"],
  "Variable boundaries": {"U": "inlet"},
  "Normalize fields": {
    "Scale": ["d", "f"], "Standardize": ["C", "U", "p"]
  },
  "Dims": ["x", "y"]
}
```

The ```Scale``` fields will be normalized to the [0,1] range, the ```Standardize``` will be scaled using the Z-Score. Use empty lists to disable normalization and variable boundary conditions.

### Data Generator
If further customization is needed, define a new data generator by extending one of the base classes ```DataGeneratorBase```, ```Generator2DBase``` or ```Generator3DBase``` and override the abstract functions:

```python
from datagen.generator_2d import Generator2DBase
from pathlib import Path
import shutil
import glob


class CustomGenerator2D(Generator2DBase):
    def generate_transformed_meshes(self, meshes_dir: Path, dest_dir: Path, rng):
        # Copy all meshes to destination folder
        # It is also possible to load the transforms.json and config.json for further processing
        for m in glob.glob(f'{meshes_dir}/*.obj'):
            mesh_name = Path(m).name
            shutil.copyfile(m, f'{dest_dir}/{mesh_name}')

    def generate_openfoam_cases(self, meshes_dir, dest_dir, case_config_dir, rng):
        for m in glob.glob(f"{meshes_dir}/*.obj"):
            # Extract the name of the case from the mesh path
            case_path = f"{dest_dir}/{Path(m).stem}"

            # Copy the 2D template case
            shutil.copytree(self.case_template_dir, case_path)

            # Copy the mesh. The 2D templates use mesh extrusion
            shutil.copyfile(m, f"{case_path}/snappyHexMesh/constant/triSurface/mesh.obj")

            # Set snappyHexMesh parameters
            self.write_locations_in_mesh(f'{case_path}/snappyHexMesh', self.get_location_inside(m))

            # Set multi processing parameters
            self.set_decompose_par(f'{case_path}/snappyHexMesh')
            self.set_decompose_par(f'{case_path}/simpleFoam')
```

Add a ```generate_data.py``` script:

```python
from datagen.data_generator import build_arg_parser
from examples.your_example.custom_generator import CustomGenerator2D


def run():
    args = build_arg_parser().parse_args()
    openfoam_cmd = f'{args.openfoam_dir}/etc/openfoam'
    generator = CustomGenerator2D('assets', openfoam_cmd, args.openfoam_procs, 0.5, args.meta_only)
    generator.generate(args.data_root_dir)


if __name__ == '__main__':
    run()
```

### Custom dataset
If custom features or sampling are needed, extend the ```FoamDataset``` class and override the required functions:

```python
from pandas import DataFrame
from dataset.foam_dataset import FoamDataset
import numpy as np


class CustomDataset(FoamDataset):
    def add_features(self, internal_fields: DataFrame, boundary_fields: DataFrame):
        # Add SDF and boundary id
        super().add_features(internal_fields, boundary_fields)

        # Add an empty feature. The DataFrame uses multi indexing
        internal_fields['empty', ''] = np.zeros(len(internal_fields))
        boundary_fields['empty', ''] = np.zeros(len(boundary_fields))

    def sample_internal(self, internal_fields: DataFrame) -> DataFrame:
        # Sample random points
        return internal_fields.sample(self.n_internal, random_state=self.rng)

```

### Training
To configure the layers and data loader, create a ```train.py``` script inside the base experiment directory:

```python
from numpy.random import default_rng
from common.training import train, build_arg_parser
from examples.your_example.custom_dataset import CustomDataset
from models.pipn.pipn_foam import PipnFoam


def run():
    # Parse the arguments
    args = build_arg_parser().parse_args()

    n_internal = args.n_internal
    n_boundary = args.n_boundary
    n_obs = args.n_observations

    # Load the training and validtion data
    rng = default_rng(8421)
    train_data = CustomDataset(args.train_dir, n_internal, n_boundary, n_obs, rng=rng)
    val_data = CustomDataset(args.val_dir, n_internal, n_boundary, n_obs, rng=rng, meta_dir=args.train_dir)

    # Define the model
    model = PipnFoam(nu=0.01, d=50, f=1, fe_local_layers=[2, 64], fe_global_layers=[69, 1024],
                             seg_layers=[1024 + 64, 3],scalers=train_data.normalizers)

    # Train the model
    train(args, model, train_data, val_data)


if __name__ == '__main__':
    run()
```

### Inference
Create an ```inference.py``` script and create the ```process_sample_fn``` function. It will be called on each sample of the inference data:

```python
import numpy as np
from common.inference import build_arg_parser, predict
from examples.your_example.custom_dataset import CustomDataset
from models.pipn.pipn_foam import PipnFoam
from visualization.visualization_2d import plot_fields


def sample_process_fn(data, target, predicted, case_path, plot_path):
    # Plot the prediction
    plot_fields(f'Predicted', target['C'],
                predicted['U'],
                predicted['p'],
                target['cellToRegion'].numpy(),
                save_path=plot_path)


def run():
    args = build_arg_parser().parse_args()

    rng = np.random.default_rng(8421)
    model = PipnFoam.load_from_checkpoint(args.checkpoint)
    val_data = CustomDataset(args.data_dir, args.n_internal, args.n_boundary, args.n_observations, rng, args.meta_dir)

    predict(args, model, val_data, sample_process_fn)


if __name__ == '__main__':
    run()

```

### Evaluation
Similar to inference, create a ```evaluation.py``` script. By default the evaluation pipeline already saves a base set of plots (MAE, Maximum errors, timing, etc...). However it is possible to add specific plots for each experiment:

```python
from numpy.random import default_rng
from common.evaluation import build_arg_parser, evaluate
from examples.your_example.custom_dataset import CustomDataset
from models.pipn.pipn_foam import PipnFoam


def sample_process(data, predicted, target, extras):
    # Return the previously added empty feature
    return {'empty': target['empty']}


def postprocess_fn(data, results, plots_path):
    # Here it is possible to create custom plots with the data collected from the sample_process_fn.
    # Tensor objects will be converted to numpy arrays automatically to ease the plotting.
    empty = results['empty']


def run():
    args = build_arg_parser().parse_args()

    model = PipnFoam.load_from_checkpoint(args.checkpoint)

    rng = default_rng(8421)
    data = CustomDataset(args.data_dir, args.n_internal, args.n_boundary, args.n_observations, rng, args.meta_dir)

    evaluate(args, model, data, True, None, None)


if __name__ == '__main__':
    run()

```


## Customize the models
A base model class was created which allows to abstract the management of the subdomains (internal, boundary, etc..) and the computation of losses. To create a new model extend the ```PorousPinnBase``` class. The base class uses *PyTorch Lightning*. For example a simple linear layer would be:

```python
import torch.nn as nn
import torch
from dataset.foam_data import FoamData
from models.model_base import PorousPinnBase
from models.losses import ContinuityLoss, MomentumLossManufactured
from torch import Tensor


class MlpPinn(PorousPinnBase):
    def __init__(self, nu, d, f, in_features, out_features):
        super().__init__(out_features, enable_data_loss=False, loss_scaler=None)
        self.save_hyperparameters()

        # Define the layers
        self.mlp = nn.Linear(in_features, out_features)

        # Define the physics losses
        self.momentum_loss = MomentumLossManufactured(nu, d, f)
        self.continuity_loss = ContinuityLoss()

    def forward(self, autograd_points: Tensor, x: FoamData) -> FoamData:
        # Here only the points coordinates are used. Other features are available from x
        y = self.mlp.forward(autograd_points)

        return FoamData(y, self.predicted_labels, x.domain)

    # Define the optimizer
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001, eps=1e-6)
```

It is also possible to customize the losses and loss scalers. Examples can be found in *models/losses.py*.

 The ```FoamData``` object is a wrapper of a ```Tensor``` which allows indexing by feature and by subdomain:

```python
u = x['U']
u_x = x['Ux']
inlet_u = x['U-inlet']

internal = x['internal']
boundary = x['boundary']
interface = x['interface']

u_boundary = boundary['U']
boundary_inlet_ux = boundary['U-inletx']
```

For composite fields the dimension name is appended to the name, for variable boundary conditions the name of the boundary condition is appended to the field. The ```autograd_points``` arguments are the points used to calculate the loss derivatives. They must be processed by the model.