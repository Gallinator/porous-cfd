import shutil
from multiprocessing import Process
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from lightning import Trainer
from scipy.interpolate import griddata
from starlette.staticfiles import StaticFiles
from torch.utils.data import DataLoader

from app.config import AppSettings
from dataset.foam_data import FoamData
from dataset.foam_dataset import FoamDataset, collate_fn
from models.pipn.pipn_foam import PipnFoam, PipnFoamPp, PipnFoamPpMrg
from app.api_models import Predict2dInput, Response2d


def get_model(model_type):
    match model_type:
        case 'pipn':
            return PipnFoam.load_from_checkpoint("assets/weights/pipn.ckpt")
        case 'pipn_pp':
            return PipnFoamPp.load_from_checkpoint("assets/weights/pipn-pp.ckpt")
        case 'pipn_pp_mrg':
            return PipnFoamPpMrg.load_from_checkpoint("assets/weights/pipn-pp-mrg.ckpt")
        case _:
            raise NotImplementedError


def get_interpolation_grid(points, grid_res) -> list[np.ndarray]:
    points_x, points_y = points[:, 0].flatten(), points[:, 1].flatten()
    xx = np.linspace(points_x.min(), points_x.max(), grid_res)
    yy = np.linspace(points_y.min(), points_y.max(), grid_res)
    return np.meshgrid(xx, yy)


def interpolate_on_grid(grid, points, *data) -> list:
    return [griddata(points, d, tuple(grid), method='cubic', fill_value=0).flatten() for d in data]


def ndarrays_to_list(data: dict[str:np.ndarray]):
    for k, v in data.items():
        data[k] = v.flatten().tolist()
    return data


def inverse_transform_output(dataset: FoamDataset, data: FoamData, *fields) -> list[np.ndarray]:
    return [dataset.normalizers[f].inverse_transform(data[f].numpy(force=True)) for f in fields]


def generate_f(input_data: Predict2dInput, session_root: str):
    # Only import blender in a subprocess, as the path is passed from the main process (see https://projects.blender.org/blender/blender/issues/98534)
    # This has to be done here otherwise the context is incorrect
    from examples.duct_fixed_boundary.generator_2d_fixed import Generator2DFixed
    from app.preprocessing import path_to_obj, create_session_folders

    create_session_folders("assets", session_root)

    path_to_obj(input_data.points["x"], input_data.points["y"], f"{session_root}/assets/meshes/split")

    datagen = Generator2DFixed(f"{session_root}/assets", openfoam_cmd, settings.n_procs, 0, False)
    datagen.write_momentum = False
    datagen.save_plots = False

    datagen.generate(f"{session_root}/data")


settings = AppSettings()
app = FastAPI()
openfoam_cmd = f'{settings.openfoam_dir}/etc/openfoam'


@app.post("/predict", summary="Predict flow from porous object", response_model=dict[str, Response2d])
async def predict(input_data: Predict2dInput):
    session_dir = f"sessions/{input_data.uuid}"

    try:
        # Generate mesh using a new process due to blender import issues
        predict_process = Process(target=generate_f, args=(input_data, session_dir))
        predict_process.start()
        predict_process.join()

        # Override the generated min_points.json
        shutil.copy("assets/min_points.json", f"{session_dir}/data")

        dataset = FoamDataset(f"{session_dir}/data/split", 1000, 200, 500, np.random.default_rng(8421), meta_dir="assets")

        model = get_model(input_data.model)
        model.verbose_predict = True

        torch.manual_seed(8421)
        data_loader = DataLoader(dataset,
                                 1,
                                 num_workers=settings.n_procs,
                                 persistent_workers=True,
                                 shuffle=False,
                                 pin_memory=True,
                                 collate_fn=collate_fn)

        trainer = Trainer(logger=False,
                          enable_checkpointing=False,
                          inference_mode=False)

        predicted, residuals = trainer.predict(model, dataloaders=data_loader)[0]

        shutil.rmtree(session_dir)

        c, tgt_u, tgt_p = inverse_transform_output(dataset, dataset[0], "C", "U", "p")
        points = {"x": c[..., 0],
                  "y": c[..., 1]}

        target = {"Ux": tgt_u[..., 0],
                  "Uy": tgt_u[..., 1],
                  "U": np.linalg.norm(tgt_u, axis=1),
                  "p": tgt_p}

        pred_u, pred_p = inverse_transform_output(dataset, predicted, "U", "p")

        pred = {"Ux": pred_u[0, ..., 0],
                "Uy": pred_u[0, ..., 1],
                "U": np.linalg.norm(pred_u[0], axis=1),
                "p": pred_p[0]}

        error_u, error_p = np.abs(pred_u - tgt_u), np.abs(pred_p - tgt_p)
        error = {"Ux": error_u[0, ..., 0],
                 "Uy": error_u[0, ..., 1],
                 "U": np.linalg.norm(error_u[0], axis=1),
                 "p": error_p[0]}

        residuals = {"Momentumx": residuals["Momentumx"].numpy(force=True)[0],
                     "Momentumy": residuals["Momentumy"].numpy(force=True)[0],
                     "Momentum": np.linalg.norm(residuals["Momentum"].numpy(force=True)[0], axis=1),
                     "div": residuals["div"].numpy(force=True)[0]}

        porous_ids = dataset[0]["cellToRegion"].flatten()

        grid = get_interpolation_grid(c, 50)
        grid_points = {"x": grid[0].flatten(), "y": grid[1].flatten()}

        grid_pred = interpolate_on_grid(grid, c, pred["Ux"], pred["Uy"], pred["U"], pred["p"])
        grid_pred = dict(zip(pred.keys(), grid_pred))

        grid_target = interpolate_on_grid(grid, c, target["Ux"], target["Uy"], target["U"], target["p"])
        grid_target = dict(zip(target.keys(), grid_target))

        grid_error = interpolate_on_grid(grid, c, error["Ux"], error["Uy"], error["U"], error["p"])
        grid_error = dict(zip(pred.keys(), grid_error))

        internal_c = dataset.normalizers["C"].inverse_transform(dataset[0]["internal"]["C"].numpy(force=True))

        grid_residuals = interpolate_on_grid(grid, internal_c, residuals["Momentumx"],
                                             residuals["Momentumy"],
                                             residuals["Momentum"],
                                             residuals["div"])
        grid_residuals = dict(zip(residuals.keys(), grid_residuals))

        raw_data = Response2d(points=ndarrays_to_list(points),
                              target=ndarrays_to_list(target),
                              porous_ids=porous_ids.tolist(),
                              predicted=ndarrays_to_list(pred),
                              error=ndarrays_to_list(error),
                              residuals=ndarrays_to_list(residuals))

        grid_data = Response2d(points=ndarrays_to_list(grid_points),
                               target=ndarrays_to_list(grid_target),
                               predicted=ndarrays_to_list(grid_pred),
                               error=ndarrays_to_list(grid_error),
                               residuals=ndarrays_to_list(grid_residuals))

        return {"raw_data": raw_data, "grid_data": grid_data}
    except:
        traceback.print_exc()
        raise HTTPException(status_code=500)


app.mount("/", StaticFiles(directory="static", html=True), name="static")
