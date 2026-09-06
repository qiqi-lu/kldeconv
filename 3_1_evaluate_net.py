"""
Evaluate the trained conventional network.
"""

import torch, os, pandas, tqdm, json, time
import numpy as np
from skimage import io
from models.dfcan_2d import DFCAN
from models.dfcan_3d import DFCAN3D
from models.rln_3d import RLN3D
from utils.data import win2linux, SRDataset, NormalizePercentile, read_txt
from checkpoint_list import checkpoints_v1 as checkpoints_list
from utils.optimize import on_load_checkpoint

# ------------------------------------------------------------------------------
#                             Paramsters
# ------------------------------------------------------------------------------
id_device = "cuda:0"
# id_device = "cpu"
# ------------------------------------------------------------------------------
# model_name = "dfcan"
model_name = "rln"

dataset_list = (
    # --------------------------------------------------------------------------
    # ("SimuMix3D-128-31-0-0-1", "SimuMix3D-128-31-0-0-1"),
    # ("SimuMix3D-128-31-05-1-01", "SimuMix3D-128-31-05-1-01"),
    ("SimuMix3D-512-31-05-1-01", "SimuMix3D-128-31-05-1-01"),
    # ("SimuMix3D-1024-31-05-1-01", "SimuMix3D-128-31-05-1-01"),
    # --------------------------------------------------------------------------
    # ("Microtubule2-3d-1024", "Microtubule2-3d-1024"),
    # ("Microtubule2-3d-1024", "Nuclear-pore-complex2-1024"),
    # ("Nuclear-pore-complex2-1024", "Nuclear-pore-complex2-1024"),
    # ("Nuclear-pore-complex2-1024", "Microtubule2-3d-1024"),
    # ("biotisr-3d-mt-1", "biotisr-3d-mt-1"),
    # ("biotisr-3d-mt-2", "biotisr-3d-mt-2"),
    # ("biotisr-3d-mito-1", "biotisr-3d-mito-1"),
    # ("biotisr-3d-mito-2", "biotisr-3d-mito-2"),
    # ("biotisr-3d-factin-1", "biotisr-3d-factin-1"),
    # ("biotisr-3d-factin-2", "biotisr-3d-factin-2"),
    # --------------------------------------------------------------------------
    # ("F-actin-nonlinear-9", "F-actin-nonlinear-9"),
    # ("F-actin-nonlinear-9", "Microtubules2-9"),
    # ("F-actin-nonlinear-9", "CCPs-9"),
    # ("F-actin-nonlinear-9", "ER-6"),
    # ("F-actin-nonlinear-9", "F-actin-9"),
    # ("F-actin-nonlinear-1", "F-actin-nonlinear-1"),
    # ("F-actin-nonlinear-1", "Microtubules2-1"),
    # ("F-actin-nonlinear-1", "CCPs-1"),
    # ("F-actin-nonlinear-1", "ER-1"),
    # ("F-actin-nonlinear-1", "F-actin-1"),
    # ("Microtubules2-9", "Microtubules2-9"),
    # ("Microtubules2-9", "F-actin-nonlinear-9"),
    # ("Microtubules2-9", "CCPs-9"),
    # ("Microtubules2-9", "ER-6"),
    # ("Microtubules2-9", "F-actin-9"),
    # ("Microtubules2-8", "Microtubules2-9"),
    # ("Microtubules2-7", "Microtubules2-9"),
    # ("Microtubules2-6", "Microtubules2-9"),
    # ("Microtubules2-5", "Microtubules2-9"),
    # ("Microtubules2-4", "Microtubules2-9"),
    # ("Microtubules2-3", "Microtubules2-9"),
    # ("Microtubules2-2", "Microtubules2-9"),
    # ("Microtubules2-1", "Microtubules2-9"),
    # ("Microtubules2-1", "Microtubules2-1"),
    # ("Microtubules2-1", "F-actin-nonlinear-1"),
    # ("Microtubules2-1", "CCPs-1"),
    # ("Microtubules2-1", "ER-1"),
    # ("Microtubules2-1", "F-actin-1"),
    # ("CCPs-9", "Microtubules2-9"),
    # ("CCPs-9", "F-actin-nonlinear-9"),
    # ("CCPs-9", "CCPs-9"),
    # ("CCPs-9", "ER-6"),
    # ("CCPs-9", "F-actin-9"),
    # ("CCPs-1", "CCPs-1"),
    # ("CCPs-1", "F-actin-nonlinear-1"),
    # ("CCPs-1", "Microtubules2-1"),
    # ("CCPs-1", "ER-1"),
    # ("CCPs-1", "F-actin-1"),
    # ("F-actin-9", "Microtubules2-9"),
    # ("F-actin-9", "F-actin-nonlinear-9"),
    # ("F-actin-9", "CCPs-9"),
    # ("F-actin-9", "ER-6"),
    # ("F-actin-9", "F-actin-9"),
    # ("F-actin-1", "F-actin-1"),
    # ("F-actin-1", "F-actin-nonlinear-1"),
    # ("F-actin-1", "Microtubules2-1"),
    # ("F-actin-1", "CCPs-1"),
    # ("F-actin-1", "ER-1"),
    # ("ER-6", "Microtubules2-9"),
    # ("ER-6", "F-actin-nonlinear-9"),
    # ("ER-6", "CCPs-9"),
    # ("ER-6", "ER-6"),
    # ("ER-6", "F-actin-9"),
    # ("ER-1", "ER-1"),
    # ("ER-1", "F-actin-nonlinear-1"),
    # ("ER-1", "Microtubules2-1"),
    # ("ER-1", "CCPs-1"),
    # ("ER-1", "F-actin-1"),
    # --------------------------------------------------------------------------
    # ("F-actin-nonlinear-9", "F-actin-nonlinear-9"),
    # ("F-actin-nonlinear-8", "F-actin-nonlinear-8"),
    # ("F-actin-nonlinear-7", "F-actin-nonlinear-7"),
    # ("F-actin-nonlinear-6", "F-actin-nonlinear-6"),
    # ("F-actin-nonlinear-5", "F-actin-nonlinear-5"),
    # ("F-actin-nonlinear-4", "F-actin-nonlinear-4"),
    # ("F-actin-nonlinear-3", "F-actin-nonlinear-3"),
    # ("F-actin-nonlinear-2", "F-actin-nonlinear-2"),
    # ("F-actin-nonlinear-1", "F-actin-nonlinear-1"),
    # ("Microtubules2-9", "Microtubules2-9"),
    # ("Microtubules2-8", "Microtubules2-8"),
    # ("Microtubules2-7", "Microtubules2-7"),
    # ("Microtubules2-6", "Microtubules2-6"),
    # ("Microtubules2-5", "Microtubules2-5"),
    # ("Microtubules2-4", "Microtubules2-4"),
    # ("Microtubules2-3", "Microtubules2-3"),
    # ("Microtubules2-2", "Microtubules2-2"),
    # ("Microtubules2-1", "Microtubules2-1"),
    # ("CCPs-9", "CCPs-9"),
    # ("CCPs-8", "CCPs-8"),
    # ("CCPs-7", "CCPs-7"),
    # ("CCPs-6", "CCPs-6"),
    # ("CCPs-5", "CCPs-5"),
    # ("CCPs-4", "CCPs-4"),
    # ("CCPs-3", "CCPs-3"),
    # ("CCPs-2", "CCPs-2"),
    # ("CCPs-1", "CCPs-1"),
    # ("F-actin-9", "F-actin-9"),
    # ("F-actin-8", "F-actin-8"),
    # ("F-actin-7", "F-actin-7"),
    # ("F-actin-6", "F-actin-6"),
    # ("F-actin-5", "F-actin-5"),
    # ("F-actin-4", "F-actin-4"),
    # ("F-actin-3", "F-actin-3"),
    # ("F-actin-2", "F-actin-2"),
    # ("F-actin-1", "F-actin-1"),
    # ("ER-6", "ER-6"),
    # ("ER-5", "ER-5"),
    # ("ER-4", "ER-4"),
    # ("ER-3", "ER-3"),
    # ("ER-2", "ER-2"),
    # ("ER-1", "ER-1"),
    # --------------------------------------------------------------------------
    # ("biotisr-ccps-1", "biotisr-ccps-1"),
    # ("biotisr-ccps-2", "biotisr-ccps-2"),
    # ("biotisr-ccps-3", "biotisr-ccps-3"),
    # ("biotisr-factin-1", "biotisr-factin-1"),
    # ("biotisr-factin-2", "biotisr-factin-2"),
    # ("biotisr-factin-3", "biotisr-factin-3"),
    # ("biotisr-factin-nonlinear-1", "biotisr-factin-nonlinear-1"),
    # ("biotisr-factin-nonlinear-2", "biotisr-factin-nonlinear-2"),
    # ("biotisr-factin-nonlinear-3", "biotisr-factin-nonlinear-3"),
    # ("biotisr-lysosomes-1", "biotisr-lysosomes-1"),
    # ("biotisr-lysosomes-2", "biotisr-lysosomes-2"),
    # ("biotisr-lysosomes-3", "biotisr-lysosomes-3"),
    # ("biotisr-mito-1", "biotisr-mito-1"),
    # ("biotisr-mito-2", "biotisr-mito-2"),
    # ("biotisr-mito-3", "biotisr-mito-3"),
    # ("biotisr-mt-1", "biotisr-mt-1"),
    # ("biotisr-mt-2", "biotisr-mt-2"),
    # ("biotisr-mt-3", "biotisr-mt-3"),
    # ("deepbacs-ecoli-ave2", "deepbacs-ecoli-ave2"),
    # ("deepbacs-saureus-ave2", "deepbacs-saureus-ave2"),
    # ("w2s-0-sim-ave", "w2s-0-sim-ave"),
    # ("w2s-1-sim-ave", "w2s-1-sim-ave"),
    # ("w2s-2-sim-ave", "w2s-2-sim-ave"),
)


device = torch.device(id_device)
print(f"[INFO] Device: {device}")
# num_data, id_repeat = 80, 1
num_data, id_repeat = 1, 1
print(f"[INFO] Num data: {num_data}")
print(f"[INFO] Id repeat: {id_repeat}")
info_xlsx = pandas.read_excel("datasets_test.xlsx")

for dataset_names in dataset_list:
    dataset_name_test, dataset_name_train = dataset_names

    # id_sample = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    # id_sample = [0, 1, 2, 3, 4, 5, 6]
    # id_sample = [0]
    id_sample = []

    print("-" * 80)
    print(f"[INFO] Dataset test: {dataset_name_test}")
    print(f"[INFO] Dataset train: {dataset_name_train}")
    print("-" * 80)

    # --------------------------------------------------------------------------
    path_prediction = os.path.join(
        "outputs",
        "predictions",
        dataset_name_test,
        model_name,
        dataset_name_train,
        f"n{num_data}_r{id_repeat}",
    )
    os.makedirs(path_prediction, exist_ok=True)

    info = info_xlsx[info_xlsx["id"] == dataset_name_test].iloc[0]

    params = dict(
        dataset_name_test=dataset_name_test,
        dataset_name_train=dataset_name_train,
        ndim=int(info["ndim"]),
        ratio=float(info["ratio"]),
        path_lr=win2linux(info["path_lr"]),
        path_hr=win2linux(info["path_hr"]),
        path_lr_txt=win2linux(info["path_txt"]),
        path_hr_txt=win2linux(info["path_txt"]),
        in_channels=1,
        normalization=(False, False),
        id_sample=id_sample,
        num_data=num_data,
        id_repeat=id_repeat,
        id_device=id_device,
        norm_pred=(0.03, 0.995),
    )

    # --------------------------------------------------------------------------
    filenames = read_txt(params["path_lr_txt"])
    num_samples = len(filenames)
    if params["id_sample"] == []:
        print("[INFO] Predict all samples.")
        params["id_sample"] = list(range(num_samples))

    print("-" * 80)
    print(f"[INFO] Dataset: {dataset_name_test}")
    print(f"[INFO] Training dataset: {dataset_name_train}")
    print(f"[INFO] Id sample test: {params['id_sample']}")
    print("-" * 80)
    for key, value in params.items():
        print(f"[INFO] {key}: {value}")
    print("-" * 80)

    # --------------------------------------------------------------------------
    #                             Load data
    # --------------------------------------------------------------------------
    dataset_test = SRDataset(
        lr_root_path=params["path_lr"],
        hr_root_path=params["path_hr"],
        lr_txt_file_path=params["path_lr_txt"],
        hr_txt_file_path=params["path_hr_txt"],
        normalization=params["normalization"],
        id_range=None,
        transform=None,
    )

    # --------------------------------------------------------------------------
    #                             Load model
    # --------------------------------------------------------------------------
    if params["ndim"] == 2:
        model = DFCAN(
            in_channels=params["in_channels"],
            scale_factor=1,
            num_features=64,
            num_groups=4,
        )
    elif params["ndim"] == 3:
        if model_name == "rln":
            model = RLN3D(
                scale=1, in_channels=params["in_channels"], n_features=4, kernel_size=3
            )
        elif model_name == "dfcan":
            model = DFCAN3D(
                in_channels=params["in_channels"],
                scale_factor=1,
                num_features=64,
                num_groups=4,
            )
        else:
            raise ValueError(
                f"[ERROR] Model is not supported. {model_name} ({params['ndim']}D)"
            )
    else:
        raise ValueError(f"[ERROR] Dimension is not supported. {params['dim']}")

    model = model.to(device)

    # load trained parameters --------------------------------------------------
    path_checkpoint = win2linux(
        checkpoints_list[dataset_name_train][model_name][f"n{num_data}_r{id_repeat}"]
    )
    assert os.path.exists(
        path_checkpoint
    ), f"[ERROR] Checkpoint not found. {path_checkpoint}"
    print(f"[INFO] Load checkpoint from {path_checkpoint}")
    params["path_checkpoint"] = path_checkpoint

    state_dict = torch.load(path_checkpoint, map_location=device, weights_only=True)[
        "model_state_dict"
    ]
    state_dict = on_load_checkpoint(state_dict, complie_mode=False)
    model.load_state_dict(state_dict)
    model.eval()

    path_params_json = os.path.join(path_prediction, "params.json")
    with open(path_params_json, "w") as f:
        json.dump(params, f, indent=4)
    print(f"[INFO] Parameters are saved to {path_params_json}")

    # --------------------------------------------------------------------------
    #                                 Evaluate
    # --------------------------------------------------------------------------
    print("-" * 80)
    print("[INFO] Start evaluating...")
    assert (
        params["id_sample"] is not None and len(params["id_sample"]) > 0
    ), "[ERROR] id_sample is None"

    normalizer = NormalizePercentile(
        p_low=params["norm_pred"][0],
        p_high=params["norm_pred"][1],
        ndim=params["ndim"],
    )

    # --------------------------------------------------------------------------
    time_list = []
    pbar = tqdm.tqdm(total=len(params["id_sample"]), desc="Evaluating", ncols=80)
    for id in params["id_sample"]:
        data = dataset_test[id]
        x = torch.clamp(data["lr"], min=0.0, max=None)
        x = normalizer(x).to(device)[None]

        with torch.no_grad():
            if "cuda" in id_device:
                torch.cuda.synchronize(device=device)
            tic = time.time()
            y_pred = model(x)
            if "cuda" in id_device:
                torch.cuda.synchronize(device=device)
            toc = time.time()
            used_time = toc - tic
            time_list.append(used_time)
            print(f"[INFO] Sample {id}: {used_time:.4f}s")

        # save results ---------------------------------------------------------
        path_sample = os.path.join(path_prediction, filenames[id].split(".")[0])
        os.makedirs(path_sample, exist_ok=True)

        x = x.cpu().detach().numpy()[0, 0]
        y_pred = y_pred.cpu().detach().numpy()[0, 0]

        io.imsave(
            os.path.join(path_sample, "x.tif"),
            x.astype(np.float32),
            check_contrast=False,
        )
        io.imsave(
            os.path.join(path_sample, "y_pred.tif"),
            y_pred.astype(np.float32),
            check_contrast=False,
        )
        pbar.update(1)
    pbar.close()
    print("-" * 80)

    # save the itme used for prediction into excel -----------------------------
    df = pandas.DataFrame(columns=["time (s)"])
    df["time (s)"] = time_list
    df.to_excel(
        os.path.join(path_prediction, f"time_{id_device.replace(':', '_')}.xlsx"),
        index=False,
    )
