"""
Train network.
- (image,) to (image,)
"""

import numpy as np
import torch, os, tqdm, json, pandas, datetime
from torchinfo import summary
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from models.dfcan_2d import DFCAN
from models.dfcan_3d import DFCAN3D
from models.rln_3d import RLN3D

from utils.data import win2linux, SRDataset, NormalizePercentile
from utils.optimize import on_load_checkpoint, StepLR_iter
import utils.evaluation as utils_eva

today_date = datetime.date.today()
# ------------------------------------------------------------------------------
# parameters
# ------------------------------------------------------------------------------
params = {
    # device
    "device": "cuda:0",
    "random_seed": 7,
    "data_shuffle": True,
    "num_workers": 8,
    # "complie": True,
    "complie": False,
    "enable_amp": False,
    "enable_gradscaler": False,
    # model parameters ---------------------------------------------------------
    # "model_name": "dfcan",
    "model_name": "rln",
    # loss function ------------------------------------------------------------
    # "loss": "mse",
    "loss": "mae",
    # learning rate ------------------------------------------------------------
    # 2D real ------------------------------------------------------------------
    # "lr": 0.001,
    # "batch_size": 16, # 2D
    # "num_epochs": 15000,
    # 3D real ------------------------------------------------------------------
    # "lr": 0.01,
    # "batch_size": 4,  # 3D
    # "num_epochs": 700,
    # 3D simu ------------------------------------------------------------------
    "lr": 0.01,
    "batch_size": 1,  # 3D
    "num_epochs": 380,
    # --------------------------------------------------------------------------
    "warm_up": 0,
    "lr_decay_every_iter": 10000,
    "lr_decay_rate": 0.5,
    "lr_min": 0.0000001,
    "save_every_iter": 1000,
    "plot_every_iter": 100,
    "print_loss": False,
    # validation ---------------------------------------------------------------
    "enable_validation": False,
    "frac_val": 0.2,
    "validate_every_iter": 500,
    # dataset ------------------------------------------------------------------
    "path_dataset_excel": "datasets_train.xlsx",
    "datasets_id": [
        # "F-actin-nonlinear-9",
        # "F-actin-nonlinear-8",
        # "F-actin-nonlinear-7",
        # "F-actin-nonlinear-6",
        # "F-actin-nonlinear-5",
        # "F-actin-nonlinear-4",
        # "F-actin-nonlinear-3",
        # "F-actin-nonlinear-2",
        # "F-actin-nonlinear-1",
        # "Microtubules2-9",
        # "Microtubules2-8",
        # "Microtubules2-7",
        # "Microtubules2-6",
        # "Microtubules2-5",
        # "Microtubules2-4",
        # "Microtubules2-3",
        # "Microtubules2-2",
        # "Microtubules2-1",
        # "CCPs-9",
        # "CCPs-8",
        # "CCPs-7",
        # "CCPs-6",
        # "CCPs-5",
        # "CCPs-4",
        # "CCPs-3",
        # "CCPs-2",
        # "CCPs-1",
        # "F-actin-12",
        # "F-actin-11",
        # "F-actin-10",
        # "F-actin-9",
        # "F-actin-8",
        # "F-actin-7",
        # "F-actin-6",
        # "F-actin-5",
        # "F-actin-4",
        # "F-actin-3",
        # "F-actin-2",
        # "F-actin-1",
        # "ER-6",
        # "ER-5",
        # "ER-4",
        # "ER-3",
        # "ER-2",
        # "ER-1",
        # "Microtubule2-3d-1024",
        # "Nuclear-pore-complex2-1024",
        # "SimuMix3D-128-31-0-0-1",
        # "SimuMix3D-128-31-05-1-1",
        # "SimuMix3D-128-31-05-1-03",
        "SimuMix3D-128-31-05-1-01",
    ],
    "sample_range": (0, 80),
    "scale_factor": 1,
    "normalization": (False, False),
    "normalization_eva": (0.03, 0.995),
    "data_clip_eva": (0.0, 2.5),
    # checkpoints --------------------------------------------------------------
    "suffix": "",
    "path_checkpoints": "checkpoints",
    # --------------------------------------------------------------------------
    "saved_checkpoint": None,
}

# ------------------------------------------------------------------------------
device = torch.device(params["device"])
torch.manual_seed(params["random_seed"])
dict_clip = {
    "min": torch.Tensor([params["data_clip_eva"][0]]).to(device),
    "max": torch.Tensor([params["data_clip_eva"][1]]).to(device),
}
data_range = params["data_clip_eva"][1] - params["data_clip_eva"][0]
params["suffix"] += f"_id_{params['sample_range'][0]}_{params['sample_range'][1]}"

path_save_model = os.path.join(
    params["path_checkpoints"],
    params["datasets_id"][0],
    params["model_name"],
    "{}_{}_bs_{}_lr_{}{}".format(
        params["model_name"],
        params["loss"],
        params["batch_size"],
        params["lr"],
        params["suffix"],
    ),
)
os.makedirs(path_save_model, exist_ok=True)

# ------------------------------------------------------------------------------
#                                 load dataset
# ------------------------------------------------------------------------------
assert len(params["datasets_id"]) == 1, "[ERROR] Only one dataset is supported."
data_frame = pandas.read_excel(params["path_dataset_excel"])
info = data_frame[data_frame["id"] == params["datasets_id"][0]].iloc[0]

if "Simu" not in params["datasets_id"][0]:
    path_dataset_lr = win2linux(info["path_lr"]) + "_patch"
    path_dataset_hr = win2linux(info["path_hr"]) + "_patch"
    path_index_file = win2linux(info["path_txt"]).replace(
        ".txt", f"_patch_{params['sample_range'][0]}_{params['sample_range'][1]}.txt"
    )
else:
    path_dataset_lr = win2linux(info["path_lr"]) + "_norm"
    path_dataset_hr = win2linux(info["path_hr"]) + "_norm"
    path_index_file = win2linux(info["path_txt"]).replace(
        ".txt", f"_{params['sample_range'][0]}_{params['sample_range'][1]}.txt"
    )

# ratio = info["ratio"]
ratio = 1.0

for path in [path_dataset_lr, path_dataset_hr, path_index_file]:
    assert os.path.exists(path), f"[ERROR] {path} does not exist."

params.update(
    {
        "path_dataset_lr": path_dataset_lr,
        "path_dataset_hr": path_dataset_hr,
        "path_index_file": path_index_file,
        "ratio": float(ratio),
        "dim": int(info["ndim"]),
    }
)

normalizer = NormalizePercentile(
    params["normalization_eva"][0],
    params["normalization_eva"][1],
    ndim=params["dim"],
)

dataset_all = SRDataset(
    hr_root_path=path_dataset_hr,
    lr_root_path=path_dataset_lr,
    hr_txt_file_path=path_index_file,
    lr_txt_file_path=path_index_file,
    normalization=params["normalization"],
    transform=None,
    id_range=None,
)

# create training and validation dataset
dataloader_train, dataloader_val = None, None
if params["enable_validation"]:
    # split whole dataset into training and validation dataset
    dataset_train, dataset_validation = random_split(
        dataset_all,
        [1.0 - params["frac_val"], params["frac_val"]],
        generator=torch.Generator().manual_seed(7),
    )

    dataloader_val = DataLoader(
        dataset=dataset_validation,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=params["num_workers"],
    )
    num_batch_val = len(dataloader_val)
else:
    dataset_train = dataset_all
    num_batch_val = 0

dataloader_train = DataLoader(
    dataset=dataset_train,
    batch_size=params["batch_size"],
    shuffle=params["data_shuffle"],
    num_workers=params["num_workers"],
)
num_batches_train = len(dataloader_train)

# ------------------------------------------------------------------------------
# data infomation
img_lr_shape = dataset_train[0]["lr"].shape
img_hr_shape = dataset_train[0]["hr"].shape


print(f"[INFO] Num of Batches (train| valid): {num_batches_train}|{num_batch_val}")
print(f"[INFO] Image shape: (input) {img_lr_shape} | (gt) {img_hr_shape}")

# print parameters in the dict
print("-" * 80)
print("[INFO] Parameters:")
for key, value in params.items():
    print(f"- {key}: {value}")
# save parameters
with open(os.path.join(path_save_model, f"params-{today_date}.json"), "w") as f:
    f.write(json.dumps(params, indent=1))

# ------------------------------------------------------------------------------
#                                     model
# ------------------------------------------------------------------------------
# 2D models
if params["model_name"] == "dfcan" and params["dim"] == 2:
    print("[INFO] Using DFCAN model (2D version).")
    model = DFCAN(
        in_channels=1,
        scale_factor=params["scale_factor"],
        num_features=64,
        num_groups=4,
    )
elif params["model_name"] == "dfcan" and params["dim"] == 3:
    print("[INFO] Using DFCAN model (3D version).")
    model = DFCAN3D(
        in_channels=1,
        scale_factor=params["scale_factor"],
        num_features=64,
        num_groups=4,
    )
elif params["model_name"] == "rln" and params["dim"] == 3:
    print("[INFO] Using RLN model (3D version).")
    model = RLN3D(
        scale=params["scale_factor"],
        in_channels=1,
        n_features=4,
        kernel_size=3,
    )
else:
    print(f"[ERROR] Model name ({params['model_name']}) is not supported.")

model.to(device=device)

# complie
if params["complie"]:
    print("[INFO] Complie model.")
    model = torch.compile(model)

# load pre-trained model parameters --------------------------------------------
if params["saved_checkpoint"] is not None:
    print(f"[INFO] Load saved pre-trained model parameters:")
    print(f"[INFO] {params['saved_checkpoint']}")
    state_dict = torch.load(
        params["saved_checkpoint"],
        map_location=device,
        weights_only=True,
    )["model_state_dict"]
    state_dict = on_load_checkpoint(state_dict, complie_mode=params["complie"])
    model.load_state_dict(state_dict)
    start_iter = params["saved_checkpoint"].split(".")[-2].split("_")[-1]
    start_iter = int(start_iter)
    print(f"[INFO] Start iteration: {start_iter}")
else:
    print("[INFO] Training model from scratch.")
    start_iter = 0

# ------------------------------------------------------------------------------
summary(model=model, input_size=(1,) + img_lr_shape)

model_parameters = list(model.named_parameters())
num_parameters = sum(p[1].numel() for p in model_parameters if p[1].requires_grad)
torch.set_float32_matmul_precision("high")
print(f"[INFO] Number of trainable parameters: {num_parameters}")

# ------------------------------------------------------------------------------
# optimization
# ------------------------------------------------------------------------------
optimizer = torch.optim.Adam(params=model_parameters, lr=params["lr"])
log_writer = SummaryWriter(os.path.join(path_save_model, "log"))

LR_schedule = StepLR_iter(
    lr_start=params["lr"],
    optimizer=optimizer,
    decay_every_iter=params["lr_decay_every_iter"],
    lr_min=params["lr_min"],
    warm_up=params["warm_up"],
    decay_rate=params["lr_decay_rate"],
)
LR_schedule.init(start_iter)

# ------------------------------------------------------------------------------
# train
# ------------------------------------------------------------------------------
print(
    f"[INFO] Batch size: {params['batch_size']} | Num of Batches: {num_batches_train}"
)
print(f"[INFO] Save model to {path_save_model}")

scaler = torch.GradScaler("cuda", enabled=params["enable_gradscaler"])

disable_pbar_epoch = True if num_batches_train > 10 else False
try:
    pbar_epoch = tqdm.tqdm(
        total=params["num_epochs"], desc="Epoch", ncols=80, disable=disable_pbar_epoch
    )
    for i_epoch in range(params["num_epochs"]):
        pbar = tqdm.tqdm(
            total=num_batches_train,
            desc=f"Epoch {i_epoch + 1}|{params['num_epochs']}",
            ncols=80,
            disable=not disable_pbar_epoch,
        )
        pbar_epoch.update(1)

        # ----------------------------------------------------------------------
        for i_batch, data in enumerate(dataloader_train):
            i_iter = (i_batch + i_epoch * num_batches_train) + start_iter
            pbar.update(1)

            imgs_lr, imgs_hr = data["lr"], data["hr"] * params["ratio"]
            imgs_lr, imgs_hr = imgs_lr.to(device), imgs_hr.to(device)

            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=params["enable_amp"]
            ):
                imgs_est = model(imgs_lr)

                if params["loss"] == "mse":
                    loss = torch.nn.MSELoss()(imgs_est, imgs_hr)
                elif params["loss"] == "mae":
                    loss = torch.nn.L1Loss()(imgs_est, imgs_hr)
                else:
                    raise ValueError(
                        f"Loss function {params['loss']} is not supported."
                    )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update learning rate ---------------------------------------------
            LR_schedule.update(i_iter=i_iter)

            # evaluation -------------------------------------------------------
            if params["print_loss"]:
                imgs_hr = torch.clamp(normalizer(imgs_hr), **dict_clip)
                imgs_est = torch.clamp(normalizer(imgs_est), **dict_clip)

                dict_eva = dict(
                    img_true=imgs_hr, img_test=imgs_est, data_range=data_range
                )
                psnr = utils_eva.PSNR_tb(**dict_eva)
                ssim = utils_eva.SSIM_tb(**dict_eva)

                if i_iter % 10 == 0:
                    pbar.set_postfix(
                        Loss=f"{loss.cpu().detach().numpy():>.4f}, PSNR: {psnr:>.4f}, SSIM: {ssim:>.4f}"
                    )

            # ------------------------------------------------------------------
            # log
            if i_iter % params["plot_every_iter"] == 0:
                if log_writer is not None:
                    log_writer.add_scalar(
                        "Learning rate", optimizer.param_groups[-1]["lr"], i_iter
                    )
                    log_writer.add_scalar(params["loss"], loss, i_iter)
                    if params["print_loss"]:
                        log_writer.add_scalar("PSNR", psnr, i_iter)
                        log_writer.add_scalar("SSIM", ssim, i_iter)

            if i_iter % params["save_every_iter"] == 0:
                print(f"\n[INFO] Save model (epoch: {i_epoch}, iter: {i_iter})")
                model_dict = {
                    "model_state_dict": getattr(model, "_orig_mod", model).state_dict()
                }
                torch.save(
                    model_dict,
                    os.path.join(path_save_model, f"epoch_{i_epoch}_iter_{i_iter}.pt"),
                )

            # ------------------------------------------------------------------
            # validation
            if (i_iter % params["validate_every_iter"] == 0) and (
                params["enable_validation"] == True
            ):
                pbar_val = tqdm.tqdm(desc="VALIDATION", total=num_batch_val, ncols=80)
                model.eval()  # convert model to evaluation model

                # --------------------------------------------------------------
                running_val_ssim, running_val_psnr, running_val_mse = 0, 0, 0
                with torch.autocast(
                    "cuda", torch.float16, enabled=params["enable_amp"]
                ):
                    with torch.no_grad():
                        for i_batch_val, data_val in enumerate(dataloader_val):
                            imgs_lr_val, imgs_hr_val = (
                                data_val["lr"],
                                data_val["hr"] * params["ratio"],
                            )

                            imgs_lr_val = imgs_lr_val.to(device)
                            imgs_hr_val = imgs_hr_val.to(device)

                            imgs_est_val = model(imgs_lr_val)

                            # evaluation
                            # linear transform
                            imgs_est_val = torch.clamp(
                                normalizer(imgs_est_val), **dict_clip
                            )
                            imgs_hr_val = torch.clamp(
                                normalizer(imgs_hr_val), **dict_clip
                            )

                            dict_eva_val = dict(
                                img_true=imgs_hr_val, img_test=imgs_est_val
                            )
                            mse_val = utils_eva.MSE(**dict_eva_val)
                            psnr_val = utils_eva.PSNR_tb(
                                data_range=data_range, **dict_eva_val
                            )
                            ssim_val = utils_eva.SSIM_tb(
                                data_range=data_range, **dict_eva_val
                            )

                            if not np.isinf(psnr_val):
                                running_val_psnr += psnr_val
                                running_val_ssim += ssim_val
                                running_val_mse += mse_val

                            if i_batch_val % 10 == 0:
                                pbar_val.set_postfix(
                                    PSNR="{:>.6f}, SSIM= {:>.6f}, MSE={:>.4f}".format(
                                        running_val_psnr / (i_batch_val + 1),
                                        running_val_ssim / (i_batch_val + 1),
                                        running_val_mse / (i_batch_val + 1),
                                    )
                                )
                            pbar_val.update(1)

                del imgs_lr_val, imgs_hr_val

                if log_writer is not None:
                    log_writer.add_scalar(
                        "psnr_val", running_val_psnr / num_batch_val, i_iter
                    )
                    log_writer.add_scalar(
                        "ssim_val", running_val_ssim / num_batch_val, i_iter
                    )
                    log_writer.add_scalar(
                        f"{params['loss']}_val", running_val_mse / num_batch_val, i_iter
                    )
                pbar_val.close()
                # convert model to train mode
                model.train(True)
        pbar.close()
    pbar_epoch.close()

    # ------------------------------------------------------------------------------
    # save and finish
    # ------------------------------------------------------------------------------
    print(f"\n[INFO] save model (epoch: {i_epoch}, iter: {i_iter})")

    # saving general checkpoint
    model_dict = {"model_state_dict": getattr(model, "_orig_mod", model).state_dict()}
    torch.save(
        model_dict,
        os.path.join(path_save_model, f"epoch_{i_epoch}_iter_{i_iter}.pt"),
    )

    log_writer.flush()
    log_writer.close()
    print("[INFO] Training done.")

except KeyboardInterrupt:
    print("\n[INFO] Training stop, saving model ...")
    print(f"\n[INFO] save model (epoch: {i_epoch}, iter: {i_iter})")

    # saving general checkpoint
    model_dict = {"model_state_dict": getattr(model, "_orig_mod", model).state_dict()}
    torch.save(
        model_dict,
        os.path.join(path_save_model, f"epoch_{i_epoch}_iter_{i_iter}.pt"),
    )

    pbar.close()
    pbar_epoch.close()
    log_writer.flush()
    log_writer.close()
    print("[INFO] Training done.")
