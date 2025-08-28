"""
KLDeconv training.
"""

import torch, os, time, sys, pandas, tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import skimage.io as io
from fft_conv_pytorch import fft_conv
from utils.data import win2linux, SRDataset, text2tuple
from utils.optimize import step_lr_schedule
from utils import evaluation as eva
from models import kernelnet
from torchinfo import summary
import statistics

# ------------------------------------------------------------------------------
torch.manual_seed(7)
input_normalization = 0
validation_enable = False
normalization = (False, False)

# ------------------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------------------
dataset_name = (
    "F-actin-nonlinear-9",
    # "Microtubules2-9",
    # "SimuBeads3D-128-31-0-0-1",
    # "SimuBeads3D-128-31-05-1-1",
    # "SimuBeads3D-128-31-05-1-03",
    # "SimuBeads3D-128-31-05-1-01",
    # "SimuMix3D-128-31-0-0-1",
    # "SimuMix3D-128-31-05-1-1",
    # "SimuMix3D-128-31-05-1-03",
    # "SimuMix3D-128-31-05-1-01",
    # "SimuMix3D-256-31-0-0-1",
    # "SimuMix3D-256-31-05-1-1",
    # "SimuMix3D-256-31-05-1-03",
    # "SimuMix3D-256-31-05-1-01",
    # "SimuMix3D-382-101-05-1-1-560",
    # "SimuMix3D-382-101-05-1-1-642",
    # "Microtubule-3d-128-0",
    # "Microtubule-3d-1024",
    # "Microtubule2-3d-512",
    # "Microtubule2-3d-1024",
    # "Nuclear-pore-complex-128-0",
    # "Nuclear-pore-complex-1024",
    # "Nuclear-pore-complex2-512",
    # "Nuclear-pore-complex2-1024",
    # "ZeroShotDeconvNet-642",
    # "ZeroShotDeconvNet-560",
)

assert len(dataset_name) == 1, "[ERROR] Only one dataset can be selected."
path_train_excel = os.path.join("datasets_train.xlsx")
dataset_id = dataset_name[0]

if dataset_id in [
    "Microtubule",
    "Microtubule2",
    "Nuclear_Pore_complex",
    "Nuclear_Pore_complex2",
    "F-actin_Nonlinear",
    "Microtubules2",
]:
    FP_type, BP_type = "pre-trained", None
    print("[INFO] Use pre-trained forward kernel, and learn backward kernel.")
else:
    FP_type, BP_type = "known", None
    print("[INFO] Known forward kernel, and to learn backward kernel.")


# ------------------------------------------------------------------------------
# model_name = 'kernet_fp'
model_name = "kernet"

# ------------------------------------------------------------------------------
# load dataset info
# load excel file to get dataset info
df = pandas.read_excel(path_train_excel)
info = df.loc[df["id"] == dataset_id].loc[0]

params_dict = dict(
    device="cuda:0",
    num_workers=6,
    path_checkpoint=os.path.join("checkpoints", "v2"),
    dataset_dim=info["ndim"],
    in_channels=1,
    hr_root_path=win2linux(info["path_hr"]),
    lr_root_path=win2linux(info["path_lr"]),
    hr_txt_file_path=win2linux(info["path_txt"]),
    lr_txt_file_path=win2linux(info["path_txt"]),
    kernel_size_fp=text2tuple(info["kf_size"]),
    kernel_size_bp=text2tuple(info["kb_size"]),
    scale_factor=info["scale_factor"],
    ratio=info["ratio"],
    id_range=text2tuple(info["id_sample"]),
    id_range_val=text2tuple(info["id_sample_val"]),
    std_init=info["ker_std_init"],
    epoch_fp=info["epoch_fp"],
    epoch_bp=info["epoch_bp"],
    FP_path=win2linux(info["path_fp"]),
    PSF_path=win2linux(info["path_psf"]),
    conv_mode="fft",
    padding_mode="reflect",
    kernel_init="gauss",
    interpolation=True,
    kernel_norm_fp=False,
    kernel_norm_bp=True,
    over_sampling=2,
    warm_up=0,
    use_lr_schedule=True,
    scheduler_cus={
        "lr": 0.00001,
        "every": 2000,  # 300
        "rate": 0.5,
        "min": 0.00000001,
    },
)

device = torch.device(params_dict["device"])
training_data_size = params_dict["id_range"][1] - params_dict["id_range"][0]
ker_size_fp = params_dict["kernel_size_fp"][-1]
ker_size_bp = params_dict["kernel_size_bp"][-1]

print(
    f"[INFO] Device:{params_dict['device']} | Num of workers:{params_dict['num_workers']}"
)
print(f"[INFO] Path to checkpoint: {params_dict['path_checkpoint']}")
print(f"[INFO] Dataset: {dataset_id} | Dim: {params_dict['dataset_dim']}")
print(f"[INFO] HR: {params_dict['hr_root_path']}")
print(f"[INFO] LR: {params_dict['lr_root_path']}")
print(f"[INFO] TXT: {params_dict['hr_txt_file_path']}")
print(f"[INFO] Kernel size FP: {params_dict['kernel_size_fp']}")
print(f"[INFO] Kernel size BP: {params_dict['kernel_size_bp']}")
print(f"[INFO] Scale factor: {params_dict['scale_factor']}")
print(
    f"[INFO] Train data size: {training_data_size} | ID range: {params_dict['id_range']}"
)
print(
    f"[INFO] Validation data size: {params_dict['id_range_val']} | ID range: {params_dict['id_range_val']}"
)
print(f"[INFO] Std init: {params_dict['std_init']}")
print(f"[INFO] Epoch FP: {params_dict['epoch_fp']} | BP: {params_dict['epoch_bp']}")
print(f"[INFO] FP path: {params_dict['FP_path']}")

# ------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------
if dataset_id in [
    "SimuMix3D_382",
    "ZeroShotDeconvNet",
    "Microtubule",
    "Microtubule2",
    "Nuclear_Pore_complex",
    "Nuclear_Pore_complex2",
]:
    batch_size = training_data_size

# ------------------------------------------------------------------------------
if model_name == "kernet_fp":
    suffix = f"_ker_{ker_size_fp}_mse_over{params_dict['over_sampling']}_inter_normx_{params_dict['conv_mode']}_ts_{params_dict['id_range'][0]}_{params_dict['id_range'][1]}_s100"
    multi_out = False
    self_supervised = False
    loss_main = torch.nn.MSELoss()
    optimizer_type = "adam"
    # start_learning_rate = 0.0001
    start_learning_rate = 0.001
    # optimizer_type = 'lbfgs'
    # start_learning_rate = 1
    epochs = params_dict["epoch_fp"]

if model_name == "kernet":
    num_iter = 2
    lam = 0.0  # lambda for prior
    multi_out = False
    shared_bp = True
    self_supervised = False
    # self_supervised = True

    if self_supervised:
        ss_marker = "_ss"
    else:
        ss_marker = ""

    suffix = f"_iter_{num_iter}_ker_{ker_size_bp}_mse_over{params_dict['over_sampling']}_inter_norm_{params_dict['conv_mode']}_ts_{params_dict['id_range'][0]}_{params_dict['id_range'][1]}{ss_marker}"

    loss_main = torch.nn.MSELoss()

    optimizer_type = "adam"
    if self_supervised:
        start_learning_rate = 0.000001
    else:
        # start_learning_rate = 0.00001
        start_learning_rate = 0.000001
    # start_learning_rate = 0.000001
    epochs = params_dict["epoch_bp"]

# ------------------------------------------------------------------------------
params_dict["scheduler_cus"]["lr"] = start_learning_rate
print_every_iter = 1000

if model_name == "kernet":
    save_every_iter, plot_every_iter, val_every_iter = 1000, 50, 1000
if model_name == "kernet_fp":
    save_every_iter, plot_every_iter, val_every_iter = 5, 2, 1000

# ------------------------------------------------------------------------------
#                                   Data
# ------------------------------------------------------------------------------
# Training data
training_data = SRDataset(
    hr_root_path=params_dict["hr_root_path"],
    lr_root_path=params_dict["lr_root_path"],
    hr_txt_file_path=params_dict["hr_txt_file_path"],
    lr_txt_file_path=params_dict["lr_txt_file_path"],
    normalization=normalization,
    id_range=params_dict["id_range"],
)

train_dataloader = DataLoader(
    dataset=training_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=params_dict["num_workers"],
)

# Validation data
if validation_enable == True:
    validation_data = SRDataset(
        hr_root_path=params_dict["hr_root_path"],
        lr_root_path=params_dict["lr_root_path"],
        hr_txt_file_path=params_dict["hr_txt_file_path"],
        lr_txt_file_path=params_dict["lr_txt_file_path"],
        normalization=normalization,
        id_range=params_dict["id_range_val"],
    )

    valid_dataloader = DataLoader(
        dataset=validation_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=params_dict["num_workers"],
    )

# ------------------------------------------------------------------------------
#                                   Model
# ------------------------------------------------------------------------------
if model_name == "kernet":
    FP, BP = None, None
    if FP_type == "pre-trained":
        print("[INFO] Load pre-trained PSF")
        print(f"[INFO] Load from: {params_dict['FP_path']}")

        # load FP parameters
        FP = kernelnet.ForwardProject(
            dim=params_dict["dataset_dim"],
            in_channels=params_dict["in_channels"],
            scale_factor=params_dict["scale_factor"],
            kernel_size=params_dict["kernel_size_fp"],
            std_init=params_dict["std_init"],
            padding_mode=params_dict["padding_mode"],
            init=params_dict["kernel_init"],
            trainable=False,
            interpolation=params_dict["interpolation"],
            kernel_norm=params_dict["kernel_norm_fp"],
            over_sampling=params_dict["over_sampling"],
            conv_mode=params_dict["conv_mode"],
        )

        FP_para = torch.load(params_dict["FP_path"], map_location=device)
        FP.load_state_dict(FP_para["model_state_dict"])
        FP.eval()

    if FP_type == "known":
        print("[INFO] Use known PSF")
        if params_dict["dataset_dim"] == 3:
            psf_path = params_dict["PSF_path"]
            print("[INFO] Load from: ", psf_path)

            assert psf_path is not None, "[ERROR] PSF path is not provided."
            assert os.path.exists(psf_path), "[ERROR] PSF path does not exist."
            assert psf_path.endswith(".tif"), "[ERROR] PSF path should be a tif file."

            PSF_true = io.imread(psf_path).astype(np.float32)
            PSF_true = torch.tensor(PSF_true[None, None]).to(
                device=device
            )  # [1, 1, Nz, Ny, Nx]
            PSF_true = torch.round(PSF_true, decimals=16)
            ks = PSF_true.shape
            padd_fp = lambda x: torch.nn.functional.pad(
                input=x,
                pad=(
                    ks[-1] // 2,
                    ks[-1] // 2,
                    ks[-2] // 2,
                    ks[-2] // 2,
                    ks[-3] // 2,
                    ks[-3] // 2,
                ),
                mode=params_dict["padding_mode"],
            )
            if params_dict["conv_mode"] == "direct":
                conv_fp = lambda x: torch.nn.functional.conv3d(
                    input=padd_fp(x), weight=PSF_true, groups=params_dict["in_channels"]
                )
            if params_dict["conv_mode"] == "fft":
                conv_fp = lambda x: fft_conv(
                    signal=padd_fp(x),
                    kernel=PSF_true,
                    groups=params_dict["in_channels"],
                )
            FP = lambda x: torch.nn.functional.avg_pool3d(
                conv_fp(x),
                kernel_size=params_dict["scale_factor"],
                stride=params_dict["scale_factor"],
            )

    # --------------------------------------------------------------------------
    model = kernelnet.KernelNet(
        dim=params_dict["dataset_dim"],
        in_channels=params_dict["in_channels"],
        scale_factor=params_dict["scale_factor"],
        num_iter=num_iter,
        kernel_size_fp=params_dict["kernel_size_fp"],
        kernel_size_bp=params_dict["kernel_size_bp"],
        std_init=params_dict["std_init"],
        init=params_dict["kernel_init"],
        FP=FP,
        BP=BP,
        lam=lam,
        padding_mode=params_dict["padding_mode"],
        multi_out=multi_out,
        interpolation=params_dict["interpolation"],
        kernel_norm=params_dict["kernel_norm_bp"],
        over_sampling=params_dict["over_sampling"],
        shared_bp=shared_bp,
        self_supervised=self_supervised,
        conv_mode=params_dict["conv_mode"],
    ).to(device)

# ------------------------------------------------------------------------------
if model_name == "kernet_fp":
    model = kernelnet.ForwardProject(
        dim=params_dict["dataset_dim"],
        in_channels=params_dict["in_channels"],
        scale_factor=params_dict["scale_factor"],
        kernel_size=params_dict["kernel_size_fp"],
        std_init=params_dict["std_init"],
        init=params_dict["kernel_init"],
        padding_mode=params_dict["padding_mode"],
        trainable=True,
        kernel_norm=params_dict["kernel_norm_fp"],
        interpolation=params_dict["interpolation"],
        conv_mode=params_dict["conv_mode"],
        over_sampling=params_dict["over_sampling"],
    ).to(device)

# ------------------------------------------------------------------------------
eva.count_parameters(model)
print(model)
if params_dict["dataset_dim"] == 2:
    summary(model, input_size=(1, 1, 128, 128))
if params_dict["dataset_dim"] == 3:
    summary(model, input_size=(1, 1, 128, 128, 128))

# ------------------------------------------------------------------------------
# save
if model_name == "kernet_fp":
    model_part = "forward"
if model_name == "kernet":
    model_part = "backward"

path_model = os.path.join(
    params_dict["path_checkpoint"],
    dataset_id,
    model_part,
    f"{model_name}_bs_{batch_size}_lr_{start_learning_rate}{suffix}",
)


print("[INFO] Save model to", path_model)
writer = SummaryWriter(os.path.join(path_model, "log"))

# ------------------------------------------------------------------------------
# OPTIMIZATION
# ------------------------------------------------------------------------------
if optimizer_type == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=start_learning_rate)
if optimizer_type == "lbfgs":
    # optimizer = torch.optim.LBFGS(model.parameters(), lr=start_learning_rate)
    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=start_learning_rate, line_search_fn="strong_wolfe"
    )


num_batches = len(train_dataloader)
num_batches_val = len(valid_dataloader) if validation_enable == True else 0

print("[INFO] Start training ... ")
print(f"[INFO] Start time: {time.asctime(time.localtime(time.time()))}")
print(f"[INFO] Num of batches: (train) {num_batches}, (valid) {num_batches_val}")
print(f"[INFO] Training under self-supervised mode: {self_supervised}")

# pre-load data to save trianing time
if training_data_size == 1:
    sample = training_data[0]
    x, y = sample["lr"].to(device)[None], sample["hr"].to(device)[None]
    y = y * params_dict["ratio"]
elif training_data_size > 1:
    x, y = [], []
    for i in range(training_data_size):
        sample = training_data[i]
        x.append(sample["lr"])
        y.append(sample["hr"])
    x = torch.stack(x)
    y = torch.stack(y)
    x, y = x.to(device), y.to(device)
    y = y * params_dict["ratio"]
else:
    print("[ERROR] Training data size is 0!")

print(f"[INFO] Num of baches: {num_batches}")
print(f"[INFO] Epoch: {epochs} | Batch size: {batch_size}")
print("-" * 80)

# ------------------------------------------------------------------------------
pbar = tqdm.tqdm(total=epochs, desc="Training", ncols=80)
for i_epoch in range(epochs):
    ave_ssim, ave_psnr = 0, 0
    print_loss, print_ssim, print_psnr = [], [], []

    model.train()
    for i_batch in range(num_batches):
        i_iter = i_batch + i_epoch * num_batches  # index of iteration

        # load data
        # x, y = sample['lr'].to(device), sample['hr'].to(device)
        # y = y * ratio

        # set input and target
        if model_name == "kernet_fp":
            inpt, gt = y, x
        elif model_name == "kernet":
            if self_supervised:
                inpt, gt = x, x
            else:
                inpt, gt = x, y
        else:
            print("[ERROR] Model name is not defined!")

        # optimize -------------------------------------------------------------
        if optimizer_type == "lbfgs":
            # L-BFGS optimizer, may be better for simulated data
            loss, pred = 0.0, 0.0

            def closure():
                global loss, pred
                pred = model(inpt)
                optimizer.zero_grad()
                loss = loss_main(pred, gt)
                loss.backward()
                return loss

            optimizer.step(closure)

        else:
            optimizer.zero_grad()
            pred = model(inpt)
            loss = loss_main(pred, gt)
            loss.backward()
            optimizer.step()

        # ----------------------------------------------------------------------
        # custom learning rate scheduler
        step_lr_schedule(
            optimizer=optimizer,
            i_iter=i_iter,
            scheduler_cus=params_dict["scheduler_cus"],
            warm_up=params_dict["warm_up"],
            use_lr_schedule=params_dict["use_lr_schedule"],
        )

        pbar.update(1)

        # ----------------------------------------------------------------------
        # plot loss and metrics
        out = pred if multi_out == False else pred[-1]

        if params_dict["dataset_dim"] == 2:
            s, p = eva.measure_2d(img_test=out, img_true=gt, data_range=None)
        if params_dict["dataset_dim"] == 3:
            s, p = eva.measure_3d(img_test=out, img_true=gt, data_range=None)
        loss_cpu = loss.cpu().detach().numpy()
        pbar.set_postfix({"loss": loss_cpu, "psnr": p, "ssim": s})

        if i_iter % plot_every_iter == 0:
            if writer != None:
                writer.add_scalar("loss", loss, i_iter)
                writer.add_scalar("psnr", ave_psnr, i_iter)
                writer.add_scalar("ssim", ave_ssim, i_iter)
                writer.add_scalar("lr", optimizer.param_groups[-1]["lr"], i_iter)

        # ----------------------------------------------------------------------
        # save model and relative information
        if i_iter % save_every_iter == 0:
            print("[INFO] Save model ...")
            model_dict = {"model_state_dict": model.state_dict()}
            torch.save(model_dict, os.path.join(path_model, f"epoch_{i_iter}.pt"))

        # ----------------------------------------------------------------------
        # validation
        if (i_iter % val_every_iter == 0) and (validation_enable == True):
            loss_val, ssim_val, psnr_val = [], [], []
            model.eval()
            for i_batch_val, sample_val in enumerate(valid_dataloader):
                x_val = sample_val["lr"].to(device)
                y_val = sample_val["hr"].to(device)
                if model_name == "kernel_fp":
                    inpt, gt = y_val, x_val
                if model_name == "kernet":
                    inpt, gt = x_val, y_val

                pred_val = model(inpt)
                loss_val = loss_main(pred_val, gt)

                out_val = pred_val[-1] if multi_out == True else pred_val

                if params_dict["dataset_dim"] == 2:
                    ave_ssim, ave_psnr = eva.measure_2d(
                        img_test=out_val, img_true=gt, data_range=None
                    )
                if params_dict["dataset_dim"] == 3:
                    ave_ssim, ave_psnr = eva.measure_3d(
                        img_test=out_val, img_true=gt, data_range=None
                    )

                loss_val.append(loss_val.cpu().detach().numpy())
                psnr_val.append(ave_psnr)
                ssim_val.append(ave_ssim)

            if writer != None:
                writer.add_scalar("loss_val", statistics.mean(loss_val), i_iter)
                writer.add_scalar("psnr_val", statistics.mean(psnr_val), i_iter)
                writer.add_scalar("ssim_val", statistics.mean(ssim_val), i_iter)
            model.train()
pbar.close()
# ------------------------------------------------------------------------------
# save the last one model
print(f"[INFO] Save model ... (Epoch: {i_epoch}, Iteration: {i_iter + 1})")
model_dict = {"model_state_dict": model.state_dict()}
torch.save(model_dict, os.path.join(path_model, f"epoch_{i_iter + 1}.pt"))

writer.flush()
writer.close()
print("Training done!")
