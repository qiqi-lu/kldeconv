"""
KLDeconv trianing.
"""

import torch, os, time, sys, pandas
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import skimage.io as io
from fft_conv_pytorch import fft_conv
from utils.data import win2linux, SRDataset, text2tuple
from utils import evaluation as eva
from models import kernelnet

# ------------------------------------------------------------------------------
print("-" * 80)
if sys.platform == "win32":
    device, num_workers = torch.device("cpu"), 0
    root_path = os.path.join("F:", os.sep, "Datasets")

if sys.platform == "linux" or sys.platform == "linux2":
    device, num_workers = torch.device("cpu"), 0
    # device, num_workers = torch.device("cuda"), 6
    root_path = "data"

print(f"[INFO] Device:{device} | Num of workers:{num_workers}")

# ------------------------------------------------------------------------------
torch.manual_seed(7)
input_normalization = 0
path_checkpoint = os.path.join("checkpoints", "v2")
validation_enable = False
data_range = None
normalization = (False, False)
in_channels = 1

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

# ------------------------------------------------------------------------------
# model_name = 'kernet_fp'
model_name = "kernet"

# ------------------------------------------------------------------------------
# load dataset info
dataset_id = dataset_name[0]
# load excel file to get dataset info
df = pandas.read_excel(os.path.join("datasets_train.xlsx"))
info = df.loc[df["id"] == dataset_id].loc[0]

params_dict = dict(
    dataset_dim=info["ndim"],
    hr_root_path=win2linux(info["path_hr"]),
    lr_root_path=win2linux(info["path_lr"]),
    hr_txt_file_path=win2linux(info["path_txt"]),
    lr_txt_file_path=win2linux(info["path_txt"]),
    kernel_size_fp=text2tuple(info["kf_size"]),
    kernel_size_bp=text2tuple(info["kb_size"]),
    scale_factor=info["scale_factor"],
    id_range=text2tuple(info["id_sample"]),
    std_init=info["ker_std_init"],
    epoch_fp=info["epoch_fp"],
    epoch_bp=info["epoch_bp"],
    FP_path=win2linux(info["path_fp"]),
    conv_mode="fft",
    padding_mode="reflect",
    kernel_init="gauss",
    interpolation=True,
    kernel_norm_fp=False,
    kernel_norm_bp=True,
    over_sampling=2,
)

training_data_size = params_dict["id_range"][1] - params_dict["id_range"][0]
ker_size_fp = params_dict["kernel_size_fp"][-1]
ker_size_bp = params_dict["kernel_size_bp"][-1]

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
warm_up = 0
use_lr_schedule = True
scheduler_cus = {}
scheduler_cus["lr"] = start_learning_rate
scheduler_cus["every"] = 2000  # 300
scheduler_cus["rate"] = 0.5
scheduler_cus["min"] = 0.00000001

# ------------------------------------------------------------------------------
if dataset_dim == 2:
    if model_name == "kernet":
        save_every_iter, plot_every_iter, val_every_iter = 1000, 50, 1000
        print_every_iter = 1000
    if model_name == "kernet_fp":
        save_every_iter, plot_every_iter, val_every_iter = 5, 2, 1000
        print_every_iter = 1000

if dataset_dim == 3:
    if model_name == "kernet":
        save_every_iter, plot_every_iter, val_every_iter = 1000, 50, 1000
        print_every_iter = 1000
    if model_name == "kernet_fp":
        save_every_iter, plot_every_iter, val_every_iter = 5, 2, 1000
        print_every_iter = 1000

# ------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------
# Training data
training_data = SRDataset(
    hr_root_path=hr_root_path,
    lr_root_path=lr_root_path,
    hr_txt_file_path=hr_txt_file_path,
    lr_txt_file_path=lr_txt_file_path,
    normalization=normalization,
    id_range=[0, training_data_size],
)

train_dataloader = DataLoader(
    dataset=training_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
)

# ------------------------------------------------------------------------------
# Validation data
if validation_enable == True:
    validation_data = SRDataset(
        hr_root_path=hr_root_path,
        lr_root_path=lr_root_path,
        hr_txt_file_path=hr_txt_file_path,
        lr_txt_file_path=lr_txt_file_path,
        normalization=normalization,
        id_range=[training_data_size, -1],
    )

    valid_dataloader = DataLoader(
        dataset=validation_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

# ------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------
if model_name == "kernet":
    FP, BP = None, None
    # FP_type, BP_type = 'pre-trained', None
    FP_type, BP_type = "known", None

    if dataset_id in [
        "Microtubule",
        "Microtubule2",
        "Nuclear_Pore_complex",
        "Nuclear_Pore_complex2",
        "F-actin_Nonlinear",
        "Microtubules2",
    ]:
        FP_type, BP_type = "pre-trained", None
        print("[INFO] Pre-trained forward kernel, and to learn backward kernel.")

    # --------------------------------------------------------------------------
    if FP_type == "pre-trained":
        print("[INFO] Pred-trained PSF")

        # load FP parameters
        FP = kernelnet.ForwardProject(
            dim=dataset_dim,
            in_channels=in_channels,
            scale_factor=scale_factor,
            kernel_size=kernel_size_fp,
            std_init=std_init,
            padding_mode=padding_mode,
            init=kernel_init,
            trainable=False,
            interpolation=interpolation,
            kernel_norm=kernel_norm_fp,
            over_sampling=over_sampling,
            conv_mode=conv_mode,
        )

        FP_para = torch.load(FP_path, map_location=device)
        FP.load_state_dict(FP_para["model_state_dict"])
        FP.eval()

        print(">> Load from: ", FP_path)

    if FP_type == "known":
        print(">> Known PSF")
        if dataset_dim == 2:
            ks, std = 25, 2.0
            ker = kernelnet.gauss_kernel_2d(shape=[ks, ks], std=std).to(device=device)
            ker = ker.repeat(repeats=(in_channels, 1, 1, 1))
            padd_fp = lambda x: torch.nn.functional.pad(
                input=x, pad=(ks // 2, ks // 2, ks // 2, ks // 2), mode=padding_mode
            )
            conv_fp = lambda x: torch.nn.functional.conv2d(
                input=padd_fp(x), weight=ker, groups=in_channels
            )
            FP = lambda x: torch.nn.functional.avg_pool2d(
                conv_fp(x), kernel_size=25, stride=scale_factor
            )

        if dataset_dim == 3:
            if dataset_id == "ZeroShotDeconvNet":
                psf_path = os.path.join(lr_root_path, "PSF_odd.tif")
            else:
                psf_path = os.path.join(lr_root_path, "PSF.tif")
            PSF_true = io.imread(psf_path).astype(np.float32)
            PSF_true = torch.tensor(PSF_true[None, None]).to(device=device)
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
                mode=padding_mode,
            )
            if conv_mode == "direct":
                conv_fp = lambda x: torch.nn.functional.conv3d(
                    input=padd_fp(x), weight=PSF_true, groups=in_channels
                )
            if conv_mode == "fft":
                conv_fp = lambda x: fft_conv(
                    signal=padd_fp(x), kernel=PSF_true, groups=in_channels
                )
            FP = lambda x: torch.nn.functional.avg_pool3d(
                conv_fp(x), kernel_size=scale_factor, stride=scale_factor
            )

            print(">> Load from :", psf_path)

    # --------------------------------------------------------------------------
    model = kernelnet.KernelNet(
        dim=dataset_dim,
        in_channels=in_channels,
        scale_factor=scale_factor,
        num_iter=num_iter,
        kernel_size_fp=kernel_size_fp,
        kernel_size_bp=kernel_size_bp,
        std_init=std_init,
        init=kernel_init,
        FP=FP,
        BP=BP,
        lam=lam,
        padding_mode=padding_mode,
        multi_out=multi_out,
        interpolation=interpolation,
        kernel_norm=kernel_norm_bp,
        over_sampling=over_sampling,
        shared_bp=shared_bp,
        self_supervised=self_supervised,
        conv_mode=conv_mode,
    ).to(device)

# ------------------------------------------------------------------------------
if model_name == "kernet_fp":
    model = kernelnet.ForwardProject(
        dim=dataset_dim,
        in_channels=in_channels,
        scale_factor=scale_factor,
        kernel_size=kernel_size_fp,
        std_init=std_init,
        init=kernel_init,
        padding_mode=padding_mode,
        trainable=True,
        kernel_norm=kernel_norm_fp,
        interpolation=interpolation,
        conv_mode=conv_mode,
        over_sampling=over_sampling,
    ).to(device)

# ------------------------------------------------------------------------------
eva.count_parameters(model)
print(model)

# ------------------------------------------------------------------------------
# save
if model_name == "kernet_fp":
    path_model = os.path.join(
        path_checkpoint,
        dataset_id,
        "forward",
        "{}_bs_{}_lr_{}{}".format(model_name, batch_size, start_learning_rate, suffix),
    )

if model_name == "kernet":
    path_model = os.path.join(
        path_checkpoint,
        dataset_id,
        "backward",
        "{}_bs_{}_lr_{}{}".format(model_name, batch_size, start_learning_rate, suffix),
    )

writer = SummaryWriter(os.path.join(path_model, "log"))
print(">> Save model to", path_model)

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

print(">> Start training ... ")
print(time.asctime(time.localtime(time.time())))

num_batches = len(train_dataloader)
num_batches_val = 0
if validation_enable == True:
    num_batches_val = len(valid_dataloader)

print(
    ">> Number of training batches: {}, validation batches: {}".format(
        num_batches, num_batches_val
    )
)

if self_supervised == True:
    print("Training under self-supervised mode.")

if training_data_size == 1:
    sample = training_data[0]
    x, y = sample["lr"].to(device)[None], sample["hr"].to(device)[None]
    y = y * ratio
else:
    x, y = [], []
    for i in range(training_data_size):
        sample = training_data[i]
        x.append(sample["lr"])
        y.append(sample["hr"])
    x = torch.stack(x)
    y = torch.stack(y)
    x, y = x.to(device), y.to(device)
    y = y * ratio

for i_epoch in range(epochs):
    print("\n" + "-" * 98)
    print(
        "Epoch: {}/{} | Batch size: {} | Num of Batches: {}".format(
            i_epoch + 1, epochs, batch_size, num_batches
        )
    )
    print("-" * 98)
    # --------------------------------------------------------------------------
    ave_ssim, ave_psnr = 0, 0
    print_loss, print_ssim, print_psnr = [], [], []

    start_time = time.time()
    # --------------------------------------------------------------------------
    model.train()
    # for i_batch, sample in enumerate(train_dataloader):
    for i_batch in range(num_batches):
        i_iter = i_batch + i_epoch * num_batches  # index of iteration
        # ----------------------------------------------------------------------
        # load data
        # x, y = sample['lr'].to(device), sample['hr'].to(device)
        # y = y * ratio

        if model_name == "kernet_fp":
            inpt, gt = y, x
        if model_name == "kernet":
            if self_supervised == True:
                inpt, gt = x, x
            else:
                inpt, gt = x, y

        # ----------------------------------------------------------------------
        # optimize
        if optimizer_type == "lbfgs":
            # L-BFGS
            loss = 0.0
            pred = 0.0

            def closure():
                global loss
                global pred
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
        if use_lr_schedule == True:
            if (warm_up > 0) and (i_iter < warm_up):
                lr = (i_iter + 1) / warm_up * scheduler_cus["lr"]
                # set learning rate
                for g in optimizer.param_groups:
                    g["lr"] = lr

            if i_iter >= warm_up:
                if (i_iter + 1 - warm_up) % scheduler_cus["every"] == 0:
                    lr = scheduler_cus["lr"] * (
                        scheduler_cus["rate"]
                        ** ((i_iter + 1 - warm_up) // scheduler_cus["every"])
                    )
                    lr = np.maximum(lr, scheduler_cus["min"])
                    for g in optimizer.param_groups:
                        g["lr"] = lr
        else:
            if (warm_up > 0) and (i_iter < warm_up):
                lr = (i_iter + 1) / warm_up * scheduler_cus["lr"]
                for g in optimizer.param_groups:
                    g["lr"] = lr

            if i_iter >= warm_up:
                for g in optimizer.param_groups:
                    g["lr"] = scheduler_cus["lr"]

        # ----------------------------------------------------------------------
        if multi_out == False:
            out = pred
        if multi_out == True:
            out = pred[-1]

        # ----------------------------------------------------------------------
        # plot loss and metrics
        if i_iter % plot_every_iter == 0:
            if dataset_dim == 2:
                ave_ssim, ave_psnr = eva.measure_2d(
                    img_test=out, img_true=gt, data_range=data_range
                )
            if dataset_dim == 3:
                ave_ssim, ave_psnr = eva.measure_3d(
                    img_test=out, img_true=gt, data_range=data_range
                )
            if writer != None:
                writer.add_scalar("loss", loss, i_iter)
                writer.add_scalar("psnr", ave_psnr, i_iter)
                writer.add_scalar("ssim", ave_ssim, i_iter)
                writer.add_scalar(
                    "Leanring Rate", optimizer.param_groups[-1]["lr"], i_iter
                )
            # if (i_iter > 5000) & (ave_psnr < 10.0):
            #     print('\nPSNR ({:>.4f}) is too low, break!'.format(ave_psnr))
            #     writer.flush()
            #     writer.close()
            #     os._exit(0)

        # ----------------------------------------------------------------------
        # print and save model
        if dataset_dim == 2:
            s, p = eva.measure_2d(img_test=out, img_true=gt, data_range=data_range)
        if dataset_dim == 3:
            s, p = eva.measure_3d(img_test=out, img_true=gt, data_range=data_range)
        print_loss.append(loss.cpu().detach().numpy())
        print_ssim.append(s)
        print_psnr.append(p)
        print("#", end="")

        if i_iter % print_every_iter == 0:
            print(
                "\nEpoch: {}, Iter: {}, loss: {:>.5f}, PSNR: {:>.5f},\
                SSIM: {:>.5f}".format(
                    i_epoch,
                    i_iter,
                    np.mean(print_loss),
                    np.mean(print_psnr),
                    np.mean(print_ssim),
                )
            )
            print("Computation time: {:>.2f} s".format(time.time() - start_time))
            start_time = time.time()
            print_loss, print_ssim, print_psnr = [], [], []

        # ----------------------------------------------------------------------
        # save model and relative information
        if i_iter % save_every_iter == 0:
            print("\nSave model ... (Epoch: {}, Iteration: {})".format(i_epoch, i_iter))
            model_dict = {"model_state_dict": model.state_dict()}
            torch.save(
                model_dict, os.path.join(path_model, "epoch_{}.pt".format(i_iter))
            )

        # ----------------------------------------------------------------------
        # validation
        if (i_iter % val_every_iter == 0) and (validation_enable == True):
            print("validation ...")
            running_val_loss, running_val_ssim, running_val_psnr = 0, 0, 0
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

                if multi_out == True:
                    out_val = pred_val[-1]
                if multi_out == False:
                    out_val = pred_val

                if dataset_dim == 2:
                    ave_ssim, ave_psnr = eva.measure_2d(
                        img_test=out_val, img_true=gt, data_range=data_range
                    )
                if dataset_dim == 3:
                    ave_ssim, ave_psnr = eva.measure_3d(
                        img_test=out_val, img_true=gt, data_range=data_range
                    )

                running_val_loss += loss_val.cpu().detach().numpy()
                running_val_psnr += ave_psnr
                running_val_ssim += ave_ssim
                print("#", end="")

            print(
                "\nValidation, Loss: {:>.5f}, PSNR: {:>.5f}, SSIM: {:>.5f}".format(
                    running_val_loss / num_batches_val,
                    running_val_psnr / num_batches_val,
                    running_val_ssim / num_batches_val,
                )
            )

            if writer != None:
                writer.add_scalar(
                    "loss_val", running_val_loss / num_batches_val, i_iter
                )
                writer.add_scalar(
                    "psnr_val", running_val_psnr / num_batches_val, i_iter
                )
                writer.add_scalar(
                    "ssim_val", running_val_ssim / num_batches_val, i_iter
                )
            model.train()

# ------------------------------------------------------------------------------
# save the last one model
print("\nSave model ... (Epoch: {}, Iteration: {})".format(i_epoch, i_iter + 1))
model_dict = {"model_state_dict": model.state_dict()}
torch.save(model_dict, os.path.join(path_model, "epoch_{}.pt".format(i_iter + 1)))

# ------------------------------------------------------------------------------
writer.flush()
writer.close()
print("Training done!")
