"""
Train `kernelnet` (forward + backward) model for deconvolution.
1.  Select the forward or backward model to train. (`model_name`). You should
    first learn the forward projector before learning the backward projector.
2.  Select the dataset to train. (`dataset_name`)
3.  Set the training parameters (the learning rate may need to be adjusted
    according to the dataset)
"""

import torch, os, time, pandas, tqdm, json, statistics, cupy
from cupyx.scipy.ndimage import median_filter
import numpy as np
import skimage.io as io
import utils.evaluation as utils_eva

# from fft_conv_pytorch import fft_conv
from methods.deconvolution import fftn_conv_real as fft_conv
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.data import win2linux, SRDataset, text2tuple, NormalizePercentile
from utils.optimize import step_lr_schedule
from checkpoint_list import checkpoints_v1 as checkpoints_list
from models import kernelnet

torch.manual_seed(7)
# ------------------------------------------------------------------------------
# model_name = "kernet_fp"
model_name = "kernet"

# ------------------------------------------------------------------------------
datasets_id = (
    # "F-actin-nonlinear-9",
    # "Microtubules2-9",
    # "CCPs-9",
    # "ER-6",
    # "F-actin-9",
    # ----------------------------------------------------------------------
    # "Microtubules2-6",
    # "CCPs-6",
    # "ER-6",
    # "F-actin-6",
    # ----------------------------------------------------------------------
    # "Microtubules2-3",
    # "CCPs-3",
    # "ER-3",
    # "F-actin-3",
    # ----------------------------------------------------------------------
    # "SimuBeads3D-128-31-0-0-1",
    # "SimuBeads3D-128-31-05-1-1",
    # "SimuBeads3D-128-31-05-1-03",
    # "SimuBeads3D-128-31-05-1-01",
    # ----------------------------------------------------------------------
    # "SimuMix3D-128-31-0-0-1",
    # "SimuMix3D-128-31-05-1-1",
    # "SimuMix3D-128-31-05-1-03",
    # "SimuMix3D-128-31-05-1-01",
    # "SimuMix3D-128-31-05-1-01-31x31",
    # ----------------------------------------------------------------------
    # "SimuMix3D-256-31-0-0-1",
    # "SimuMix3D-256-31-05-1-1",
    # "SimuMix3D-256-31-05-1-03",
    # "SimuMix3D-256-31-05-1-01",
    # ----------------------------------------------------------------------
    # "SimuMix3D-382-101-05-1-1-560",
    # "SimuMix3D-382-101-05-1-1-642",
    # ----------------------------------------------------------------------
    # "Microtubule2-3d-1024",
    # "Nuclear-pore-complex2-1024",
    # ----------------------------------------------------------------------
    # "SirDNA-1024",
    # ----------------------------------------------------------------------
    # "biotisr-3d-factin-1",
    # "biotisr-3d-factin-2",
    # "biotisr-3d-mt-1",
    # "biotisr-3d-mt-2",
    # "biotisr-3d-mito-1",
    "biotisr-3d-mito-2",
    # ----------------------------------------------------------------------
    # "ZeroShotDeconvNet-642",
    # "ZeroShotDeconvNet-560",
    # ----------------------------------------------------------------------
    # "deepbacs-ecoli-ave2",
    # "deepbacs-saureus-ave2",
    # ----------------------------------------------------------------------
    # "biotisr-ccps-1",
    # "biotisr-ccps-2",
    # "biotisr-ccps-3",
    # "biotisr-factin-1",
    # "biotisr-factin-2",
    # "biotisr-factin-3",
    # "biotisr-factin-nonlinear-1",
    # "biotisr-factin-nonlinear-2",
    # "biotisr-factin-nonlinear-3",
    # "biotisr-lysosomes-1",
    # "biotisr-lysosomes-2",
    # "biotisr-lysosomes-3",
    # "biotisr-mt-1",
    # "biotisr-mt-2",
    # "biotisr-mt-3",
    # "biotisr-mito-1",
    # "biotisr-mito-2",
    # "biotisr-mito-3",
    # ----------------------------------------------------------------------
    # "w2s-0-sim-ave",
    # "w2s-1-sim-ave",
    # "w2s-2-sim-ave",
    # "w2s-0-wf-ave-400",
    # "w2s-1-wf-ave-400",
    # "w2s-2-wf-ave-400",
    # ----------------------------------------------------------------------
)


# ------------------------------------------------------------------------------
for dataset_id in datasets_id:
    params = {
        "device_id": "cuda:1",
        "num_workers": 6,
        # --------------------------------------------------------------------------
        "batch_size": 1,
        "normalization": (False, False),
        "in_channels": 1,
        "data_clip_eva": (0, 2.5),
        # --------------------------------------------------------------------------
        "FP_type": "pre-trained",  # real 2d or 3d data
        # "FP_type": "known",  # simulated data
        "BP_type": None,
        "conv_mode": "fft",
        "padding_mode": "reflect",
        "kernel_init": "gauss",
        "interpolation": True,
        "kernel_norm_fp": False,  # default
        # "kernel_norm_fp": True,
        "kernel_norm_bp": True,
        "over_sampling": 2,
        # --------------------------------------------------------------------------
        "experiment": "n1_r1",
        "sample_range": (0, 1),
        # "sample_range": (0, 2),
        # "sample_range": (0, 3),
        # "sample_range": (0, 4),
        # "sample_range": (0, 5),
        # "loss_function": "mse",
        "loss_function": "mae",
        "use_lr_schedule": True,
        "scheduler_cus": {
            "lr": 0.00001,
            "every": 2000,  # 300
            "rate": 0.5,
            "min": 0.00000001,
        },
        "warm_up": 0,
        "eva_during_train": False,
        # --------------------------------------------------------------------------
        "validation_enable": True,
        "sample_range_val": (10, 20),
        # --------------------------------------------------------------------------
        "path_info": "datasets_train.xlsx",
        "normalization_eva": (0.03, 0.995),
        "path_checkpoint_save": "checkpoints",
        # "saved_checkpoint": "checkpoints/SimuMix3D-128-31-05-1-01/kernelnet/backward/kernet_bs_1_lr_1e-06_iter_1_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3_median_in/epoch_9999_9999.pt",
        # "saved_checkpoint": "checkpoints/SimuMix3D-128-31-05-1-01/kernelnet/backward/kernet_bs_1_lr_1e-06_iter_1_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_2_v3_median_in/epoch_5000_10000.pt",
        # "saved_checkpoint": "checkpoints/SimuMix3D-128-31-05-1-01/kernelnet/backward/kernet_bs_1_lr_1e-06_iter_1_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_3_v3_median_in/epoch_3333_10000.pt",
        # "saved_checkpoint": "checkpoints/SimuMix3D-128-31-05-1-01/kernelnet/backward/kernet_bs_1_lr_1e-06_iter_1_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_4_v3_median_in/epoch_2500_10000.pt",
        # "saved_checkpoint": "checkpoints/SimuMix3D-128-31-05-1-01/kernelnet/backward/kernet_bs_1_lr_1e-06_iter_1_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_5_v3_median_in/epoch_2000_10000.pt",
        # "saved_checkpoint": "checkpoints/CCPs-9/kernelnet/backward/kernet_bs_1_lr_0.0001_iter_1_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3/epoch_9999_9999.pt",
        # "saved_checkpoint": "checkpoints/Microtubule2-3d-1024/kernelnet/backward/kernet_bs_1_lr_1e-05_iter_1_ker_31_mse_over2_inter_fp_normx_bp_norm_fft_ts_0_1_v3_median_in/epoch_9999_9999.pt",
        "saved_checkpoint": None,
    }

    # --------------------------------------------------------------------------
    print("-" * 80)
    print("[INFO] Dataset ID (train):", dataset_id)
    model_part = "forward" if model_name == "kernet_fp" else "backward"

    df_info = pandas.read_excel(win2linux(params["path_info"]))
    info = df_info[df_info["id"] == dataset_id].iloc[0]

    enable_median_filter = int(info["median_filter"])
    print(f"[INFO] Enable median filter: {enable_median_filter}")
    enable_dark = int(info["dark"])
    print(f"[INFO] Enable dark: {enable_dark}")

    path_fp = None
    if model_part == "backward" and params["FP_type"] == "pre-trained":
        try:
            path_fp = checkpoints_list[dataset_id]["forward"][params["experiment"]]
        except:
            print("[WARNNING] No pre-trained forward model available.")

    params.update(
        {
            "FP_path": path_fp,
            "PSF_path": win2linux(info["path_psf"]),
            "ndim": int(info["ndim"]),
            "hr_root_path": win2linux(info["path_hr"]),
            "lr_root_path": win2linux(info["path_lr"]),
            "hr_txt_file_path": win2linux(info["path_txt"]),
            "lr_txt_file_path": win2linux(info["path_txt"]),
            "kernel_size_fp": text2tuple(info["kf_size"]),
            "kernel_size_bp": text2tuple(info["kb_size"]),
            "scale_factor": int(info["scale_factor"]),
            "ratio": float(info["ratio"]),
            "std_init": text2tuple(info["ker_std_init"]),
            "num_iter_fp": int(info["num_iter_fp"]),
            "num_iter_bp": int(info["num_iter_bp"]),
            "sample_range_val": eval(info["id_sample_val"]),
        }
    )

    device = torch.device(params["device_id"])
    training_data_size = params["sample_range"][1] - params["sample_range"][0]
    ker_size_fp = params["kernel_size_fp"]
    ker_size_bp = params["kernel_size_bp"]

    # ------------------------------------------------------------------------------
    if model_name == "kernet_fp":
        norm_tag = "norm" if params["kernel_norm_fp"] else "normx"

        suffix = f"_ker_{ker_size_fp}_{params['loss_function']}_over{params['over_sampling']}_inter_{norm_tag}_{params['conv_mode']}_ts_{params['sample_range'][0]}_{params['sample_range'][1]}_s100_v3"
        if enable_median_filter:
            suffix += "_median_in"
        if enable_dark:
            suffix += "_dark"
        params.update(
            {
                "multi_out": False,
                "self_supervised": False,
                "optimizer_type": "adam",  # real data
                # "optimizer_type": "lbfgs",
                "save_every_iter": 100,
                "plot_every_iter": 2,
                "val_every_iter": 100,
                "print_every_iter": 1000,
            }
        )
        # start_learning_rate = 1  # simumix
        # start_learning_rate = 0.1
        start_learning_rate = 0.01  # 3d real
        # start_learning_rate = 0.001  # 2d real
        # start_learning_rate = 0.0001
        # start_learning_rate = 0.00001
        epochs = params["num_iter_fp"]
    # ------------------------------------------------------------------------------
    elif model_name == "kernet":
        params.update(
            {
                # "num_iter": 1,
                # "num_iter": 2,  # default
                # "num_iter": 3,
                # "num_iter": 4,
                "num_iter": 5,
                # "num_iter": 6,
                # "num_iter": 7,
                # "num_iter": 10,
                # "num_iter": 11,
                "lam": 0.0,  # lambda for prior
                # "multi_out": False,
                "multi_out": True,
                "shared_bp": True,
                # "shared_bp": False,
                "self_supervised": False,
                # 'self_supervised': True,
                "optimizer_type": "adam",
                "save_every_iter": 1000,
                "plot_every_iter": 50,
                "val_every_iter": 1000,
                "print_every_iter": 1000,
            }
        )

        ss_marker = "_ss" if params["self_supervised"] else ""
        norm_tag = "fp_norm" if params["kernel_norm_fp"] else "fp_normx"
        norm_tag += "_bp_norm" if params["kernel_norm_bp"] else "_bp_normx"
        suffix = f"_iter_{params['num_iter']}_ker_{ker_size_bp}_{params['loss_function']}_over{params['over_sampling']}_inter_{norm_tag}_{params['conv_mode']}_ts_{params['sample_range'][0]}_{params['sample_range'][1]}{ss_marker}_v3"

        if params["multi_out"]:
            suffix += "_multiout"
        if params["saved_checkpoint"] is not None:
            suffix += "_continue"
        if enable_median_filter:
            suffix += "_median_in"
        if enable_dark:
            suffix += "_dark"
        if not params["shared_bp"]:
            suffix += "_noshared_bp"
        # start_learning_rate = 0.001
        # start_learning_rate = 0.0001  # 2D real
        start_learning_rate = 0.00001  # 3D real
        # start_learning_rate = 0.000001  # simumix
        # start_learning_rate = 0.000002  # simumix
        # start_learning_rate = 0.000005  # simumix
        epochs = params["num_iter_bp"]
    else:
        raise ValueError(f"[ERROR] Unknown model name: {model_name}")

    params["scheduler_cus"]["lr"] = start_learning_rate

    # ------------------------------------------------------------------------------
    # print params dict
    print("-" * 80)
    for key, value in params.items():
        print(f"[INFO] {key}: {value}")
    print("-" * 80)

    # ------------------------------------------------------------------------------
    #                                   Dataset
    # ------------------------------------------------------------------------------
    print("INFO] Load data...")
    dict_data = dict(
        hr_root_path=params["hr_root_path"],
        lr_root_path=params["lr_root_path"],
        hr_txt_file_path=params["hr_txt_file_path"],
        lr_txt_file_path=params["lr_txt_file_path"],
        normalization=params["normalization"],
    )

    if enable_dark:
        dict_data.update({"lr_root_path": params["lr_root_path"] + "_dark"})

    dict_dataloader = dict(
        batch_size=params["batch_size"], num_workers=params["num_workers"]
    )
    # Training data ----------------------------------------------------------------
    training_data = SRDataset(id_range=params["sample_range"], **dict_data)
    train_dataloader = DataLoader(
        dataset=training_data, shuffle=True, **dict_dataloader
    )

    # Validation data --------------------------------------------------------------
    if params["validation_enable"]:
        validation_data = SRDataset(id_range=params["sample_range_val"], **dict_data)
        valid_dataloader = DataLoader(
            dataset=validation_data, shuffle=False, **dict_dataloader
        )

    # ------------------------------------------------------------------------------
    #                                   Model
    # ------------------------------------------------------------------------------
    dict_model = dict(
        dim=params["ndim"],
        in_channels=params["in_channels"],
        scale_factor=params["scale_factor"],
        std_init=params["std_init"],
        init=params["kernel_init"],
        padding_mode=params["padding_mode"],
        conv_mode=params["conv_mode"],
        over_sampling=params["over_sampling"],
        interpolation=params["interpolation"],
    )

    # KernelNet (forward + backward) -----------------------------------------------
    if model_name == "kernet":
        print("[INFO] Backward Kernel Learning ...)")
        FP, BP = None, None

        # pre-trained forward projector --------------------------------------------
        if params["FP_type"] == "pre-trained":
            print("[INFO] Load pre-trained PSF ...")

            assert params["FP_path"] is not None, "[ERROR] FP path is not provided."
            assert os.path.exists(params["FP_path"]), "[ERROR] FP path does not exist."

            print(f"[INFO] Load from: {params['FP_path']}")

            # create forward projector
            FP = kernelnet.ForwardProject(
                kernel_size=params["kernel_size_fp"],
                kernel_norm=params["kernel_norm_fp"],
                trainable=False,
                **dict_model,
            )

            # load parameters
            FP.load_state_dict(
                torch.load(params["FP_path"], map_location=device, weights_only=True)[
                    "model_state_dict"
                ]
            )
            FP.eval()

        # known FP (i.e., PSF) -----------------------------------------------------
        elif params["FP_type"] == "known":
            print("[INFO] Load known PSF ...")

            if params["ndim"] == 3:  # only 3d data has PSF in our study
                psf_path = params["PSF_path"]

                assert psf_path is not None, "[ERROR] PSF path is not provided."
                assert os.path.exists(psf_path), "[ERROR] PSF path does not exist."
                assert psf_path.endswith(
                    ".tif"
                ), "[ERROR] PSF path should be a tif file."

                print("[INFO] Load from: ", psf_path)

                PSF_true = io.imread(psf_path)[None, None].astype(np.float32)
                PSF_true = torch.tensor(PSF_true).to(
                    device=device
                )  # [1, 1, Nz, Ny, Nx]
                PSF_true = torch.round(PSF_true, decimals=16)
                ks = PSF_true.shape

                # pad input image with a half kernel size
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
                    mode=params["padding_mode"],
                )
                if params["conv_mode"] == "direct":
                    conv_fp = lambda x: torch.nn.functional.conv3d(
                        input=padd_fp(x),
                        weight=PSF_true,
                        groups=params["in_channels"],
                    )
                elif params["conv_mode"] == "fft":
                    conv_fp = lambda x: fft_conv(
                        signal=padd_fp(x),
                        kernel=PSF_true,
                        groups=params["in_channels"],
                    )
                else:
                    raise ValueError(
                        f"[ERROR] Unknown conv mode: {params['conv_mode']}"
                    )

                FP = lambda x: torch.nn.functional.avg_pool3d(
                    conv_fp(x),
                    kernel_size=params["scale_factor"],
                    stride=params["scale_factor"],
                )
            else:
                raise ValueError(f"[ERROR] Unsupported ndim: {params['ndim']}")
        else:
            raise ValueError(f"[ERROR] Unknown FP type: {params['FP_type']}")

        # combine into a single model ----------------------------------------------
        model = kernelnet.KernelNet(
            num_iter=params["num_iter"],
            kernel_size_fp=params["kernel_size_fp"],
            kernel_size_bp=params["kernel_size_bp"],
            FP=FP,
            BP=None,
            lam=params["lam"],
            multi_out=params["multi_out"],
            kernel_norm=params["kernel_norm_bp"],
            shared_bp=params["shared_bp"],
            self_supervised=params["self_supervised"],
            **dict_model,
        )

        # load pre-trained model ---------------------------------------------------
        if params["saved_checkpoint"] is not None:
            print(f"[INFO] Load exist checkpoint from: {params['saved_checkpoint']}")
            model.load_state_dict(
                torch.load(
                    win2linux(params["saved_checkpoint"]),
                    map_location=device,
                )["model_state_dict"]
            )

    # Only the Forward pojector ----------------------------------------------------
    elif model_name == "kernet_fp":
        model = kernelnet.ForwardProject(
            kernel_size=params["kernel_size_fp"],
            kernel_norm=params["kernel_norm_fp"],
            trainable=True,
            **dict_model,
        )

    else:
        raise ValueError(f"[ERROR] Unknown model name: {model_name}")

    model = model.to(device)

    # if params["ndim"] == 2:
    #     summary(model, input_size=(1, 1, 128, 128), device=device)
    # if params["ndim"] == 3:
    #     summary(model, input_size=(1, 1, 128, 128, 128), device=device)

    utils_eva.count_parameters(model)

    # ------------------------------------------------------------------------------
    # save
    path_model_save_to = os.path.join(
        params["path_checkpoint_save"],
        dataset_id,
        "kernelnet",
        model_part,
        f"{model_name}_bs_{params['batch_size']}_lr_{start_learning_rate}{suffix}",
    )

    print("[INFO] Save model to", path_model_save_to)
    writer = SummaryWriter(os.path.join(path_model_save_to, "log"))

    # save parameters into a json file
    path_params_json = os.path.join(path_model_save_to, "params.json")
    with open(path_params_json, "w") as f:
        json.dump(params, f, indent=4)

    # ------------------------------------------------------------------------------
    #                                   Training
    # ------------------------------------------------------------------------------
    # loass function
    if params["loss_function"] == "mse":
        loss_main = torch.nn.MSELoss()
    elif params["loss_function"] == "mae":
        loss_main = torch.nn.L1Loss()

    # optimizer --------------------------------------------------------------------
    if params["optimizer_type"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=start_learning_rate)
    elif params["optimizer_type"] == "lbfgs":
        # optimizer = torch.optim.LBFGS(model.parameters(), lr=start_learning_rate)
        optimizer = torch.optim.LBFGS(
            model.parameters(), lr=start_learning_rate, line_search_fn="strong_wolfe"
        )
    else:
        raise ValueError(
            f"[ERROR] Unsupported optimizer type: {params['optimizer_type']}"
        )

    # ------------------------------------------------------------------------------
    num_batches = len(train_dataloader)
    num_batches_val = (
        len(valid_dataloader) if params["validation_enable"] == True else 0
    )

    print("-" * 80)
    print("[INFO] Start training ... ")
    print(f"[INFO] Start time: {time.asctime(time.localtime(time.time()))}")
    print(f"[INFO] Num of batches: (train | valid) {num_batches} | {num_batches_val}")
    print(f"[INFO] Training under self-supervised mode: {params['self_supervised']}")

    # ------------------------------------------------------------------------------
    # pre-load data to save trianing time
    if training_data_size == 1:
        sample = training_data[0]
        x, y = sample["lr"].to(device)[None], sample["hr"].to(device)[None]
        y = y * params["ratio"]
    elif training_data_size > 1:
        x, y = [], []
        for i in range(training_data_size):
            sample = training_data[i]
            x.append(sample["lr"])
            y.append(sample["hr"])
        x = torch.stack(x).to(device)
        y = torch.stack(y).to(device)
        y = y * params["ratio"]
    else:
        print("[ERROR] Training data size is 0!")
    print(f"[INFO] Pre-load training data shape: {x.shape} | {y.shape}")

    if enable_median_filter:
        print("[INFO] Enable median filter...")
        # median filter to remove noise in x
        x_sum = x.sum()
        x = cupy.asarray(x)
        tmp_x = cupy.zeros_like(x)
        for i in range(x.shape[0]):
            tmp_x[i, 0] = median_filter(x[i, 0], size=3)
        x = torch.as_tensor(tmp_x, device=device)
        x = x * (x_sum / x.sum())

    print(f"[INFO] Num of baches: {num_batches}")
    print(f"[INFO] Epoch: {epochs} | Batch size: {params['batch_size']}")
    print("-" * 80)

    # ------------------------------------------------------------------------------
    normalizer_eva = NormalizePercentile(
        params["normalization_eva"][0],
        params["normalization_eva"][1],
        ndim=params["ndim"],
    )

    dict_clip = {
        "min": torch.Tensor([params["data_clip_eva"][0]]).to(device),
        "max": torch.Tensor([params["data_clip_eva"][1]]).to(device),
    }
    data_range = params["data_clip_eva"][1] - params["data_clip_eva"][0]
    # ------------------------------------------------------------------------------
    pbar = tqdm.tqdm(total=epochs, desc="Training", ncols=80)
    for i_epoch in range(epochs):
        ave_ssim, ave_psnr = 0, 0
        print_loss, print_ssim, print_psnr = [], [], []

        model.train()
        for i_batch in range(num_batches):
            i_iter = i_batch + i_epoch * num_batches  # index of iteration
            pbar.update(1)

            # load data
            # x, y = sample['lr'].to(device), sample['hr'].to(device)
            # y = y * ratio

            # set input and target
            if model_name == "kernet_fp":
                inpt, gt = y, x
            elif model_name == "kernet":
                if params["self_supervised"]:
                    inpt, gt = x, x
                else:
                    inpt, gt = x, y
            else:
                print("[ERROR] Model name is not defined!")

            # optimize -------------------------------------------------------------
            if params["optimizer_type"] == "lbfgs":
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
                scheduler_cus=params["scheduler_cus"],
                warm_up=params["warm_up"],
                use_lr_schedule=params["use_lr_schedule"],
            )

            # ----------------------------------------------------------------------
            out = pred if params["multi_out"] == False else pred[-1]
            # ----------------------------------------------------------------------

            if params["eva_during_train"]:
                # plot loss and metrics
                out = torch.clamp(normalizer_eva(out), **dict_clip)
                gt = torch.clamp(normalizer_eva(gt), **dict_clip)

                dict_eva = dict(img_true=gt, img_test=out, data_range=data_range)
                ave_psnr = utils_eva.PSNR_tb(**dict_eva)
                ave_ssim = utils_eva.SSIM_tb(**dict_eva)

            if i_iter % params["plot_every_iter"] == 0:
                if writer != None:
                    writer.add_scalar(params["loss_function"], loss, i_iter)
                    writer.add_scalar(
                        "Learning rate", optimizer.param_groups[-1]["lr"], i_iter
                    )
                    if params["eva_during_train"]:
                        writer.add_scalar("PSNR", ave_psnr, i_iter)
                        writer.add_scalar("SSIM", ave_ssim, i_iter)

            # ----------------------------------------------------------------------
            # save model and relative information
            if i_iter % params["save_every_iter"] == 0:
                # print("[INFO] Save model ...")
                torch.save(
                    {"model_state_dict": model.state_dict()},
                    os.path.join(path_model_save_to, f"epoch_{i_epoch}_{i_iter}.pt"),
                )

            # ----------------------------------------------------------------------
            # validation
            # ----------------------------------------------------------------------
            if (i_iter == 0 or (i_iter + 1) % params["val_every_iter"] == 0) and (
                params["validation_enable"] == True
            ):
                loss_val_list, ssim_val_list, psnr_val_list = [], [], []
                model.eval()
                with torch.no_grad():
                    for i_batch_val, sample_val in enumerate(valid_dataloader):
                        x_val = sample_val["lr"].to(device)
                        y_val = sample_val["hr"].to(device)

                        if enable_median_filter:
                            x_val = cupy.asarray(x_val)
                            tmp_x_val = cupy.zeros_like(x_val)
                            for i in range(x_val.shape[0]):
                                tmp_x_val[i, 0] = median_filter(x_val[i, 0], size=3)
                            x_val = torch.as_tensor(tmp_x_val, device=device)

                        if model_name == "kernet_fp":
                            inpt_val, gt_val = y_val, x_val
                        if model_name == "kernet":
                            inpt_val, gt_val = x_val, y_val

                        pred_val = model(inpt_val)
                        loss_val = loss_main(pred_val, gt_val)

                        out_val = (
                            pred_val[-1] if params["multi_out"] == True else pred_val
                        )

                        out_val = torch.clamp(normalizer_eva(out_val), **dict_clip)
                        gt_val = torch.clamp(normalizer_eva(gt_val), **dict_clip)
                        dict_eva_val = dict(
                            img_true=gt_val, img_test=out_val, data_range=data_range
                        )

                        ave_psnr = utils_eva.PSNR_tb(**dict_eva_val)
                        ave_ssim = utils_eva.SSIM_tb(**dict_eva_val)

                        loss_val_list.append(float(loss_val.cpu().detach().numpy()))
                        psnr_val_list.append(ave_psnr)
                        ssim_val_list.append(ave_ssim)

                if writer != None:
                    writer.add_scalar(
                        f"{params['loss_function']}_val",
                        statistics.mean(loss_val_list),
                        i_iter,
                    )
                    writer.add_scalar(
                        "psnr_val", statistics.mean(psnr_val_list), i_iter
                    )
                    writer.add_scalar(
                        "ssim_val", statistics.mean(ssim_val_list), i_iter
                    )
                model.train()
            # ----------------------------------------------------------------------
    pbar.close()
    # ------------------------------------------------------------------------------
    # save the last one model
    print(f"[INFO] Save model ... (Epoch: {i_epoch}, Iteration: {i_iter})")
    torch.save(
        {"model_state_dict": model.state_dict()},
        os.path.join(path_model_save_to, f"epoch_{i_epoch}_{i_iter}.pt"),
    )

    writer.flush()
    writer.close()
    print("[INFO] Training done!")
