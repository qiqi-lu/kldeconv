"""
Inference using KLDeconv algorithm.
"""

import torch, os, pandas, tqdm, json, time, cupy
from cupyx.scipy.ndimage import median_filter
import numpy as np
import skimage.io as io
import methods.deconvolution as dcv
from models import kernelnet

# from fft_conv_pytorch import fft_conv

from methods.deconvolution import fftn_conv_real as fft_conv

from utils.data import text2tuple, win2linux, SRDataset, padding_kernel, read_txt
from checkpoint_list import checkpoints_v1 as checkpoints_list

enable_prediction = False
enable_prediction = True
# ------------------------------------------------------------------------------
#                             Parameter setting
# ------------------------------------------------------------------------------
# id_device = "cpu"
id_device = "cuda:0"
# output_inter = True  # output intermediate results
output_inter = False  # not to output intermediate results

# FP_type, BP_type = "known", "learned"  # simulation data
# FP_type, BP_type = 'known', 'known'
FP_type, BP_type = "pre-trained", "learned"  # 2D and 3D real data
# FP_type, BP_type = 'pre-trained', 'known'

# ------------------------------------------------------------------------------
num_data_fp, id_repeat_fp = 1, 1
# ------------------------------------------------------------------------------
num_data_bp, id_repeat_bp = 1, 1
# num_data_bp, id_repeat_bp = 2, 1
# num_data_bp, id_repeat_bp = 3, 1
# num_data_bp, id_repeat_bp = 4, 1
# num_data_bp, id_repeat_bp = 5, 1

# ------------------------------------------------------------------------------
# num_iter_train = 1
# num_iter_train = 2
# num_iter_train = 3
# num_iter_train = 4
num_iter_train = 5

# num_iter_test = 2
num_iter_test = num_iter_train

# ------------------------------------------------------------------------------
#                  test dataset | train dataset
# ------------------------------------------------------------------------------
dataset_names_list = (
    # ("SimuMix3D-128-31-0-0-1", "SimuMix3D-128-31-0-0-1"),
    # ("SimuMix3D-128-31-05-1-01", "SimuMix3D-128-31-05-1-01"),
    # ("SimuMix3D-512-31-05-1-01", "SimuMix3D-128-31-05-1-01"),
    # ("SimuMix3D-1024-31-05-1-01", "SimuMix3D-128-31-05-1-01"),
    # ("SimuMix3D-128-31-05-1-03", "SimuMix3D-128-31-05-1-03"),
    # ("SimuMix3D-128-31-05-1-1", "SimuMix3D-128-31-05-1-1"),
    # --------------------------------------------------------------------------
    # ("Microtubule2-3d-1024", "Microtubule2-3d-1024"),
    # ("Nuclear-pore-complex2-1024", "Nuclear-pore-complex2-1024"),
    # --------------------------------------------------------------------------
    # ("SirDNA-1024", "SirDNA-1024"),
    # ("SirDNA-1024-train", "SirDNA-1024"),
    # ("SirDNA-1024-live-cell-1", "SirDNA-1024"),
    # ("SirDNA-1024-live-cell-2", "SirDNA-1024"),
    # --------------------------------------------------------------------------
    ("biotisr-3d-factin-1", "biotisr-3d-factin-1"),
    ("biotisr-3d-factin-2", "biotisr-3d-factin-2"),
    ("biotisr-3d-mito-1", "biotisr-3d-mito-1"),
    ("biotisr-3d-mito-2", "biotisr-3d-mito-2"),
    ("biotisr-3d-mt-1", "biotisr-3d-mt-1"),
    ("biotisr-3d-mt-2", "biotisr-3d-mt-2"),
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
    # --------------------------------------------------------------------------
    # ("Microtubules2-9", "Microtubules2-9"),
    # ("Microtubules2-9", "CCPs-9"),
    # ("Microtubules2-9", "ER-6"),
    # ("Microtubules2-9", "F-actin-9"),
    # ("ER-6", "Microtubules2-9"),
    # ("ER-6", "CCPs-9"),
    # ("ER-6", "ER-6"),
    # ("ER-6", "F-actin-9"),
    # ("CCPs-9", "Microtubules2-9"),
    # ("CCPs-9", "CCPs-9"),
    # ("CCPs-9", "ER-6"),
    # ("CCPs-9", "F-actin-9"),
    # ("F-actin-9", "Microtubules2-9"),
    # ("F-actin-9", "CCPs-9"),
    # ("F-actin-9", "ER-6"),
    # ("F-actin-9", "F-actin-9"),
    # # --------------------------------------------------------------------------
    # ("Microtubules2-6", "Microtubules2-6"),
    # ("Microtubules2-6", "CCPs-6"),
    # ("Microtubules2-6", "ER-6"),
    # ("Microtubules2-6", "F-actin-6"),
    # ("ER-6", "Microtubules2-6"),
    # ("ER-6", "CCPs-6"),
    # ("ER-6", "ER-6"),
    # ("ER-6", "F-actin-6"),
    # ("CCPs-6", "Microtubules2-6"),
    # ("CCPs-6", "CCPs-6"),
    # ("CCPs-6", "ER-6"),
    # ("CCPs-6", "F-actin-6"),
    # ("F-actin-6", "Microtubules2-6"),
    # ("F-actin-6", "CCPs-6"),
    # ("F-actin-6", "ER-6"),
    # ("F-actin-6", "F-actin-6"),
    # # --------------------------------------------------------------------------
    # ("Microtubules2-3", "Microtubules2-3"),
    # ("Microtubules2-3", "CCPs-3"),
    # ("Microtubules2-3", "ER-3"),
    # ("Microtubules2-3", "F-actin-3"),
    # ("ER-3", "Microtubules2-3"),
    # ("ER-3", "CCPs-3"),
    # ("ER-3", "ER-3"),
    # ("ER-3", "F-actin-3"),
    # ("CCPs-3", "Microtubules2-3"),
    # ("CCPs-3", "CCPs-3"),
    # ("CCPs-3", "ER-3"),
    # ("CCPs-3", "F-actin-3"),
    # ("F-actin-3", "Microtubules2-3"),
    # ("F-actin-3", "CCPs-3"),
    # ("F-actin-3", "ER-3"),
    # ("F-actin-3", "F-actin-3"),
)


for dataset_names in dataset_names_list:
    # dataset_name_test, dataset_name_train = dataset_names
    dataset_name_train, dataset_name_test = dataset_names

    # id_sample = [0, 346, 609, 700, 770, 901]
    # id_sample = [0, 1, 2, 3, 4, 5]
    # id_sample = range(0, 1000, 4)
    # id_sample = [0, 1, 2, 3, 4, 5, 6]
    # id_sample = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    # id_sample = [0, 1, 2, 3, 4, 5, 6]
    # id_sample = [0]
    id_sample = []  # use all the samples
    # id_sample = None # will only save the kernels

    # --------------------------------------------------------------------------
    path_prediction = os.path.join(
        "outputs", "predictions", dataset_name_test, "kernelnet", dataset_name_train
    )
    # --------------------------------------------------------------------------

    if FP_type == "known" and BP_type == "learned":
        folder = f"fp_knonw_bp_n{num_data_bp}_r{id_repeat_bp}"
    elif FP_type == "pre-trained" and BP_type == "learned":
        folder = f"fp_n{num_data_fp}_r{id_repeat_fp}_bp_n{num_data_bp}_r{id_repeat_bp}"
    elif FP_type == "pre-trained" and BP_type == "known":
        folder = f"fp_n{num_data_fp}_r{id_repeat_fp}_bp_known"
    elif FP_type == "known" and BP_type == "known":
        folder = f"fp_known_bp_known"
    else:
        raise ValueError("Invalid FP_type and BP_type")

    path_prediction = os.path.join(path_prediction, folder)
    os.makedirs(path_prediction, exist_ok=True)

    # --------------------------------------------------------------------------
    info_xlsx = pandas.read_excel("datasets_test.xlsx")
    info = info_xlsx[info_xlsx["id"] == dataset_name_test].iloc[0]

    enable_median_filter = int(info["median_filter"])
    print("-" * 80)
    print(f"[INFO] Enable median filter: {enable_median_filter}")

    params_dict = dict(
        kernel_size_fp=text2tuple(info["ks_fp"]),
        kernel_size_bp=text2tuple(info["ks_bp"]),
        dim=int(info["ndim"]),
        ratio=float(info["ratio"]),
        eps=0.000001,
        scale_factor=int(info["scale_factor"]),
        interpolation=True,
        kernel_norm_fp=False,  # default
        # kernel_norm_fp=True,
        kernel_norm_bp=True,
        over_sampling=2,
        padding_mode="reflect",
        std_init=text2tuple(info["ker_std_init"]),
        shared_bp=True,
        conv_mode="fft",
        lr_root_path=win2linux(info["path_lr"]),
        hr_root_path=win2linux(info["path_hr"]),
        lr_txt_file_path=win2linux(info["path_txt"]),
        hr_txt_file_path=win2linux(info["path_txt"]),
        path_psf=win2linux(info["path_psf"]),
        normalization=(False, False),
        in_channels=1,
        train_mode=info["train_mode"],
        num_iter_test=num_iter_test,
    )

    # ------------------------------------------------------------------------------
    device = torch.device(id_device)
    suffix_net = "_ss" if params_dict["train_mode"] == "ss" else ""

    params_dict["conv_mode"] = "direct" if params_dict["dim"] == 2 else "fft"
    # params_dict["conv_mode"] = "fft"

    print("-" * 80)
    print(f"[INFO] Dataset (test): {dataset_name_test}")
    print(f"[INFO] Dataset (train): {dataset_name_train}")
    print(f"[INFO] Device: {id_device}")
    print(f"[INFO] Output intermediate results: {output_inter}")

    # print all the elements in the dict
    print("-" * 80)
    for key, value in params_dict.items():
        print(f"[INFO] {key}: {value}")
    print("-" * 80)

    # ------------------------------------------------------------------------------
    #                                  Dataset
    # ------------------------------------------------------------------------------
    if os.path.exists(params_dict["path_psf"]):
        print(f'[INFO] Load PSF from: {params_dict["path_psf"]}')
        PSF_true = io.imread(params_dict["path_psf"]).astype(np.float32)
    else:
        print(f"[WARNNING] PSF not found, use all zeros.")
        PSF_true = np.zeros(shape=params_dict["kernel_size_fp"]).astype(np.float32)

    # ------------------------------------------------------------------------------
    print(f'[INFO] Load LR data from: {params_dict["lr_root_path"]}')
    print(f"[INFO] Load HR data from: {params_dict['hr_root_path']}")

    dataset_test = SRDataset(
        hr_root_path=params_dict["hr_root_path"],
        lr_root_path=params_dict["lr_root_path"],
        hr_txt_file_path=params_dict["hr_txt_file_path"],
        lr_txt_file_path=params_dict["lr_txt_file_path"],
        normalization=params_dict["normalization"],
        id_range=None,
    )

    filenames = read_txt(params_dict["lr_txt_file_path"])
    num_samples_all = len(filenames)
    if id_sample == []:
        # use all the samples
        id_sample = list(range(num_samples_all))

    # ------------------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------------------
    FP, BP = None, None
    # Forward Projection
    print("-" * 80)
    # ------------------------------------------------------------------------------
    if FP_type == "pre-trained":
        FP_path = checkpoints_list[dataset_name_train]["forward"][
            f"n{num_data_fp}_r{id_repeat_fp}"
        ]
        FP_path = win2linux(FP_path)

        assert os.path.exists(FP_path), f"[ERROR] FP_path not found: {FP_path}"
        print("[INFO] Use pre-trained forward projector.")
        print(f"[INFO] Load model from : {FP_path}")

        FP = kernelnet.ForwardProject(
            dim=params_dict["dim"],
            in_channels=params_dict["in_channels"],
            scale_factor=params_dict["scale_factor"],
            kernel_size=params_dict["kernel_size_fp"],
            std_init=params_dict["std_init"],
            init="gauss",
            kernel_norm=params_dict["kernel_norm_fp"],
            padding_mode=params_dict["padding_mode"],
            interpolation=params_dict["interpolation"],
            over_sampling=params_dict["over_sampling"],
            conv_mode=params_dict["conv_mode"],
        ).to(device)

        ker_fp_init = FP.conv.get_kernel().cpu().detach().numpy()[0, 0]  # initial PSF
        FP.load_state_dict(torch.load(FP_path, map_location=device)["model_state_dict"])
        FP.eval()

    # ------------------------------------------------------------------------------
    if FP_type == "known":
        print("[INFO] Use known PSF as forward projector.")
        ks = PSF_true.shape
        weight = torch.tensor(PSF_true[None, None]).to(device=device)
        # pad image
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
        # conv
        if params_dict["conv_mode"] == "direct":
            conv_fp = lambda x: torch.nn.functional.conv3d(
                input=padd_fp(x), weight=weight, groups=params_dict["in_channels"]
            )
        if params_dict["conv_mode"] == "fft":
            conv_fp = lambda x: fft_conv(
                signal=padd_fp(x), kernel=weight, groups=params_dict["in_channels"]
            )
        # forward projection
        # FP = lambda x: torch.nn.functional.avg_pool3d(
        #     conv_fp(x),
        #     kernel_size=params_dict["scale_factor"],
        #     stride=params_dict["scale_factor"],
        # )
        FP = lambda x: conv_fp(x)

        ker_FP = weight.cpu().numpy()[0, 0]
        # The PSF now is known, setting the initial PSF as all zeros.
        ker_fp_init = np.zeros_like(ker_FP)

    # ------------------------------------------------------------------------------
    # Backward Projection
    if BP_type == "known":
        print("[INFO] Use known BP kernel.")
        BP = lambda x: dcv.Convolution(
            PSF=ker_FP,
            x=x.detach().numpy()[0, 0],
            padding_mode=params_dict["padding_mode"],
            domain=params_dict["conv_mode"],
        )
        ker_BP = PSF_true

    # ------------------------------------------------------------------------------
    model = kernelnet.KernelNet(
        in_channels=params_dict["in_channels"],
        scale_factor=params_dict["scale_factor"],
        dim=params_dict["dim"],
        num_iter=params_dict["num_iter_test"],
        kernel_size_fp=params_dict["kernel_size_fp"],
        kernel_size_bp=params_dict["kernel_size_bp"],
        std_init=params_dict["std_init"],
        init="gauss",
        padding_mode=params_dict["padding_mode"],
        FP=FP,
        BP=BP,
        lam=0.0,
        return_inter=True,
        multi_out=False,
        over_sampling=params_dict["over_sampling"],
        kernel_norm=params_dict["kernel_norm_bp"],
        interpolation=params_dict["interpolation"],
        shared_bp=params_dict["shared_bp"],
        conv_mode=params_dict["conv_mode"],
    ).to(device)

    # ------------------------------------------------------------------------------
    if BP_type == "learned":
        model_path = checkpoints_list[dataset_name_train]["backward"][
            f"n{num_data_bp}_r{id_repeat_bp}_iter{num_iter_train}"
        ]
        model_path = win2linux(model_path)

        assert os.path.exists(model_path), f"[ERROR] model_path not found: {model_path}"
        print("[INFO] Use learned BP kernel.")
        print(f"[INFO] Load model from : {model_path}")

        model.load_state_dict(
            torch.load(model_path, map_location=device)["model_state_dict"],
            strict=False,
        )
        model.eval()

        # get the learned BP kernel
        if params_dict["shared_bp"] == True:
            ker_BP = model.BP.conv.get_kernel()[0, 0].cpu().detach().numpy()
        else:
            ker_BP = model.BP[0].conv.get_kernel()[0, 0].detach().numpy()

    print("[INFO] BP kernel shape:", ker_BP.shape)

    # ------------------------------------------------------------------------------
    if FP_type == "pre-trained":
        # get the FP learned FP kernel
        ker_FP = model.FP.conv.get_kernel()[0, 0].cpu().detach().numpy()
        print("[INFO] FP kernel shape:", ker_FP.shape)

    # ------------------------------------------------------------------------------
    # Save kernels
    # ------------------------------------------------------------------------------
    path_save_kernel = os.path.join(path_prediction, f"kernel_iter_{num_iter_train}")
    save_kernel = lambda fname, arr: io.imsave(
        fname=os.path.join(path_save_kernel, fname), arr=arr, check_contrast=False
    )

    os.makedirs(path_save_kernel, exist_ok=True)
    print("[INFO] save kernels to:", path_save_kernel)

    ker_fp_init = padding_kernel(ker_fp_init, PSF_true)
    ker_FP = padding_kernel(ker_FP, PSF_true)
    ker_BP = padding_kernel(ker_BP, PSF_true)

    print(f"[INFO] Sum of FP kernel: {np.sum(ker_FP)}")
    print(f"[INFO] Sum of BP kernel: {np.sum(ker_BP)}")

    save_kernel("kernel_true.tif", PSF_true)
    save_kernel("kernel_init.tif", ker_fp_init)
    save_kernel("kernel_fp.tif", ker_FP)
    save_kernel(f"kernel_bp{suffix_net}.tif", ker_BP)

    # ------------------------------------------------------------------------------
    #                                   Prediction
    # ------------------------------------------------------------------------------
    if not enable_prediction:
        exit()

    print("-" * 80)
    print("[INFO] Prediciton ...")

    # if no sample to process, exit.
    if id_sample is None:
        print("[INFO] id_sample is None, exit.")
        os._exit(0)

    # save parameters dict into a json file.
    params_dict.update(
        {
            "id_sample": id_sample,
            "FP_type": FP_type,
            "BP_type": BP_type,
            "num_data_fp": num_data_fp,
            "id_repeat_fp": id_repeat_fp,
            "num_data_bp": num_data_bp,
            "id_repeat_bp": id_repeat_bp,
            "output_inter": output_inter,
            "id_device": id_device,
            "dataset_name_test": dataset_name_test,
        }
    )
    path_params_json = os.path.join(path_prediction, "params.json")
    with open(path_params_json, "w") as f:
        json.dump(params_dict, f, indent=4)
    print(f"[INFO] save parameters to: {path_params_json}")

    t2n = lambda x: x.cpu().detach().numpy()[0, 0]  # tensor to numpy
    clamp = lambda x: torch.clamp(x, min=0.0, max=3.0)  # clamp the value to [0, 3]

    pbar = tqdm.tqdm(total=len(id_sample), desc="Prediction", ncols=80)
    print_time_each_sample = True
    time_list = []
    for i in id_sample:
        if i >= dataset_test.__len__():
            print(f"[ERROR] Sample {i} is out of range, exit.")
            break

        data = dataset_test[i]  # load one sample

        x = torch.unsqueeze(data["lr"], 0)
        # median filter to remove the noise
        if enable_median_filter:
            x = cupy.asarray(x)
            x = median_filter(x, size=3)
            x = torch.as_tensor(x, device=device)
        else:
            x = x.to(device)
        y = torch.unsqueeze(data["hr"], 0).to(device) * params_dict["ratio"]

        # intermedia results -------------------------------------------------------
        if output_inter:
            # forward projection of gt
            y_fp = model.FP(y)

            # forward projection of the initial guess
            x0 = torch.nn.functional.interpolate(
                x, scale_factor=params_dict["scale_factor"], mode="nearest-exact"
            )
            # x0 = model.constraint(x0)
            x0_fp = model.FP(x0)

            # backward projeciton
            if params_dict["shared_bp"]:
                bp = model.BP(clamp(x / (x0_fp + params_dict["eps"])))
            else:
                bp = model.BP[0](clamp(x / (x0_fp + params_dict["eps"])))

            # tensor to numpy
            y_fp, x0, x0_fp = t2n(y_fp), t2n(x0), t2n(x0_fp)
            if BP_type == "learned":
                bp = t2n(bp)

        # final results ------------------------------------------------------------
        # measure the time used for prediction
        torch.cuda.synchronize(device=device)
        tic = time.time()
        y_pred_all = model(x)
        torch.cuda.synchronize(device=device)
        toc = time.time()
        used_time = toc - tic
        time_list.append(used_time)
        if print_time_each_sample:
            print(f"[INFO] Sample {i}: {used_time:.4f} s")
        y_pred_all = y_pred_all.cpu().detach().numpy()[:, 0, 0]
        y, x = t2n(y), t2n(x)
        pbar.update(1)

        # Save results -------------------------------------------------------------
        path_sample = os.path.join(
            path_prediction, f"train_iter_{num_iter_train}", filenames[i].split(".")[0]
        )
        os.makedirs(path_sample, exist_ok=True)

        save_image = lambda fname, arr: io.imsave(
            fname=os.path.join(path_sample, fname), arr=arr, check_contrast=False
        )

        if output_inter:
            save_image("y_fp.tif", y_fp)
            save_image("x0.tif", x0)
            save_image("x0_fp.tif", x0_fp)
            save_image("bp.tif", bp)
        save_image("y.tif", y)
        save_image("x.tif", x)

        if "ZeroShotDeconvNet" in dataset_name_test:
            # this dataset have many images.
            # only save the resutl from the last iteration to save memory.
            save_image("y_pred_all.tif", y_pred_all[-1].astype(np.uint16))
        else:
            save_image("y_pred_all.tif", y_pred_all.astype(np.float32))
    pbar.close()
    print(f"[INFO] Results have been saved into: {path_prediction}")

    # save the time used for prediction into excel ---------------------------------
    df = pandas.DataFrame(columns=["time (s)"])
    df["time (s)"] = time_list
    df.to_excel(
        os.path.join(path_prediction, f"train_iter_{num_iter_train}", "time.xlsx"),
        index=True,
    )
    # ------------------------------------------------------------------------------
