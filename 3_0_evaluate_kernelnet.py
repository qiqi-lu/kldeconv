import torch, os, pandas, tqdm, json
import numpy as np
import skimage.io as io
import methods.deconvolution as dcv
from models import kernelnet
from fft_conv_pytorch import fft_conv

# from
from utils.data import text2tuple, win2linux, SRDataset, padding_kernel
from checkpoint_list import checkpoints_v1 as checkpoints_list

# ------------------------------------------------------------------------------
#                             Parameter setting
# ------------------------------------------------------------------------------
# id_device = "cpu"
id_device = "cuda:0"
# output_inter = True  # output intermediate results
output_inter = False  # not to output intermediate results

# FP_type, BP_type = "known", "learned"
# FP_type, BP_type = 'known', 'known'
FP_type, BP_type = "pre-trained", "learned"
# FP_type, BP_type = 'pre-trained', 'known'

num_data_fp, id_repeat_fp = 1, 1
num_data_bp, id_repeat_bp = 1, 1

# id_sample = [0, 346, 609, 700, 770, 901]
# id_sample = [0, 1, 2, 3, 4, 5]
# id_sample = range(0, 1000, 4)
# id_sample = [0, 1, 2, 3, 4, 5, 6]
# id_sample = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
id_sample = [0, 1, 2, 3, 4, 5, 6]
# id_sample = [0]
# id_sample = None

# dataset_names = ("SimuMix3D-128-31-0-0-1", "SimuMix3D-128-31-0-0-1")
# dataset_names = ("Microtubule2-3d-1024", "Microtubule2-3d-1024")
# dataset_names = ("Microtubule2-3d-1024", "Nuclear-pore-complex2-1024")
# dataset_names = ("Nuclear-pore-complex2-1024", "Nuclear-pore-complex2-1024")
# dataset_names = ("Nuclear-pore-complex2-1024", "Microtubule2-3d-1024")
# dataset_names = ("F-actin-nonlinear-9", "F-actin-nonlinear-9")
# dataset_names = ("F-actin-nonlinear-9", "Microtubules2-9")
# dataset_names = ("Microtubules2-9", "F-actin-nonlinear-9")
# dataset_names = ("Microtubules2-9", "Microtubules2-9")
# dataset_names = ("Microtubules2-8", "Microtubules2-9")
# dataset_names = ("Microtubules2-7", "Microtubules2-9")
# dataset_names = ("Microtubules2-6", "Microtubules2-9")
# dataset_names = ("Microtubules2-5", "Microtubules2-9")
# dataset_names = ("Microtubules2-4", "Microtubules2-9")
# dataset_names = ("Microtubules2-3", "Microtubules2-9")
# dataset_names = ("Microtubules2-2", "Microtubules2-9")
# dataset_names = ("Microtubules2-1", "Microtubules2-9")
# dataset_names = ("CCPs-9", "Microtubules2-9")
# dataset_names = ("CCPs-9", "F-actin-nonlinear-9")
# dataset_names = ("F-actin-9", "Microtubules2-9")
# dataset_names = ("F-actin-9", "F-actin-nonlinear-9")
# dataset_names = ("ER-6", "Microtubules2-9")
dataset_names = ("ER-6", "F-actin-nonlinear-9")


dataset_name_test, dataset_name_train = dataset_names

# ------------------------------------------------------------------------------
path_prediction = os.path.join(
    "outputs", "predictions", dataset_name_test, "kernelnet", dataset_name_train
)

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

# ------------------------------------------------------------------------------
info_xlsx = pandas.read_excel("datasets_test.xlsx")
info = info_xlsx[info_xlsx["id"] == dataset_name_test].iloc[0]

params_dict = dict(
    kernel_size_fp=text2tuple(info["ks_fp"]),
    kernel_size_bp=text2tuple(info["ks_bp"]),
    dim=int(info["ndim"]),
    ratio=float(info["ratio"]),
    eps=0.000001,
    scale_factor=int(info["scale_factor"]),
    interpolation=True,
    kernel_norm_fp=False,
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
    num_iter_test=2,
)

# ------------------------------------------------------------------------------
device = torch.device(id_device)
suffix_net = "_ss" if params_dict["train_mode"] == "ss" else ""
params_dict["conv_mode"] = "direct" if params_dict["dim"] == 2 else "fft"

print("-" * 80)
print(f"[INFO] Dataset: {dataset_name_test}")
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

# ------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------
FP, BP = None, None
# Forward Projection
print("-" * 80)
# ------------------------------------------------------------------------------
if FP_type == "pre-trained":
    FP_path = win2linux(
        checkpoints_list[dataset_name_train]["forward"][
            f"n{num_data_fp}_r{id_repeat_fp}"
        ]
    )

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
    FP = lambda x: torch.nn.functional.avg_pool3d(
        conv_fp(x),
        kernel_size=params_dict["scale_factor"],
        stride=params_dict["scale_factor"],
    )

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
    model_path = win2linux(
        checkpoints_list[dataset_name_train]["backward"][
            f"n{num_data_bp}_r{id_repeat_bp}"
        ]
    )

    assert os.path.exists(model_path), f"[ERROR] model_path not found: {model_path}"
    print("[INFO] Use learned BP kernel.")
    print(f"[INFO] Load model from : {model_path}")

    model.load_state_dict(
        torch.load(model_path, map_location=device)["model_state_dict"], strict=False
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
path_save_kernel = os.path.join(path_prediction, "kernel")
save_kernel = lambda fname, arr: io.imsave(
    fname=os.path.join(path_save_kernel, fname), arr=arr, check_contrast=False
)

os.makedirs(path_save_kernel, exist_ok=True)
print("[INFO] save kernels to:", path_save_kernel)

ker_fp_init = padding_kernel(ker_fp_init, PSF_true)
ker_FP = padding_kernel(ker_FP, PSF_true)
ker_BP = padding_kernel(ker_BP, PSF_true)

save_kernel("kernel_true.tif", PSF_true)
save_kernel("kernel_init.tif", ker_fp_init)
save_kernel("kernel_fp.tif", ker_FP)
save_kernel(f"kernel_bp{suffix_net}.tif", ker_BP)

# ------------------------------------------------------------------------------
#                                   Prediction
# ------------------------------------------------------------------------------
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
for i in id_sample:
    if i >= dataset_test.__len__():
        print(f"[ERROR] Sample {i} is out of range, exit.")
        break

    data = dataset_test[i]  # load one sample

    x = torch.unsqueeze(data["lr"], 0).to(device)
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
    y_pred_all = model(x).cpu().detach().numpy()[:, 0, 0]
    y, x = t2n(y), t2n(x)
    pbar.update(1)

    # Save results -------------------------------------------------------------
    path_sample = os.path.join(path_prediction, f"sample_{i}")
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
