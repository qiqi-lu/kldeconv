"""
RLN only support even slice, remove one of the odd slice.
"""

import os, tqdm
import skimage.io as io
from utils.data import win2linux

path_list = (
    "E:\qiqilu\datasets_2\BioTISR\\transformed\Mitochondria-3D\SIM_remove_last_t0",
    "E:\qiqilu\datasets_2\BioTISR\\transformed\Mitochondria-3D\WF_noise_level_1_remove_last_t0",
    "E:\qiqilu\datasets_2\BioTISR\\transformed\Mitochondria-3D\WF_noise_level_2_remove_last_t0",
    "E:\qiqilu\datasets_2\BioTISR\\transformed\Microtubules-3D\WF_noise_level_1_remove_last_t0",
    "E:\qiqilu\datasets_2\BioTISR\\transformed\Microtubules-3D\WF_noise_level_2_remove_last_t0",
    "E:\qiqilu\datasets_2\BioTISR\\transformed\Microtubules-3D\SIM_remove_last_t0",
    "E:\qiqilu\datasets_2\BioTISR\\transformed\F-actin-3D\WF_noise_level_2_remove_last_t0",
    "E:\qiqilu\datasets_2\BioTISR\\transformed\F-actin-3D\SIM_remove_last_t0",
    "E:\qiqilu\datasets_2\BioTISR\\transformed\F-actin-3D\WF_noise_level_1_remove_last_t0",
)

for path in path_list:
    path_images = win2linux(path)
    # get all the image names end with .tif
    image_names = [f for f in os.listdir(path_images) if f.endswith(".tif")]
    num_images = len(image_names)

    print("-" * 80)
    print(f"[INFO] Path : {path_images}")
    print(f"[INFO] Number of images: {num_images}")

    path_save = path_images + "_even_slice"
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    print(f"[INFO] Saving to {path_save}")

    # ------------------------------------------------------------------------------
    pbar = tqdm.tqdm(total=num_images, desc="Odd to even slice", ncols=80)
    for image_name in image_names:
        # read the image
        image = io.imread(os.path.join(path_images, image_name))
        shape = image.shape
        # if the slice number is odd, remove the last slice
        if shape[0] % 2 != 0:
            image = image[:-1, :, :]
        # save the image
        io.imsave(os.path.join(path_save, image_name), image, check_contrast=False)
        pbar.update(1)
    pbar.close()
