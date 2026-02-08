"""
Check the shape of image in the datasets.
"""

import os, pandas
from utils.data import read_txt, win2linux
import skimage.io as io

dataset_id = "biotisr-3d-factin-1"
dataset_id = "Microtubule2-3d-1024"
# dataset_id = "Nuclear-pore-complex2-1024"

# ------------------------------------------------------------------------------
path_excel = "datasets_test.xlsx"

# read the excel file
df_info = pandas.read_excel(path_excel)
info = df_info[df_info["id"] == dataset_id].iloc[0]

path_lr = win2linux(info["path_lr"])
path_txt = win2linux(info["path_txt"])

print("-" * 80)
print(f"[INFO] {dataset_id}")
print(f"[INFO] path_lr = {path_lr}")
print(f"[INFO] path_txt = {path_txt}")

filenames = read_txt(path_txt)
for filename in filenames:
    path_image = os.path.join(path_lr, filename)
    img = io.imread(path_image)
    print(f"[INFO] {filename}: shape = {img.shape}")
