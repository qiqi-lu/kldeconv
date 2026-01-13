# Kernel Learning Deconvolution (KLDeconv)
KLDeconv is an algorithm for fluorescence microscopic image deconvolution.
It enhances the deconvolution performance and speed through learning the forward kernel and backward kernel in conventional Richardson-Lucy Deconvolution (RLD) algorithm.

KLDeconv only requires one training sample and two iterations to achieve superior deconvolution perfromance and speed compared to traditional RLD and its variants (which using an unmatched backeard kernel, such as Gaussian, Butterworth, and Wiener-Butterworth (WB) backward kernels).

This repository includes:
- `MATLAB` implementation of training data simulation
- `Python` implementation of training and inference of KLDeconv
- `Python` implementation of conventional Richardson-Lucy Deconvolution (RLD) with different backward kernels, includeing traditional, Gaussian, Butterworth, Wiener-Butterworth (WB).

We have developed a `napari` plugin for KLDeconv, named `napari-kld`.

## napari-kld plugin
`mapari-kld` is a `napari` plugin, named `napari-kld`, was developed for KLDeconv.

The source code is saved at https://github.com/qiqi-lu/napari-kld, please go to this repository for more information. And it is also accessable through `napari hub` at https://www.napari-hub.org/plugins/napari-kld.

### Installation

You should install `napari` firstly and then install `napari-kld`.

#### **Install `napari`**

You can download the `napari` bundled app for a simple installation via https://napari.org/stable/tutorials/fundamentals/quick_start.html#installation.

Or, you can install `napari` with Python using pip:

```
conda create -y -n napari-env -c conda-forge python=3.10
conda activate napari-env
python -m pip install 'napari[all]'
```

Refer to https://napari.org/stable/tutorials/fundamentals/quick_start.html#installation.

#### **Install `napari-kld`**

You can install napari-kld plugin with napari:

Plugins > Install/Uninstall Plugins… > [input napari-kld] > install

You can install `napari-kld` via [pip](https://pypi.org/project/pip/):

```
pip install napari-kld
```

## File structure
- `./checkpoints`: include the saved models. (ignored in `git`)
- `./methods`: code for conventional RLD methods and `MATLAB` codes for phantom generation.
- `./models`: code for different model architectures.
- `./others`: the backup of the codes.
- `./outputs`: saves the output results. (ignored in `git`)
- `./utils`: includes the functions used to processing data, quantitatively evaluation, plot images.

## Requirements
We run our codes on `Windows 11` (optional). 
The version of `Python` is 3.11.9, which must higher then 3.7.

The python package used in our projects:
- torch==2.0
- torchvision
- tensorboard
- numpy
- matplotlib
- scikit-image
- pydicom
- pytorch-msssim
- fft-conv-pytorch

To use our code, you shold create a virtual enviroment and install the required packages first:

```
$ conda create -n pytorch python=3.11.9 
$ conda activate pytorch
$ pip install -r requirements.txt
```

**Please always pay attention to the path setting in the code, and modifiy the to your own working path.**
**Specific parameters can be modified according to you needs.**

## Datasets
### Simulation datasets
We use `MATLAB` code in [Richardson-Lucy-Net](https://github.com/MeatyPlus/Richardson-Lucy-Net/tree/main/Phantom_generate) to generate simulated phantoms with bead structures or mixed structures. The modified codes are save in `methods\phantom_generate` folder.

- `generate_synthetic_data.py`: generate the simulated datasets with different Poisson/Gaussin noise level.

### Real datasets
For the images in BioSR and Confocal/STED volume data set, the images should be preprocessing before training. 
We use `real_data_preprocessing.py` to preprocess the real biology images in Confocal/STED volume data set.
The preprocessing file `preprocess.py` save in `data/BioSR` is used to proprocess the data of BioSR. 

## Model training
We use `main_kernelnet.py` to train a new model. (please modify the `root_path` to the path saved data sets) The training parameters can be modified directly in the source code.

The model weights will be saved in `./checkpoints` folder.

## Model evaluation
To test a well-trained model, we use `evaluate_model.py`, the output results will be save in `./outputs/figures`.

Some pre-trained models are provides in `./checkpoints`.

## Other files
- `deconv3D_w_gt.py` is used to deconv the 3D images in `simulation` data set, using conventional RLD methods.

- `deconv3D_live.py` is used to deconv the 3D volumes in `LLSM volume` data set using conventional RLD methods. The parameter `id_sample` and `wave_length` should be modified according to your data directory. Please enable specific method to do deconvolution.

- `deconv2D_real.py` is used to deconv the real 2D biological images in `BioSR` data set using conventional RLD methods. As there is no PSF is provided, the PSF are learned form the paired data, and then used in this file. You should use `main_kernelnet.py` to train the model, and use `evaluate_model.py` to generate the learned PSF, and then use it to do deconvolution.

- `deconv3D_real.py` is used to deconv the real 3D biological images in `Confocal/STED volume` data set using conventional RLD methods. As there is no PSF is provided, the PSF are learned form the paired data, and then used in this file. You should use `main_kernelnet.py` to train the model, and use `evaluate_model.py` to generate the learned PSF, and then use it to do deconvolution.

## Additional informations
### open access datasets
The BioSR data set is publicly accessible at https://doi.org/10.6084/m9.figshare.13264793.v9. The confocal/STED volume data set is publicly available at https://zenodo.org/record/4624364#.YF3jsa9Kibg. The LLSM volume data set is publicly accessible at https://zenodo.org/records/7261163.

### open access code
The `MATLAB` codes for generation of simulation phantom are publicly accessible at GitHub repository (https://github.com/MeatyPlus/Richardson-Lucy-Net).

## Acknowledgements
We thank Yue Li, et al for publicly releasing the code for phantom simulation, Chang Qiao, et al. for publicly releasing the BioSR data set and LLSM volume data set, Jiji Chen, et al. for publicly releasing the Confocal/STED volume data set, which were significantly contributed to the advancement of our study.