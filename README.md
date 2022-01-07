# Automated FPW Measurements using DL
Developed by Najafian Lab.

# Overview
This repository contains the trained model, vision scripts, and relevant code to process foot process width measurements on electron microscopy images. The model was trained on roughly 1000x1000 square images taken at 50kX that were downscaled to 640x640 for the model. Please ensure good contrast and quality of datasets if intended for use. Please read more at the paper here **LINK** 

# Installation
## System Recommendations
For best/quick results we recommend a GPU that's equivalent to or better than a GTX 1080 Ti, and a modern PC. Just as a reference we used an AMD Ryzen 3700X with a Titan RTX for training with 64Gb of system ram and 24Gb of VRAM. Although other configurations will work, we cannot ensure inference time will be fast or if the model will even load with systems that have low GPU RAM or system RAM. We wouldn't even consider running tensorflow on a CPU with this model.  

### Anaconda and Command Line
The installation does require a recent installation of anaconda. Please visit [anacondas page](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html) for more info. Furthermore, installation and usage of this repo requires knowledge of command line. This project was developed on/ran in linux, but should be able to run on any system.

### Clone the repo
```bash
git clone <insert URL>
cd <path>
```

### Create the anaconda environment and download dependencies
This will create a new anaconda environment called `fpw-dl` and download all of the dependencies. NOTE: this package requires usage of Keras 2.2.X and Tensorflow 1.15 (CUDA 10.0, CUDNN 7.6). The model and weights were trained before v2. So other versions of TF will not work. If other dependencies are not found for your distribution, then please modify the `environment.yml` dependencies.
```bash
conda env create --file environment.yml
```

### Activate the environment
```bash
conda activate fpw-dl
```

## Downloading the Dataset
The dataset that was used to evaluate the model, as presented in the paper, is too large for github. So a public google drive link [INSERT LINK HERE] has been provided and an easy to use download script is in the repository. This download script will download the `dataset.zip` from google drive and extract it in the repository directory. This dataset will consume **~34Gb** of disk space. 
```bash
python download.py  # please be patient as this might take a while
```

## Analyzing FPW on two datasets
The two datasets analyzed in the paper were a random assortment of TEM images of diseased (fabry) and normal patients. Please see the [Overview](#Overview) for a description of the images. Use the `analysis.py` script to process the images. <br>
**Note:** The prediction `.tiff` masks will be produced in the *prediction/* folders of the input dataset. So once this script is complete check out the `dataset/fabry/prediction` and `dataset/normal/prediction` once the processing is complete.

```bash
# "dataset/fabry" and "dataset/normal" are the folders to process 
# "--bulk" flag indicates multiple-subfolders to process 
# "-bs 12" flag indicates a batch size of 12 (CHANGE THIS)
python analysis.py dataset/fabry --bulk --use_file_average -bs 12  # process fabry
python analysis.py dataset/normal --bulk --use_file_average -bs 12  # process normal
```
There are certain flags that might be useful, if you've already processed the dataset once. For example, `--pskip` will skip prediction and go straight to vision procesisng. Please see `python analysis.py --help` for more options.



## Issues
If there are any issues with the repository or process, please message <smerkd@uw.edu> and/or <najafian@uw.edu>.