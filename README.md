# Automated FPW Measurements using DL
Developed by Najafian Lab.

# Overview
This repository contains the trained model, vision scripts, and relevant code to process foot process width measurements on electron microscopy images. The model was trained on roughly 1000x1000 square images taken where the scaling is about 10nm/pixel (~30,000X magnification), that were downscaled to 640x640 for the ForkNet model. Please ensure good contrast and quality of datasets if intended for use. Please read more at the paper here **[LINK INSERT]** 

## Image Examples
Here are examples of EM images with varying quality.
| [![Good Quality Image](images/good.png)](images/good.png)  | [![Bad Quality Image](images/bad.png)](images/bad.png) |
|:---:|:---:|
| Higher quality image with slits/membrane edges that are well defined | Very low quality image with slightly harder to identify slits and poor membrane edge definition |

| [![Good Quality Image](images/good2.png)](images/good2.png)  | [![Okay Quality Image](images/okay.png)](images/okay.png) |
|:---:|:---:|
| Decent quality image with good slit definition and good membrane/podocyte contrast | Lower quality image that has bad contrast, artifacts, but okay slit definition |

A general rule is, the easier it is for you to identify the components of the image, the better the results. As described in the paper, please use images that have good contrast between cells and membranes, with fewer artificats, and are of similar magnification.

# Installation
## System Recommendations
For best/quick results we recommend a GPU that's equivalent to or better than a GTX 1080 Ti, and a modern PC. Just as a reference we used an AMD Ryzen 3900X with a Titan RTX, to generate the results, with 64Gb of system ram and 24Gb of VRAM. Although other configurations will work, we cannot ensure inference time will be fast or if the model will even load with systems that have low GPU RAM or system RAM. Although you could run this model on CPU, we wouldn't recommend it.

### Anaconda and Command Line
The installation does require a recent installation of anaconda. Please visit [anacondas page](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html) for more info. Furthermore, installation and usage of this repo requires knowledge of command line. This project was developed on/ran in linux, but should be able to run on any system.

### Clone the repo
```bash
git clone https://github.com/najafian-lab/fpw-dl-v1.git
cd fpw-dl-v1
```

### Create the anaconda environment and download dependencies
This will create a new anaconda environment called `fpw-dl` and download all of the dependencies. NOTE: this package requires usage of Keras 2.2.4 and Tensorflow 1.15 (CUDA 10.0, CUDNN 7.6). The model and weights were trained before v2. So other versions of TF will not work. If other dependencies are not found for your distribution, then please modify the `environment.yml` dependencies.
```bash
conda env create --file environment.yml
```
*Note for **Windows** users: if you're having issues activating on command prompt/powershell and aren't familiar with Anaconda, I recommend just using the Anaconda Prompt (search it in the windows search bar)*

### Activate the environment
```bash
conda activate fpw-dl
```

## Downloading the Dataset
The dataset that was used to evaluate the model, as presented in the paper, is too large for github. So a public [google drive link](https://drive.google.com/file/d/1bAQLG-5c1JxkwHm8ttPqh-I7JjfSEJYG/view?usp=sharing) has been provided and an easy to use download script is in the repository. This download script will download the `dataset.zip` from google drive and extract it in the repository directory. This dataset will consume **~34Gb** of disk space. 
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


## Generating Figures
After analyzing a dataset you may use the `figure.py` script to generate some figures using matplotlib. For example, use the following command to generate the running average graphs after running the `analysis.py` on the datasets.
```bash
# note: the running_average_num is a MAX biopsy count. Does not guarantee that biopsy count

# figure for fabry dataset
python figure.py --running_average --running_average_file dataset/fabry/prediction/running_average_individual.json --running_average_num 20 --running_average_offset 0 --running_average_title "Running average of fabry samples" --running_average_use_overall_average --running_average_show_convergence

# figure for normal dataset
python figure.py --running_average --running_average_file dataset/normal/prediction/running_average_individual.json --running_average_num 20 --running_average_offset 0 --running_average_title "Running average of normal samples" --running_average_use_overall_average --running_average_show_convergence
```
That command will produce something similar to. This graph is particularly useful for noise analysis and understanding, roughly, how many images need to be sampled per biopsy for the FPW averages to converge.
|![Running average of normal biopsies (images/running_average_normal.png)](images/running_average_normal.png)|![Running average of fabry biopsies (images/running_average_fabry.png)](images/running_average_fabry.png)|
Please use `python figure.py --help` for more options.


## Issues
If there are any issues with the repository or process, please message David Smerkous at smerkd@uw.edu, and/or Dr. Behzad Najafian at najafian@uw.edu.