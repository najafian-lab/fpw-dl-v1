# Automated FPW Measurements using DL
Developed by David Smerkous at Najafian Lab.

# Overview
This repository contains the trained model, vision scripts, and relevant code to automate foot process width estimation on electron microscopy images. The model was trained on ~1024x1024 segmented EM images. The sample EM images were taken where the scaling was about 10nm/pixel, at ~30,000X magnification. The images were then downscaled to 640x640 for the custom ForkNet model. Please ensure good contrast and quality of datasets if intended for use. Please read more at the paper here **[LINK INSERT LATER]** 

## Image Examples
Here are a few examples of EM images with varying quality.
| [![Good Quality Image](images/good.png)](images/good.png)  | [![Bad Quality Image](images/bad.png)](images/bad.png) |
|:---:|:---:|
| Higher quality image with slits/membrane edges that are well defined | Lower quality image with harder to identify slits and poor membrane edge contrast |

| [![Good Quality Image](images/good2.png)](images/good2.png)  | [![Okay Quality Image](images/okay.png)](images/okay.png) |
|:---:|:---:|
| Okay quality image with good slit definition and good membrane/podocyte contrast | Lower quality image that contains bad contrast and artifacts |

The easier it is for you to identify the components and slits in the image, the better the results of the ML model/vision scripts will be. As described in the paper, please use images that have good contrast between cells and membranes, with fewer artificats, and are of similar magnification that the model was trained on.

# Installation
## System Recommendations
For faster results we recommend an NVIDIA GPU that's equivalent to or better than a GTX 1070, and a modern PC. Just as a reference we used an AMD Ryzen 3900X with a Titan RTX, to generate the FPW estimates for the paper, with 64Gb of system ram and 24Gb of VRAM. Although other configurations will work, we cannot ensure inference time will be fast or if the model will even load with systems that have low GPU RAM or system RAM. Although you could run this model on CPU, we wouldn't recommend it for larger datasets.

### Anaconda and Command Line
The installation does require a recent installation of anaconda. Please visit [anacondas page](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html) for more info. Furthermore, installation and usage of this repo requires basic knowledge of command line. This project was developed in linux, but should be able to run on any system.

### Clone the repo
```bash
git clone https://github.com/najafian-lab/fpw-dl-v1.git
cd fpw-dl-v1
```

### Create the anaconda environment and download dependencies
```bash
conda env create --file environment.yml
```
This will create a new anaconda environment called `fpw-dl` and download all of the dependencies. Also, this package requires usage of Keras 2.2.4 and Tensorflow 1.15 (GPU version requires CUDA 10.0, CUDNN 7.6). The model and weights that we used for this paper were trained before v2. So other versions of TF will not work. If some of the dependencies are not found for your distribution, then please modify the `environment.yml` dependencies.

*Note for **Windows** users: if you're having issues activating on command prompt/powershell and aren't familiar with Anaconda, I recommend just using the Anaconda Prompt (search it in the windows search bar)*

### Activate the environment
```bash
conda activate fpw-dl
```

## Downloading the Dataset
The dataset that was used to evaluate the model, as presented in the paper, is too large for github. So a public [google drive link](https://drive.google.com/file/d/1bAQLG-5c1JxkwHm8ttPqh-I7JjfSEJYG/view?usp=sharing) has been provided, and an easy to use download script is in this repo. The download script will download the zip file from google drive and extract it in the repository directory. This dataset will consume **~34Gb** of disk space. 
```bash
python download.py  # please be patient as this might take a while
```

## Analyzing FPW on two datasets
The two datasets analyzed in the paper were a random assortment of TEM images of mostly fabry and some normal patients. Please see the [Overview](#Overview) for a description of the images. Use the `analysis.py` script to produces the segmentation masks and process the images.
```bash
# "dataset/fabry" and "dataset/normal" are the folders to process 
# "--bulk" flag indicates multiple-subfolders to process 
# "-bs 12" flag indicates a batch size of 12 (CHANGE THIS)
python analysis.py dataset/fabry --bulk --use_file_average -bs 12  # process fabry
python analysis.py dataset/normal --bulk --use_file_average -bs 12  # process normal
```
There are certain flags that might be useful, if you've already processed the dataset once. For example, `--pskip` will skip prediction and go straight to vision processing. Please see `python analysis.py --help` for more options.


## Viewing the results
### Segmentation Masks
In the `--bulk` report each folder inside the input folder is treated as a single biopsy, and predicted masks are outputed to each biopsy folder. For example, `dataset/fabry/15-0079/prediction` will contain all of the `.tiff` layered segmentation masks for that biopsy.

### Vision results
The `analysis.py` script will produce the layered segmentation masks and also process the output of the masks. The script will process the membrane edge and slits to generate an estimate of the FPW. All of the processed FPW estimate results are going to be located in the `dataset/fabry/prediction` and `dataset/normal/prediction` folders in a spreadsheet called `bulk_report.xlsx`, only if the bulk flag - as described below, - is specified. Finally, more descriptive file by file measurements are in each of the individual biopsy folders, such as `dataset/fabry/15-0079/prediction`, where there will be a `report.xlsx`. 

### Bulk flag
The `--bulk` flag will indicate each folder inside the input folder, like `dataset/fabry`, is going to be a biopsy. Without this flag the analysis script will assume the input folder is a single biopsy. This flag will also generate a `bulk_report.xlsx` inside the prediction folder, which will show the global results for each biopsy.

### Preview flag
Using the `--preview` flag when processing with `analysis.py` will show previews of the segmentation masks and vision processing results. Please see the `ilog` function in `analysis.py` if you wish to save the results to a file location.

Here are a few examples of the previews generated
| ![Okay prediction image (images/okay_prediction.png)](images/okay_prediction.png) | ![Bad prediction image (images/bad_prediction.png)](images/bad_prediction.png) |
|:---:|:---:|
| Example of a good predicition. Example on higher quality image | Example of a bad prediction. Some could be even worse |

|![Post-processing result 1 (images/one_post_process.png)](images/one_post_process.png)|![Post-processing result 2 (images/two_post_process.png)](images/two_post_process.png)|
|:---:|:---:|
| Post-process results of random image with more membrane attachment | Post-process results of random image with less membrane attachment. |

The worse example shows the sensitivity of some ML models. This can be due to varying quality of images and the size of our training set. We're training a more complicated and better model, with a larger dataset, to better handle various images, but for best results on this model please consider pre-screening/post-screening your images.

*Note: all measurements shown in the post-processing are measured along the membrane edge and not directly. Also, you can change the individual colors of the measurements, to distinctly tell the difference, from red if you look at the end of `process_slits` function in `slits.py`*

## Evaluating the model
The evaluation dataset inside `dataset/eval` contains two folders called `images` and `masks`. The `images` folder are randomly selected EM images, with their respective ground truth masks in the `masks` folder. Some images were taken from the training masks and some from the validation set. To run the evaluation please use the following commands
```bash
# first the predicted masks need to be produced
# "--vskip" flag to skip vision processing and only generate masks
# "-bs 12" batch size of images to process (CHANGE THIS)
python analysis.py dataset/eval/images --vskip -bs 12

# generate the dice scores based off of the ground truth and the predicted masks
python eval.py
```
This will generate the dice scores on all of the input images and group them together by biopsy using their two number prefix. The results will be produced in the `dices.xlsx` file. You should get the same aggregate numbers to the ones below.

|Biopsy Prefix|Membrane Average|Membrane STD|Slit Average|Slit STD|
|:---:|:---:|:---:|:---:|:---:|
|08|0.87|0.04|0.54|0.07|
|09|0.83|0.05|0.51|0.06|
|10|0.47|0.23|0.30|0.17|
|12|0.82|0.07|0.52|0.13|
|13|0.79|0.24|0.58|0.18|
|14|0.85|0.19|0.58|0.15|
|16|0.90|0.02|0.59|0.03|

*Note: look at biopsy prefix 10 and the input images. A lot of images are of low quality compared to biopsy prefix 16*


## Generating Figures
After analyzing a dataset you may use the `figure.py` script to generate some figures using matplotlib. For example, use the following command to generate the running average graphs after running the `analysis.py` on the datasets.
```bash
# note: the running_average_num is a MAX biopsy count. Does not guarantee that biopsy count

# figure for fabry dataset
python figure.py --running_average --running_average_file dataset/fabry/prediction/running_average_individual.json --running_average_num 20 --running_average_offset 0 --running_average_title "Running average of Fabry samples" --running_average_use_overall_average --running_average_show_convergence

# figure for normal dataset
python figure.py --running_average --running_average_file dataset/normal/prediction/running_average_individual.json --running_average_num 20 --running_average_offset 0 --running_average_title "Running average of normal samples" --running_average_use_overall_average --running_average_show_convergence
```
That commands will produce something similar to the images below in the file `running_average.png`. This graph is particularly useful for noise analysis and understanding, roughly, how many images need to be sampled for FPW averages to converge per biopsy.
|![Running average of normal biopsies (images/running_average_normal.png)](images/running_average_normal.png)|![Running average of fabry biopsies (images/running_average_fabry.png)](images/running_average_fabry.png)|
Please use `python figure.py --help` for more figure options.

## Note
A cross platform version of the Report Utility has not been well tested and has some bugs. We're working on fixing them and publishing a web version of the utility. So currently all processing must be done through the CLI.

## Issues
If there are any issues with the repository or process, please message David Smerkous at smerkd@uw.edu, and/or Dr. Behzad Najafian at najafian@uw.edu.