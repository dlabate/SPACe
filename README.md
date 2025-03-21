# SPACe: an open-source, single cell analysis of Cell Painting data

[![DOI](https://zenodo.org/badge/760841105.svg)](https://zenodo.org/doi/10.5281/zenodo.13821483)

## **Purpose and Use Case**

SPACe (Swift Phenotypic Analysis of Cells) is an image analysis package designed for high throughput phenotypic high throughput microscopy (96 and 384 well plates) based on the JUMP consortium Cell Painting (CP) protocol. The input usually is 5 fluorescence channels (C1 / C2 / C3 / C4 / C5) corresponding to specific CP fluorescent dies (DAPI / Concanavalin A / SYTO14 / WGA+phalloidin / Mitotracker). 
The package contains a user-friendly and tunable interface for cellular segmentation (check the preview.ipynb notebook).  It has two GPU-back ended options for the segmentation of the nucleus and cell: [cellpose](https://github.com/MouseLand/cellpose) and [pycleranto](https://github.com/clEsperanto/pyclesperanto_prototype).  
It uses a novel method to match the segmentation of nucleus and cytoplasm, then uses the those two segmentation masks to segment the nucleoli and mitochondira as well.

SPACe is fast, ~10X faster than [Cellprofiler](https://github.com/CellProfiler/CellProfiler), using a reasonably standard desktop and not using any cloud computing resources. It takes about 6-9 hours to analyze a full 384-well plate (~17000 2000X2000 pixel images) via pytorch/GPU as well as CPU-Multiprocessing for speedup.  The output is based on single cell data and is provided as canonical well-based summary statistics (mean, median) and as earth mover’s distance measurements of each well to a DMSO control reference distribution.  

More information can be found in the [paper](https://www.nature.com/articles/s41467-024-54264-4).

## **License**
This project is licensed under the [MIT License](https://github.com/dlabate/SPACe?tab=MIT-1-ov-file).

## **Citation**
If you use **SPACe**, please cite the following publication: 

Stossi, F., Singh, P.K., Marini, M., Safari, K., Szafran, A.T., Rivera Tostado, A., Candler, C.D., Mancini, M.G., Mosa, E.A., Bolt, M.J. and Labate, D., 2024. SPACe: an open-source, single-cell analysis of Cell Painting data. Nature Communications, 15(1), p.10170. https://doi.org/10.1038/s41467-024-54264-4


## **Image Analysis Steps**

![Image Description](https://github.com/dlabate/SPACe/raw/main/figures/image%20analysis%20steps.png)

1) Preview (Check and decide how happy you are with your segmentation on a few wells!)
2) Segmentation Step 1 (Segmenting nucleus and cell)
3) Segmentation Step 2 (Matching nucleus and cell segmentation as well as segmenting nucleoli and mitchondria)
4) Light-weight Feature extraction: Shape, Intensity, and Texture Features
5) Calcultes the Wassertein-Distance Map of each biological-well from the DMSO/Vehicle condition.

## **Installation**

### System requirements
Linux, Windows and Mac OS are supported for running the code. At least 16GB of RAM is required to run the software. The software has been heavily tested on Windows 11 and Ubuntu 18.04 and less well-tested on Mac OS. Please open an issue if you have problems with installation.

### Dependencies 
SPACe relies on the following excellent packages (which are automatically installed with conda/pip if missing):

[cellpose](https://github.com/MouseLand/cellpose)==2.2 <br>
[torch](https://github.com/pytorch/pytorch)>=1.6 <br>
[torchvision](https://github.com/pytorch/vision) <br>
[pycleranto](https://github.com/clEsperanto/pyclesperanto_prototype) <br>
[sympy](https://github.com/sympy/sympy) <br>
[tifffile](https://github.com/cgohlke/tifffile) <br>
[numpy](https://github.com/numpy/numpy)>=1.20.0 <br>
[scipy](https://github.com/scipy/scipy) <br>
[scikit-image](https://github.com/scikit-image/scikit-image)>=0.20.0 <br>
[scikit-learn](https://github.com/scikit-learn/scikit-learn) <br>
[SimpleITK](https://github.com/SimpleITK/SimpleITK) <br>
[pandas](https://github.com/pandas-dev/pandas)>=2.0.0 <br>
[xlsxwriter](https://github.com/jmcnamara/XlsxWriter) <br>
[openpyxl](https://github.com/theorchard/openpyxl) <br>
[xlrd](https://github.com/python-excel/xlrd) <br>
[jupyter](https://github.com/jupyter/notebook) <br>
[matplotlib](https://github.com/matplotlib/matplotlib) <br>
[plotly](https://github.com/plotly) <br>
[pathlib](https://github.com/budlight/pathlib) <br>
[tqdm](https://github.com/tqdm/tqdm) <br>
[pyefd](https://github.com/hbldh/pyefd) <br>

### Instructions

If you do not have anaconda installed on your computer, the first step is to install anaconda3 as follows:  
  1.	Download Anaconda: Go to the Anaconda website [Anaconda](https://www.anaconda.com/download) and download the Anaconda3 installer for your operating system (Windows, macOS, or Linux).
  2.	Run Installer: Once the installer is downloaded, run it and follow the installation instructions.
  3.	Agree to Terms: Read and agree to the license agreement.
  4.	Choose Install Location: Choose a directory where Anaconda will be installed. The default location is usually fine.
  5.	Select Installation Type: Choose whether to install Anaconda just for you or for all users on the system.
  6.	Advanced Options (Optional): You may have the option to add Anaconda to your system PATH environment variable, which can make it easier to use from the command line. This is typically recommended.
  7.	Install: Click the "Install" button to begin the installation process.
  8.	Complete Installation: Once the installation is complete, you may be prompted to install Visual Studio Code (VS Code) or PyCharm. You can choose to install them or skip this step if you prefer.
  9.	Verify Installation: Open a terminal or command prompt and type ```conda --version``` to verify that Anaconda has been installed correctly. You should see the version number of Anaconda displayed.

To install SPACe python package on a conda virtualenv called tensors:

1.	Install anaconda3/miniconda3 on your windows or linux machine (see instruction above to download anaconda)
2.	Open an anaconda3 terminal (click on the search bar on your desktop and type anaconda. The Anaconda prompt will pop up):
3.	Copy and paste the following instruction: 
    1.	``` conda create --name tensors python=3.10 --no-default-packages ```
    2.	``` conda activate tensors ```
    3.	``` python -m pip install cellpose --upgrade ```
4.	NOTE: Only if you have a dedicated Nvidia GPU available do the following:
  ```
  pip uninstall torch
  conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```
5.
     **Option 1)** (< 3 minutes)
    To install SPACe directly from github:
    ``` pip install git+https://github.com/dlabate/SPACe.git ```
  
    **Option 2)** (< 3 minutes)
    To install SPACe locally on your computer and in order to be able to edit it, first download it from github to your computer, go to one directory above where your SPACe folder is
    downloaded/located/saved, through an anaconda3 terminal, the structure of the directory would look like this:
    ```
    dir/
     SPACe/
          setup.py
          README.md
          SPACe
    ```
    type ```pip install -e SPACe``` in the same terminal. The ```-e``` allows
   	one to edit the program.
  
All the required packages will be installed automatically from ``` setup.py ``` file.
If you are going to use Pycharm, its terminal might not recognize your anaconda3 virtualenv. 
Here is the fix from 
``` https://stackoverflow.com/questions/48924787/pycharm-terminal-doesnt-activate-conda-environment ```.
If the pycharm terminal does not recognize your anaconda virtualenv, do the following:

Go to ``` File -> Settings -> Tools -> Terminal ```. Replace the value in ``` Shell path ``` with ``` cmd.exe "/K" path_to_your_miniconda3\Scripts\activate.bat tensors ```.

Remember, you will be able to modify the content of this package only if you install it via Option 2).

## **Reproducibility test for JUMP-MOA (BR00115125-31)**
To test the pipeline on the JUMP Consortium datasets mentioned in the paper, follow these instructions:
1. Download the datasets from the following link: [BR00115125-31](https://cellpainting-gallery.s3.amazonaws.com/index.html#cpg0001-cellpainting-protocol/source_4/images/2020_08_11_Stain3_Yokogawa/images/)
2. Place the platemap (already filled out and available [here](SPACe/Images_example/BR00115126/Jump_Consortium_Datasets_cpgmoa_AssayPlate/platemap.xlsx) in the same directory as your experiment_path (image folder). Please note that we have already set the best hyperparameters for running the pipeline on the JUMP-MOA dataset. Rename the image folder as follows: Jump_Consortium_Datasets_cpgmoa_AssayPlate.
3. Follow the installation instructions in this [README.md](https://github.com/dlabate/SPACe/blob/master/README.md)
4. Follow the instructions on how to run the program here: [run_SPACe.md](https://github.com/dlabate/SPACe/blob/master/run_SPACe.md)
5. The expected output consists of four folders: Step2_MaskP1, Step3_MaskP2, Step4_Features, and Step5_DistanceMaps. An example of the desired output for BR00115126 can be downloaded from the following link: [SPACe_Results_BR00115126](https://s3.console.aws.amazon.com/s3/upload/space-results).
   
Runtime for BR00115126 with GPU: Tesla V100-PCIE-16GB, CPU: Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz, and RAM: 251GB. The program finished analyzing experiment BR00115126 in 6.81 hours.

## **Running SPACe**

To learn how to use the program, go to [run_SPACe.md](https://github.com/dlabate/SPACe/blob/master/run_SPACe.md).




