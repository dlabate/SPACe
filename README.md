# SPACe (Swift Phenotypic Analysis of Cells): an open-source, single cell analysis of Cell Painting data

## **Purpose and Use Case**


SPACe is an image analysis package designed for high throughput phenotypic high throughput microscopy (96 and 384 well plates) based on the JUMP consortium Cell Painting (CP) protocol. The input usually is 5 fluorescence channels (C1 / C2 / C3 / C4 / C5) corresponding to specific CP fluorescent dies (DAPI / Concanavalin A / SYTO14 / WGA+phalloidin / Mitotracker). 
The package contains a user-friendly and tunable interface for cellular segmentation (check the preview.ipynb notebook).  It has two GPU-back ended options for the segmentation of the nucleus and cell: cellpose and pycleranto.  
It uses a novel method to match the segmentation of nucleus and cytoplasm, then uses the those two segmentation masks to segment the nucleoli and mitochondira as well.

SPACe is fast, ~10X faster than CellProfiler, using a reasonably standard desktop and not using any cloud computing resources. It takes about 6-9 hours to analyze a full 384-well plate (~17000 2000X2000 pixel images) via pytorch/GPU as well as CPU-Multiprocessing for speedup.  The output is based on single cell data and is provided as canonical well-based summary statistics (mean, median) and as earth mover’s distance measurements of each well to a DMSO control reference distribution.  

## **Installation instructions**

If you do not have anaconda installed on your computer, the first step is to install anaconda3 as follows:  
  1.	Download Anaconda: Go to the Anaconda website (ADD LINK) and download the Anaconda3 installer for your operating system (Windows, macOS, or Linux).
  2.	Run Installer: Once the installer is downloaded, run it and follow the installation instructions.
  3.	Agree to Terms: Read and agree to the license agreement.
  4.	Choose Install Location: Choose a directory where Anaconda will be installed. The default location is usually fine.
  5.	Select Installation Type: Choose whether to install Anaconda just for you or for all users on the system.
  6.	Advanced Options (Optional): You may have the option to add Anaconda to your system PATH environment variable, which can make it easier to use from the command line. This is typically recommended.
  7.	Install: Click the "Install" button to begin the installation process.
  8.	Complete Installation: Once the installation is complete, you may be prompted to install Visual Studio Code (VS Code) or PyCharm. You can choose to install them or skip this step if you prefer.
  9.	Verify Installation: Open a terminal or command prompt and type “conda --version" to verify that Anaconda has been installed correctly. You should see the version number of Anaconda displayed.

To install SPACe python package on a conda virtualenv called tensors:

1.	Install anacond3/miniconda3 on your windows or linux machine (see instruction above to download anaconda)
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
   **Option 1)**
  To install SPACe directly from github:
  ``` pip install git+https://github.com/dlabate/SPACe.git ```
  **Option 2)**
  To install SPACe locally on your computer and in order to be able to edit it, first download it from github to your computer, go to one directory above where your SPACe folder is
  downloaded/located/saved, through an anaconda3 terminal, the structure of the directory would look like this:
  ```
  dir/
   cellpaint/
            setup.py
            README.md
            cellpaint
  ```
type ``` pip install -e cellpaint ``` in the same terminal. The -e allows one to edit the program.
All the required packages will be installed automatically from ``` setup.py ``` file.
If you are going to use Pycharm, its terminal might not recognize your anaconda3 virtualenv. Here is the fix from 
``` https://stackoverflow.com/questions/48924787/pycharm-terminal-doesnt-activate-conda-environment ```.
If the pycharm terminal does not recognize your anaconda virtualenv, do the following:
Go to ``` File -> Settings -> Tools -> Terminal ```. Replace the value in ``` Shell path ``` with ``` cmd.exe "/K" path_to_your_miniconda3\Scripts\activate.bat tensors ```.
Remember, you will be able to modify the content of this package only if you install it via Option 2).


## **Running SPACe**

To learn how to use the program, go to [run_SPACe.md](https://github.com/dlabate/SPACe/blob/master/run_SPACe.md).



