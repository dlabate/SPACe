**Necessary Modifications Before running the program**

1) make sure the platemap excel file is filled-in properly,
and is in the same directory as your ```experiment_path``` which is the path to your images/experiment folder.

2) Then modify the ```args``` python variable. 
The ```args``` variable is a namespace python object that holds all the user input/hyperparamter information. 
To learn more about its available keys/fields/options go to ```cellpaint/steps_single_plate/step0_args.py```
After making sure all the necessary adjustments are made to the ```args``` Namespace, and your also satisfied
with segmentation on a few images using ```preview.ipynb```.
Copy your own ```set_custom_datasets_hyperparameters``` into ```main.py```:
```

def set_default_datasets_hyperparameters(args):
	# default values for args based on the BCM/IBT perkim_elmer plate_protocol,
	# for on 2000x2000 pixel dimenion images taken at 20X magnification.
	...
	return args
def set_custom_datasets_hyperparameters(args):
    # change the  value of each args.sth according to your plate and needs
    ##############################################################################
    # intensity rescaling hyperparameters
    args.w1_intensity_bounds = (5, 99.95)
    args.w2_intensity_bounds = (5, 99.95)
    args.w3_intensity_bounds = (5, 99.95)
    args.w4_intensity_bounds = (5, 99.95)
    args.w5_intensity_bounds = (5, 99.95)
    ##########################################################################
    # background correction hyperparameters
    # Set args.bg_sub to True first,
    # if you decide to do background subtraction.
    args.bg_sub = False
    args.w1_bg_rad = 50
    args.w2_bg_rad = 100
    args.w3_bg_rad = 50
    args.w4_bg_rad = 100
    args.w5_bg_rad = 100
    #######################################################################
    # image channels order/index 
    # defined during data acquisition set by the investigator/microscope
    args.nucleus_idx = 0
    args.cyto_idx = 1
    args.nucleoli_idx = 2
    args.actin_idx = 3
    args.mito_idx = 4
    #######################################################################
    # hyperparameters/constants used in Cellpaint Step 2
    #options for args.step2_segmentation_algorithm are:
	# 1) "w1=cellpose_w2=cellpose"
	# 2) "w1=pycle_w2=pycle"
	# 3) "w1=cellpose_w2=pycle"
	# 4) "w1=pycle_w2=cellpose"
    args.step2_segmentation_algorithm = "w1=cellpose_w2=cellpose"
    args.cellpose_nucleus_diam = 100
    args.cellpose_cyto_diam = 100
    args.cellpose_batch_size = 64
    args.cellpose_model_type = "cyto2"
    # define the minimum size of segmented objects in each channel
    args.w1_min_size = 600
    args.w2_min_size = 700
    args.w3_min_size = 40
    args.w5_min_size = 200
    #######################################################
    # hyperparameters/constants used in Cellpaint Step 3
    ############################################
    # args.multi_nucleus_dist_thresh decides 
    # whether to break down a multi-nucleus cyto mask,
    # into individual cyto masks,
    # based on avg-pairwise distance of all the nucleus
    # inside that cytoplasm
    args.multi_nucleus_dist_thresh = 40
    #######################################
    args.min_nucleoli_size_multiplier = .005
    args.max_nucleoli_size_multiplier = .3
    args.nucleoli_bd_area_to_nucleoli_area_threshold = .2
    args.w3_local_rescale_intensity_ub = 99.2
    args.w5_local_rescale_intensity_ub = 99.9
    return args
```
3) Modify ```main.py``` by setting in your own ```experiment_path```, ```experiment_folder```.


**Running the program**

To run the program, you have two options, "preview" mode which allows you to inspect the segmentation
steps results closely on a few wells and "full" mode which run the entire pipline from start to finish
and should take about 8-10 hours to finish on a full 384-wellplate with 9 field of views taken from each well:
```
def main_worker(args):
    """
    This program has three modes:
        Always run main_worker using args.mode == "preview" first.

    1) args.mode="preview":
        It allows the user to see the result of segmentation
        quickly on a few set of images. This way they can make
        sure the hyperparameters of the program are chosen appropriately.

    2) args.mode="test":
        For developer only, Only if you would like to change the internals of the program
        It helps with debugging and making sure the logic of the code follows.
        It does not use the multiprocessing module in for loop.

    3) args.mode="full":
         Runs the main_worker on the entire set of tiff images in the
         args.main_path / args.experiment / args.img_folder folder.
         It uses the multiprocessing module in step 3 and 4 for speed-up.
    """
    if args.mode == "preview":
	# choose either a few wells, or pass your own sample_wellids as a list
        preview_run_loop(args, num_wells=2, sample_wellids=None)
    else:
        # segmentation of nucleus and cytoplasm
        step2_main_run_loop(args)
        # matching nucleus and cytoplasm labels,
        # and segmenting nucleoli and mitochondria
        step3_main_run_loop(args)
        # generates feature matrices as csv files
        step4_main_run_loop(args)
        # generates DistanceMaps as xlsx files
        step5 = WellAggFeatureDistanceMetrics(args)
        step5.step5_main_run_loop()

if __name__ == "__main__":
    experiment_path = WindowsPath(r"path_to_your_experiment_folder_excluding_the_experiment_folder_itself")
    experiment_folder = WindowsPath(r"your_experiment_folder")
    # entry point of the program is creating the necessary args
    start_time = time.time()
    args = Args(experiment=experiment_folder, main_path=experiment_path, mode="full").args  # mode="preview"
    # args = set_default_datasets_hyperparameters(args)
    args = set_custom_datasets_hyperparameters(args)
    main_worker(args)
    print(f"program finished analyzing experiment {args.experiment} in {(time.time()-start_time)/3600} hours ... ")
```


Finally, open an anaconda terminal and run the following commands:
```
conda activate tensors
python main.py
```

**Preparations for running it on a different sample plate than YOKO/PerkimElmer**

(modifications needed to be applied to ```cellpaint/steps_single_plate /step0_args.py```):
The structure of the directory of your data/images/plate has to be as follows:
```
Experiment_Name\
	Images_folder_Name\
		Your image tiff files
	Platemap.xlsx
```

0) The default settings of the program are based on the YOKOGAWA/PerkinElmer ```plate_protocol```, where
   the image channels and their corresponding florescent dies are:
```
self.args.channel_dies = {
    "C1": "DAPI",  # nucleus
    "C2": "Concanavalin A",  # cyto
    "C3": "Syto14",  # nucleoli
    "C4": "WGA+Phalloidin",  # actin
    "C5": "MitoTracker",  # mito}
``` 

1)	Make sure your_platemap_file.xlsx file is in the same directory as your tiff image files.
You need to make sure that your platemap format follows our empty template ```platemap_template.xlsx``` protocol.
Make sure to fill out all these sheets properly:
```
Treatment
CellLine
Dosage
Density
Other
Anchor
Control
```
2)	If you are not using the PerkinElmer plate protocol, or your images format is not 5 channels 
You need to update the ```sort_key_for_imgs``` function inside the 
```cellpaint/steps_single_plate /step0_args.py``` file. So that our cellpaint package knows 
how to extract the necessary metadata, from each individual tiff file inside that image folder:

```
def sort_key_for_imgs(file_path, sort_purpose, plate_protocol):
    """
    Get sort key from the img filename.
    The function is used to sort image filenames for a specific experiment taken with a specific plate_protocol.
    """
    # plate_protocol = plate_protocol.lower()
    if plate_protocol == "greiner" or plate_protocol == "perkinelmer":
	...
    elif plate_protocol == "combchem":
	...
    elif "cpg0012" in plate_protocol:
	...
    elif "cpg0001" in plate_protocol:
	...
    else:
        raise NotImplementedError(f"{plate_protocol} is not implemented yet!!!")

    if sort_purpose == "to_sort_channels":
        return folder, well_id, fov, channel

    elif sort_purpose == "to_group_channels":
        # print(folder, well_id, fov)
        return folder, well_id, fov

    elif sort_purpose == "to_match_it_with_mask_path":
        return f"{well_id}_{fov}"
    elif sort_purpose == "to_get_well_id":
        return well_id
    elif sort_purpose == "to_get_well_id_and_fov":
        return well_id, fov
    else:
        raise ValueError(f"sort purpose {sort_purpose} does not exist!!!")
```


Those metadata keys/values/fields are:
```
folder: The name of the image folder containing your tiff files
filename: the name of the tiff file which should be passed in as a WindowsPath/PosixPath object 
if you are using Windows/Linux respectively.  
well_id: The image filename should contain the well-id of that plate where the image tiff file is taken from.
fov: The image should contain the fov of that well where the image tiff file is taken from.
channel: Which die/channel does the image correspond to, for example:
```

For example, if using ```plate_protocol == "combchem"```:
```
elif plate_protocol == "combchem":
    """img filename example: .../P000025-combchem-v3-U2OS-24h-L1-copy1/P000025-combchem-v3-U2OS-24h-L1-copy1_B02_s5_w3C00578CF-AD4A-4A29-88AE-D2A2024F9A92.tif"""
    folder = file_path.parents[1].stem
    filename = file_path.stem
    split = filename.split("_")
    well_id = split[1]
    fov = split[2][1]
    channel = split[3][1]
```

Now you can have your own ```plate_protocol```:

```
elif plate_protocol == "myplate_protocol":
    """img filename example:
    .../myimg_with_its_own_wellid_fov_channel_info.tif"""
    folder = file_path.parents[1].stem
    filename = file_path.stem
    split = filename.split("_")
    well_id = split[...]
    fov = split[...]
    channel = split[...]
```
3)	Also, if your image folder may contain tiff other than the image files you need to figure out 
a way to filter them similar to how it is done for ```perkim-elmer```.
You may also need to provide the your own necessary sorting functions to sort the channels properly,
depending on how your microscope saves image filenames. For example, for ```perkim-elmer``` we have:
```
if self.args.plate_protocol.lower() in ["perkinelmer", "greiner"]:
    # sometimes there are other tif files in the experiment folder that are not an image, so we have to
    # remove them from img_paths, so that they do not mess-up the analysis.
    self.args.img_filepaths = list(
        filter(lambda x: x.stem.split("_")[-1][0:5] == "T0001", self.args.img_filepaths))
self.args.img_filepaths = sorted(
    self.args.img_filepaths,
    key=lambda x: sort_key_for_imgs(x, "to_sort_channels", self.args.plate_protocol))
```
Therefore, you need to have:
```
if self.args.plate_protocol.lower() in ["myplate_protocol",]:
    self.args.img_filepaths = list(
        filter(lambda x: my_filter_fn, self.args.img_filepaths))
self.args.img_filepaths = sorted(
    self.args.img_filepaths,
    key=lambda x: sort_key_for_imgs(x, "to_sort_channels", self.args.plate_protocol))
```
