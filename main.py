import time
from pathlib import WindowsPath, Path

from SPACe.steps_single_plate.step0_args import Args
from SPACe.steps_single_plate.step1_segmentation_preview import preview_run_loop
from SPACe.steps_single_plate.step2_segmentation_p1 import step2_main_run_loop
from SPACe.steps_single_plate.step3_segmentation_p2 import step3_main_run_loop
from SPACe.steps_single_plate.step4_feature_extraction import step4_main_run_loop
from SPACe.steps_single_plate.step5_distance_map_with_torch_api import WellAggFeatureDistanceMetrics


def set_default_datasets_hyperparameters(args):
    ##############################################################################
    # intensity rescaling hyperparameters
    args.w1_intensity_bounds = (5, 99.95)
    args.w2_intensity_bounds = (5, 99.95)
    args.w3_intensity_bounds = (5, 99.95)
    args.w4_intensity_bounds = (5, 99.95)
    args.w5_intensity_bounds = (5, 99.95)
    ##########################################################################
    # background correction hyperparameters
    """Set args.bg_sub to True first if you decide to do background subtraction."""
    args.bg_sub = False
    args.w1_bg_rad = 50
    args.w2_bg_rad = 100
    args.w3_bg_rad = 50
    args.w4_bg_rad = 100
    args.w5_bg_rad = 100
    #######################################################################
    # image channels order/index during data acquisition set by the investigator/microscope
    args.nucleus_idx = 0
    args.cyto_idx = 1
    args.nucleoli_idx = 2
    args.actin_idx = 3
    args.mito_idx = 4
    #######################################################################
    # hyperparameters/constants used in Cellpaint Step 2
    args.step2_segmentation_algorithm = "w1=cellpose_w2=cellpose"
    args.cellpose_nucleus_diam = 20
    args.cellpose_cyto_diam = 75
    args.cellpose_batch_size = 64
    args.cellpose_model_type = "cyto2"
    args.w1_min_size = 400
    args.w2_min_size = 700
    args.w3_min_size = 40
    args.w5_min_size = 200
    #######################################################
    # hyperparameters/constants used in Cellpaint Step 3
    args.multi_nucleus_dist_thresh = 40
    args.min_nucleoli_size_multiplier = .005
    args.max_nucleoli_size_multiplier = .3
    args.nucleoli_bd_area_to_nucleoli_area_threshold = .2
    args.w3_local_rescale_intensity_ub = 99.2
    args.w5_local_rescale_intensity_ub = 99.9
    return args


def set_seema_datasets_hyperparameters(args):
    ##############################################################################
    # intensity rescaling hyperparameters
    args.w1_intensity_bounds = (0.1, 99.99)
    args.w2_intensity_bounds = (0.1, 99.99)
    args.w3_intensity_bounds = (0.1, 99.99)
    args.w4_intensity_bounds = (0.1, 99.99)
    args.w5_intensity_bounds = (0.1, 99.99)
    ##########################################################################
    # background correction hyperparameters
    """Set args.bg_sub to True first if you decide to do background subtraction."""
    args.bg_sub = False
    args.w1_bg_rad = 50
    args.w2_bg_rad = 100
    args.w3_bg_rad = 50
    args.w4_bg_rad = 100
    args.w5_bg_rad = 100
    #######################################################################
    # image channels order/index during data acquisition set by the investigator/microscope
    args.nucleus_idx = 0
    args.cyto_idx = 1
    args.nucleoli_idx = 2
    args.actin_idx = 3
    args.mito_idx = 4
    #######################################################################
    # hyperparameters/constants used in Cellpaint Step 2
    args.step2_segmentation_algorithm = "w1=pycle_w2=pycle"
    args.cellpose_nucleus_diam = 30
    args.cellpose_cyto_diam = 30
    args.cellpose_batch_size = 64
    args.cellpose_model_type = "cyto2"
    args.w1_min_size = 200
    args.w2_min_size = 300
    args.w3_min_size = 5
    args.w5_min_size = 30
    #######################################################
    # hyperparameters/constants used in Cellpaint Step 3
    args.multi_nucleus_dist_thresh = 20
    args.min_nucleoli_size_multiplier = .000001
    args.max_nucleoli_size_multiplier = .999
    args.nucleoli_bd_area_to_nucleoli_area_threshold = .01
    args.w3_local_rescale_intensity_ub = 99.99
    args.w5_local_rescale_intensity_ub = 99.99
    return args


def set_jump_consortium_datasets_cpg0012_hyperparameters(args):
    ##########################################################################
    # background correction hyperparameters
    """Set args.bg_sub to True first if you decide to do background subtraction."""
    args.bg_sub = False
    args.w1_bg_rad = 50
    args.w2_bg_rad = 100
    args.w3_bg_rad = 50
    args.w4_bg_rad = 100
    args.w5_bg_rad = 100
    #######################################################################
    # image channels order/index during data acquisition set by the investigator/microscope
    args.nucleus_idx = 0
    args.cyto_idx = 1
    args.nucleoli_idx = 2
    args.actin_idx = 3
    args.mito_idx = 4
    #######################################################################
    # hyperparameters/constants used in Cellpaint Step 2
    args.step2_segmentation_algorithm = "w1=cellpose_w2=cellpose"
    args.cellpose_nucleus_diam = 20
    args.cellpose_cyto_diam = 75
    args.cellpose_batch_size = 64
    args.cellpose_model_type = "cyto2"
    args.w1_min_size = 400
    args.w2_min_size = 700
    args.w3_min_size = 40
    args.w5_min_size = 200
    #######################################################
    # hyperparameters/constants used in Cellpaint Step 3
    args.multi_nucleus_dist_thresh = 10
    args.min_nucleoli_size_multiplier = .000001
    args.max_nucleoli_size_multiplier = .999
    args.nucleoli_bd_area_to_nucleoli_area_threshold = .01
    args.w3_local_rescale_intensity_ub = 99.99
    args.w5_local_rescale_intensity_ub = 99.99
    args.min_nucleoli_size_multiplier = .000001
    args.max_nucleoli_size_multiplier = .999
    # args.nucleus_area_to_cyto_area_thresh = .6  this param is no longer available!!!
    return args


def set_jump_consortium_datasets_cpg0001_hyperparameters(args):
    ##########################################################################
    # background correction hyperparameters
    """Set args.bg_sub to True first if you decide to do background subtraction."""
    args.bg_sub = False
    args.w1_bg_rad = 50
    args.w2_bg_rad = 100
    args.w3_bg_rad = 50
    args.w4_bg_rad = 100
    args.w5_bg_rad = 100
    ############################################################
    # image channels order
    args.nucleus_idx = 4
    args.cyto_idx = 3
    args.nucleoli_idx = 2
    args.actin_idx = 1
    args.mito_idx = 0
    #######################################################################
    # hyperparameters/constants used in Cellpaint Step 2
    args.step2_segmentation_algorithm = "w1=cellpose_w2=cellpose"
    args.cellpose_nucleus_diam = 25
    args.cellpose_cyto_diam = 25
    args.cellpose_batch_size = 64
    args.cellpose_model_type = "cyto2"
    args.w1_min_size = 400
    args.w2_min_size = 500
    args.w3_min_size = 4
    args.w5_min_size = 30
    ###########################################################################
    # hyperparameters/constants used in Cellpaint Step 3
    args.multi_nucleus_dist_thresh = 10
    args.min_nucleoli_size_multiplier = .000001
    args.max_nucleoli_size_multiplier = .999
    args.nucleoli_bd_area_to_nucleoli_area_threshold = .01
    args.w3_local_rescale_intensity_ub = 99.99
    args.w5_local_rescale_intensity_ub = 99.99
    args.min_nucleoli_size_multiplier = .000001
    args.max_nucleoli_size_multiplier = .999
    # args.nucleus_area_to_cyto_area_thresh = .6  this param is no longer available!!!
    return args


def set_jump_consortium_datasets_cpg0001MOA_hyperparameters(args):
    ##########################################################################
    # background correction hyperparameters
    """Set args.bg_sub to True first if you decide to do background subtraction."""
    args.bg_sub = False
    args.w1_bg_rad = 50
    args.w2_bg_rad = 100
    args.w3_bg_rad = 50
    args.w4_bg_rad = 100
    args.w5_bg_rad = 100
    ############################################################
    # image channels order
    args.nucleus_idx = 0
    args.cyto_idx = 1
    args.nucleoli_idx = 2
    args.actin_idx = 3
    args.mito_idx = 4
    #######################################################################
    # hyperparameters/constants used in Cellpaint Step 2
    args.step2_segmentation_algorithm = "w1=cellpose_w2=cellpose"
    args.cellpose_nucleus_diam = 20
    args.cellpose_cyto_diam = 25
    args.cellpose_batch_size = 64
    args.cellpose_model_type = "cyto2"
    args.w1_min_size = 400
    args.w2_min_size = 500
    args.w3_min_size = 4
    args.w5_min_size = 30
    ###########################################################################
    # hyperparameters/constants used in Cellpaint Step 3
    args.multi_nucleus_dist_thresh = 10
    args.min_nucleoli_size_multiplier = .000001
    args.max_nucleoli_size_multiplier = .999
    args.nucleoli_bd_area_to_nucleoli_area_threshold = .01
    args.w3_local_rescale_intensity_ub = 99.99
    args.w5_local_rescale_intensity_ub = 99.99
    args.min_nucleoli_size_multiplier = .000001
    args.max_nucleoli_size_multiplier = .999
    # args.nucleus_area_to_cyto_area_thresh = .6  this param is no longer available!!!
    return args


def set_jump_consortium_datasets_cimini_hyperparameters(args):
    ##########################################################################
    # background correction hyperparameters
    """Set args.bg_sub to True first if you decide to do background subtraction."""
    args.bg_sub = False
    args.w1_bg_rad = 50
    args.w2_bg_rad = 100
    args.w3_bg_rad = 50
    args.w4_bg_rad = 100
    args.w5_bg_rad = 100
    ############################################################
    # image channels order
    args.nucleus_idx = 4
    args.cyto_idx = 2
    args.nucleoli_idx = 1
    args.actin_idx = 3
    args.mito_idx = 0
    #######################################################################
    # hyperparameters/constants used in Cellpaint Step 2
    args.step2_segmentation_algorithm = "w1=cellpose_w2=cellpose"
    args.cellpose_nucleus_diam = 20
    args.cellpose_cyto_diam = 25
    args.cellpose_batch_size = 64
    args.cellpose_model_type = "cyto2"
    args.w1_min_size = 400
    args.w2_min_size = 500
    args.w3_min_size = 4
    args.w5_min_size = 30
    ###########################################################################
    # hyperparameters/constants used in Cellpaint Step3
    args.multi_nucleus_dist_thresh = 10
    args.min_nucleoli_size_multiplier = .000001
    args.max_nucleoli_size_multiplier = .999
    args.nucleoli_bd_area_to_nucleoli_area_threshold = .01
    args.w3_local_rescale_intensity_ub = 99.99
    args.w5_local_rescale_intensity_ub = 99.99
    args.min_nucleoli_size_multiplier = .000001
    args.max_nucleoli_size_multiplier = .999
    # args.nucleus_area_to_cyto_area_thresh = .6  this param is no longer available!!!
    return args


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
        pass


# print(out_path)
if __name__ == "__main__":
    
    experiment_path = Path("/project/labate/CellPaint/Jump_Consortium_Datasets_cpg0001/2020_08_11_Stain3_Yokogawa")
    experiment_folder = ['BR00115127']
    
    print(experiment_path)
    print(type(experiment_path))
    print(experiment_folder)
    print(type(experiment_folder))
    
    for exp_fold in experiment_folder:
        # print(type(exp_fold))
        # print(exp_fold)
        # entry point of the program is creating the necessary args
        print("*********************************************************"
              "*********************************************************"
              "*********************************************************"
              "*********************************************************")
        start_time = time.time()
        args = Args(experiment=exp_fold, main_path=experiment_path, mode="full").args
        args = set_default_datasets_hyperparameters(args)
        main_worker(args)
        print(f"program finished analyzing experiment {args.experiment} in {(time.time()-start_time)/3600} hours ... ")
        print("*********************************************************"
              "*********************************************************"
              "*********************************************************"
              "*********************************************************")

    # for exp_fold in experiment_folder:
    #     # print(type(exp_fold))
    #     # print(exp_fold)
    #     # entry point of the program is creating the necessary args
    #     print("*********************************************************"
    #           "*********************************************************"
    #           "*********************************************************"
    #           "*********************************************************")
    #     start_time = time.time()
    #     args = Args(experiment=exp_fold, main_path=experiment_path, mode="full").args
    #     args = set_default_datasets_hyperparameters(args)
    #     main_worker(args)
    #     print(f"program finished analyzing experiment {args.experiment} in {(time.time()-start_time)/3600} hours ... ")
    #     print("*********************************************************"
    #           "*********************************************************"
    #           "*********************************************************"
    #           "*********************************************************")
