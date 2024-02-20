import time
from sys import getsizeof

import torch
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pandas as pd

from cellpaint.utils.torch_api import DistCalcDataset, DistCalcModel, dist_calc_fn
from cellpaint.utils.post_feature_extraction import FeaturePreprocessing

pd.options.mode.chained_assignment = None


# TODO: ADD self.args.control_treatments to the platemap for the QC-step


class FeatureDistMap(FeaturePreprocessing):
    """ """
    # min_feat_rows = 1000  # minimum number of cells which is the number of rows in the features dataframe
    # batch_size = 8  # 256
    analysis_step = 5
    normalize_quantile_range = (2, 98)

    def __init__(self, args):
        # initialize the parent classes
        FeaturePreprocessing.__init__(self)
        self.args = args

        # these values can be over-ridden by the subclasses
        self.metrics = ["mean", "wasserstein", "median"]  # "mode", "bimodality-index"
        self.num_metric = len(self.metrics)
        self.metrics = [f"{ii + 1}_{self.metrics[ii]}" for ii in range(self.num_metric)]

        self.calc_cols = ["exp-id", "well-id"]
        self.sort_cols = ["cell-line", "density", "treatment", "other", "dosage", ] + ["well-status", ]
        self.group_cols = self.calc_cols + self.sort_cols
        self.sort_ascending = [True, ] * len(self.sort_cols)

        self.all_features, self.cell_count, self.start_index = self.load_and_preprocess_features(
            self.args.step3_save_path, self.args.step4_save_path, self.args.min_well_cell_count, self.analysis_step)
        # ###############################################
        # # for Seema only bc of high confluency
        # if "seema" in self.args.experiment.lower():
        #     print("Seema experiments can only use CPU ...")
        #     # self.device = "cpu"
        #     self.args.distmap_batch_size = 2
        #     # torch.set_num_threads(16)
        self.feat_cols = list(self.all_features.columns)[self.start_index:]

    def main_run_loop(self):
        """
        Main function for cellpaint step IV:
            It generates distribution, median, and mean distance-maps (for a SINGLE experiment),
            based on the following three scenarios so far:

            Note that, there are 6 sheets in the platemap excel file at the moment:
                Treatment, Cellline, Density, Dosage, Other, and Anchor.

            We can use the first 5 sheets to decide/switch to/indentify the type of experiment it is.
            We use the last sheet (Anchor) to decide what is the anchor condition (our 0/zero to compare against)

                Case 1) A dose-response-benchmarking experiment, when:
                    There are multiple dosages found in the Dosage sheet, of the platemap excel file.

                Case 2) A (DMSO/Vehicle) Density-benchmarking experiment, when:
                    2-1) There is a SINGLE dosage in the Dosage sheet, AND
                    2-2) There is a SINGLE treatment/compound (DMSO/Vehicle) in the Treatment sheet, BUT
                        the are multiple densities in the Density sheet, of the platemap excel file

                Case 3) A Density-benchmarking experiment for a whole set of treatments/compounds, when:
                    3-1) There exist a single dosage in the Dosage sheet, BUT
                    3-2) There are multiple treatments/compounds in the Treatment sheet, And
                    3-3) There are multiple densities in the Density sheet, of the platemap excel file.
            Finally,
            Also whenever there multiple cell-lines in the CellLine sheet of the platemap excel file,
            we always need to compare the (DMSO/vehicle) of different cell-lines against each other, given
            the Anchor cell-line provided the Anchor sheet, of the platemap excel-file.

            It saves the resulting feature maps into separate csv files:
                self.args.distancemaps_path / f"{save_name}_distrib-dist.csv"
                self.args.distancemaps_path / f"{save_name}_mean-dist.csv"
                self.args.distancemaps_path / f"{save_name}_median-dist.csv"

            where

            if args.mode.lower() == "debug":
                self.args.distancemaps_path = args.main_path / args.experiment / "Debug" / "DistanceMaps"
            elif args.mode.lower() == "test":
                self.args.distancemaps_path = args.main_path / args.experiment / "Test" / "DistanceMaps"
            elif args.mode.lower() == "full":
                self.args.distancemaps_path = args.main_path / args.experiment / "DistanceMaps"
        """
        # basically in CellPaintArgs.add_platemap_anchor_args, we read/got the following for the last sheet the platemap
        # excel file:
        #   1) anchor compound/treatment (==dmso   99.99% percent of the time)
        #   2) anchor cell-line (only needed when there are multiples celllines)
        #   3) anchor density (only needed for density experiments)
        #   4) anchor dosage (==0   99.9% of the time, needed for dose-response experiments.)
        #   5) anchor other (==0 90% of the time. Only needed for when the other sheet of the platemap is not empty!)

        s_time = time.time()
        print("Cellpaint Step 5 ... Calculating distance maps")
        # if (len(self.args.dosages) > 1) or \
        #         (len(self.args.dosages) == 1 and len(self.args.densities) == 1 and len(self.args.treatments) == 1):
        # print("Found multiple Dosages ... Creating Distance Maps for a Dosage Response Benchmarking ...")

        # TODO: investigate this problem later
        # args.anchor_dosage = 0 if np.isin(0, args.anchor_dosage) else 0.01
        # anchor_dosage = np.amin(args.anchor_dosage)
        self.anchor_col = "dosage"
        self.anchor_col_val = self.args.anchor_dosage

        self.calc_dose_distmap_treatments_per_cellline()
        self.calc_dose_distmap_dmso_per_cellline()

        # elif len(self.args.dosages) == 1 and len(self.args.densities) > 1:
        #     print("Found a single dosage, but multiple densities ... "
        #           "Creating Distance Maps for Density Benchmarking ...")
        #     self.anchor_col = "density"
        #     self.anchor_col_val = self.args.anchor_density
        #     self.calc_density_distmap_treatments_per_cellline()
        #     self.calc_density_distmap_dmso_per_cellline()
        print(f"Cellpaint Step 5 took {time.time() - s_time} seconds to finish ...")
        print("***********************")

    def calc_distance_matrix_wellwise(self, features, anchor_features, case_name):
        start_time = time.time()
        # batch_size = self.get_batch_size([dataset.features_tensor], dataset.groups_meta)
        dataset = DistCalcDataset(features, self.group_cols, self.start_index,
                                  self.args.min_well_cell_count, self.analysis_step)
        if len(dataset.unique_indices) < 1:  # no groups with enough cells/rows
            print("All the wells for this cell-line Density Dosage Treatment have already been removed!")
        # each element in the batch contains features of all the cells within a specific well-id
        data_loader = DataLoader(
            dataset,
            batch_size=self.args.distmap_batch_size,
            shuffle=False,
            pin_memory=False,
            num_workers=0,
            collate_fn=dataset.custom_collate_fn)
        model = DistCalcModel(anchor_features, self.feat_cols, self.analysis_step)
        model.eval()
        # if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        #     model = torch.nn.DataParallel(model)
        model = model.to(self.device)
        distances = dist_calc_fn(model, data_loader, self.device, self.analysis_step)

        # saving, we have to split in two orders: 1) per experiment 2) per distance type
        for ii in range(distances.shape[0]):
            dist_data = pd.DataFrame(
                np.concatenate((dataset.groups_metadata, distances[ii]), axis=1),
                columns=["cell-count"] + self.group_cols + dataset.feat_cols)
            # #######################################
            # # Already taken care of in the PostFeatureExtraction.load_and_preprocess_features function
            # # remove well-ids with few cells
            # dist_data = dist_data.loc[dist_data["cell-count"] >= self.args.min_well_cell_count]
            ################################################
            dist_data.sort_values(by=self.sort_cols, ascending=self.sort_ascending, inplace=True)
            # Move control treatment rows to the top of the dataframe for better visibility
            dist_data = self.shift_up_control_treatments(
                dist_data, self.args.control_treatments, self.args.anchor_treatment)
            # TODO: if there are multiple experiments use this:
            # dist_data = dist_data.groupby(["exp-id"], group_keys=False)
            # for experiment, item in dist_data:
            #     save_path = self.args.main_path/experiment/"Step5_DistanceMaps"
            #     item.to_csv(save_path / f"{case_name}_{self.metrics[ii]}.csv", index=False)
            dist_data.to_csv(self.args.step5_save_path / f"{case_name}_{self.metrics[ii]}.csv", index=False)
        print("time taken in secs ... ", time.time() - start_time)
        print('\n')

    def calc_dose_distmap_treatments_per_cellline(self, ):
        """Here we assume that the anchor treatment is always gonna be a member of the control treatments!!!"""
        treatments = self.all_features["treatment"].unique()
        if len(treatments) > 1:  # only when we have more than one treatment this would make sense
            groups = self.all_features.groupby(["density", "cell-line"], group_keys=False)
            # For each (density, cell-line), compare all treatments within the same cell-line
            for ii, (key, feats) in enumerate(groups):
                case_name = f"density={key[0]}_cellline={key[1]}"
                print(f"Case {ii} {case_name}:  #cols={len(feats.columns)}")
                # the new way: After quality control was introduced
                cond = np.isin(feats["treatment"].to_numpy(), self.args.control_treatments) & \
                       (feats["well-status"].to_numpy() == "pass")
                # feats.fillna(0, inplace=True)
                feats = self.normalize_features_transformer(
                    feats.loc[cond], feats, self.start_index, self.normalize_quantile_range)
                print(self.args.anchor_treatment, self.args.anchor_other, self.anchor_col, self.anchor_col_val)

                # the old way: Before quality control was introduced
                # feats = self.normalize_features(feats, self.start_index, self.normalize_quantile_range)
                anchor_feats = self.get_anchor_features(
                    feats.loc[cond], self.args.experiment, self.args.anchor_treatment, self.args.anchor_other,
                    self.anchor_col, self.anchor_col_val)
                print("feats shape: ", feats.shape, "anchor_feats shape: ", anchor_feats.shape)
                self.calc_distance_matrix_wellwise(feats, anchor_feats, case_name)

    def calc_dose_distmap_dmso_per_cellline(self, ):
        """Restricting to self.args.anchor_treatment which is dmso 99% of the time"""
        # print("calculating distance maps for DMSO comparison across cell-lines... ")
        feats = self.all_features[self.all_features["treatment"] == self.args.anchor_treatment]
        assert len(feats) > self.args.distmap_min_feat_rows, \
            "DMSO (anchor treatment) must be present in all cell-lines with high cell count!"
        celllines, densities = feats["cell-line"].unique(), feats["density"].unique()
        print(celllines, densities)

        if len(celllines) > 1:  # only when we have more than one cell-line this would make sense
            cond = feats["well-status"].to_numpy() == "pass"
            # feats_den = self.normalize_features_transformer(
            #     feats_den.loc[cond], feats_den, self.start_index, self.normalize_quantile_range)
            feats = self.normalize_features(feats, self.start_index, self.normalize_quantile_range)
            anchor_feats = self.get_anchor_features(
                feats.loc[cond],
                self.args.experiment, self.args.anchor_treatment, self.args.anchor_other,
                "cell-line", self.args.anchor_cellline)
            print(f"feats {feats.shape}  anchor feats {anchor_feats.shape}")
            savename = f"DMSO_{self.args.anchor_cellline}"
            self.calc_distance_matrix_wellwise(feats, anchor_feats, savename)

    def calc_density_distmap_treatments_per_cellline(self, ):
        """Here we assume that the anchor treatment is always gonna be a member of the control treatments!!!"""

        print("calculating distance maps for Density Benchmarking ... ")
        # restricting to self.args.anchor_treatment which is dmso 99% of the time
        celllines, densities = self.all_features["cell-line"].unique(), self.all_features["density"].unique()
        print(celllines, densities)
        groups = self.all_features.groupby("cell-line", group_keys=False)
        for ii, (key, feats_cl) in enumerate(groups):  # for each cell-line, compare different densities
            cond = np.isin(feats_cl["treatment"].to_numpy(), self.args.control_treatments) & \
                   (~np.isin(feats_cl["well-id"].to_numpy(), self.bad_wellids))
            # feats_cl = self.normalize_features(feats_cl, self.start_index, self.normalize_quantile_range)
            feats_cl = self.normalize_features_transformer(
                feats_cl.loc[cond], feats_cl, self.start_index, self.normalize_quantile_range)
            anchor_feats = self.get_anchor_features(
                feats_cl.loc[cond],
                self.args.experiment, self.args.anchor_treatment, self.args.anchor_other,
                self.anchor_col, self.anchor_col_val)
            print(f"features {feats_cl.shape}\n anchor feats {anchor_feats.shape}")
            self.calc_distance_matrix_wellwise(feats_cl, anchor_feats, f"cellline={key}")

    def calc_density_distmap_dmso_per_cellline(self, ):
        print("calculating distance maps for Density Benchmarking ... ")
        # restricting to self.args.anchor_treatment which is dmso 99% of the time
        feats = self.all_features[self.all_features["treatment"] == self.args.anchor_treatment]
        celllines, densities = feats["cell-line"].unique(), feats["density"].unique()
        print(celllines, densities)
        if len(celllines) > 1:
            # feats = self.normalize_features(feats, self.start_index, self.normalize_quantile_range)
            cond = ~np.isin(feats["well-id"].to_numpy(), self.bad_wellids)
            feats = self.normalize_features_transformer(
                feats.loc[cond], feats, self.start_index, self.normalize_quantile_range)
            anchor_feats = self.get_anchor_features(
                feats.loc[cond], self.args.experiment, self.args.anchor_treatment, self.args.anchor_other,
                "cell-line", self.args.anchor_cellline)
            print(f"features {feats.shape}\n anchor feats {anchor_feats.shape}")
            self.calc_distance_matrix_wellwise(feats, anchor_feats, f"DMSO_{self.args.anchor_cellline}")

    def calc_treatments_across_celllines(self, replicate_exp_ids, anchor_exp_id):
        # TODO: CHANGE AND MOVE THIS FUNCTION LATER!!!

        print(replicate_exp_ids, anchor_exp_id)
        assert anchor_exp_id == replicate_exp_ids[0], """Always put the anchor as the first element."""
        # print("calculating distance maps for DRC, all cell-lines together ... ")
        all_features = pd.concat([self.load_and_preprocess_features(it) for it in replicate_exp_ids], axis=0)
        # all_features = self.exclude_celline(all_features)
        print(f"all_features after combining:  shape={all_features.shape}  #cols={len(all_features.columns)}")
        # print(np.unique(all_features["exp-id"]))
        groups = all_features.groupby(["density", "cell-line"], group_keys=False)
        for ii, (key, feats) in enumerate(groups):
            case_name = f"density={key[0]}_cellline={key[1]}"
            print(f"{case_name}  #cols={len(feats.columns)}")
            # feats = self.drop_uninformative_feature_cols(feats)
            # if feats.shape[1] <= self.min_num_cols:
            #     continue
            cond = np.isin(feats["treatment"].to_numpy(), self.args.control_treatments) & \
                   (~np.isin(feats["well-id"].to_numpy(), self.bad_wellids))
            feats = self.normalize_features_transformer(
                feats.loc[cond], feats, self.start_index, self.normalize_quantile_range)
            # feats = self.normalize_features(feats, self.start_index, self.normalize_quantile_range)
            # if there are multiple experiments with the same cell-line, then anchor experiment matters,
            # otherwise it does not.
            anchor_exp_id = anchor_exp_id if np.isin(anchor_exp_id, feats["exp-id"].values) else \
                feats["exp-id"].iloc[0]
            anchor_feats = self.get_anchor_features(
                feats[cond], anchor_exp_id, self.args.anchor_treatment, self.args.anchor_other,
                self.anchor_col, self.anchor_col_val)
            print(f"features {feats.shape}\n anchor feats {anchor_feats.shape}")
            self.calc_distance_matrix_wellwise(feats, anchor_feats, case_name)
