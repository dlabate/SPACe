import re
import time
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from cellpaint.utils.post_feature_extraction import ROCAUC, PlateMapAnnot, FeaturePreprocessing
from cellpaint.steps_single_plate.step0_args import Args
import matplotlib.pyplot as plt
import seaborn as sns
import cmcrameri as cmc


class ROCAUCAnalysis(FeaturePreprocessing, ROCAUC, PlateMapAnnot):
    """ """
    analysis_step = 7
    make_plots = True

    aggs = ["mean", "sum"]
    aggs_new = ["AVG Cell Count", "Total Cell Count"]
    part_1_group_cols = ["exp-id", "cell-line", "density", "treatment", "dosage"]
    part_2_group_cols = ["exp-id", "density", "dosage", "treatment", "cell-line"]
    hit_col_auc = "is-hit-auc"
    hit_col_diff_auc = "is-hit-deriv-auc"

    # distance_map_index = 2
    # distance_map_name = "wasserstein"

    plot_type = 0
    is_multi_dose = False
    # args.hit_compartment_count = 4
    # mad_mult = 10

    def __init__(self, args):
        """https://stackoverflow.com/questions/9575409/
        calling-parent-class-init-with-multiple-inheritance-whats-the-right-way"""

        FeaturePreprocessing.__init__(self)
        ROCAUC.__init__(self, args)
        PlateMapAnnot.__init__(self, args)
        # colors = ["black", "blue", "red", "green", "orange", "yellow", "brown", "pink", "purple"]
        self.save_path = self.args.step7_save_path
        self.n_tr = len(self.args.treatments)
        self.n_ctrl_tr = len(self.args.control_treatments)

        # assuming we already know there are only there feature categories
        assert self.num_feat_cat == 3
        # self.args.hit_mad_multipliers =
        # np.repeat(np.array([[self.args.hit_mad_mult]*self.num_feat_cat]), repeats=self.num_cin, axis=0).T
        # print(f"self.args.hit_mad_multipliers: {self.args.hit_mad_multipliers.shape}")
        self.excel_savename1 = \
            f"hits_well-level_mad-mult={self.args.hit_mad_mult}_comp-#={self.args.hit_compartment_count}.csv"
        self.excel_savename2 = \
            f"hits_treatment-level_mad-mult={self.args.hit_mad_mult}_comp-#={self.args.hit_compartment_count}.csv"

    def run_analysis(self):
        print("Cellpaint Step 7: Generating Median-ROC-Curves and Median-ROC-AUCs, Plots and excel files, "
              " for all DistanceMap csv files generated in Cellpaint Step 5...")
        self.s_time = time.time()

        print(f"# treatments: {self.n_tr}    # control treatments: {self.n_ctrl_tr}")
        assert self.n_tr >= self.n_ctrl_tr

        print("Cellpaint Step7 partI: Generating ROC-Curves and ROC-AUCs "
              "for different (Cell-line, Density, dosage) triplets ...")
        self.partI_get_graphs(controls_only=False)
        # if self.n_tr > self.n_ctrl_tr:
        #     print('All Treatments ...')
        #     self.partI_get_graphs(controls_only=False)
        #     print('Controls Only ...')
        #     # # to check on control treatments separately to see if they have worked!!!
        #     self.partI_get_graphs(controls_only=True)
        # else:
        #     print('All Treatments ...')
        #     self.partI_get_graphs(controls_only=True)

        print("Cellpaint Step7 partII: Generating ROC-Curves and ROC-AUCs to compare DMSO across Cell-lines ...")
        self.partII_get_graphs()

    def partI_get_graphs(self, controls_only):
        """https://stackoverflow.com/questions/38985053/pandas-groupby-and-sum-only-one-column"""
        controls = self.args.control_treatments

        if len(self.args.treatments) == 1:
            print(f"There is only one treatment={self.args.anchor_treatment}! "
                  "No treatment comparison is necessary!!!Skipping!!!")
            return None

        self.plot_type = 2
        # pairs = self.get_df_unix_df(self.args.platemap, ["density", "cell-line"])
        # get the list of csv files obtained in step 5, except the DMSO ones!
        csv_files = list(self.args.step5_save_path.rglob("*_wasserstein.csv"))
        csv_files = [it for it in csv_files if f"{self.args.anchor_treatment.upper()}" not in it.stem]
        if len(csv_files) == 0:
            raise ValueError("Distance maps are not generated!!! Make sure to run the 5th Cellpaint step first!")

        pass_wells, well_level_metadata, treatment_level_metadata = [], [], []
        for ii, filepath in enumerate(csv_files):
            # loading csv file
            distmap_df = self.compatibility_fn(filepath)
            distmap_df = distmap_df.loc[distmap_df["well-status"] == "pass"]
            if controls_only:
                distmap_df = distmap_df.loc[np.isin(distmap_df["treatment"].to_numpy(), self.args.control_treatments)]
            # move controls up which resets index as well
            distmap_df.sort_values(by=["exp-id", "cell-line", "density"],).sort_values(
                by=["treatment"], ascending=[True], key=lambda x: np.isin(x, controls),).sort_values(
                by=["treatment"], ascending=[True], key=lambda x: x == "DMSO",).sort_values(
                by=["dosage", "well-id"], inplace=True)
            # This is a must for the rest to work properly!!!!
            distmap_df.reset_index(inplace=True, drop=True)
            # treatment level unique values from distmap_df
            groups = distmap_df.groupby(self.part_1_group_cols, sort=False)
            unix_df = groups["cell-count"].agg(self.aggs).reset_index()
            unix_df.rename(dict(zip(self.aggs, self.aggs_new)), axis=1, inplace=True)
            # for ii, (key, it) in enumerate(groups):
            #     print(key, unix_df.iloc[ii][self.part_1_group_cols])

            # # print(unix_df.head(10))
            # unix_indices = groups.apply(lambda group: group.index.tolist())
            #
            self.is_multi_dose = self.is_exp_multi_dose(unix_df)
            # # print(ii, it.stem, "unix_df: ", unix_df.shape, list(unix_df.columns))
            treatments = np.unique(unix_df["treatment"])
            if len(treatments) <= 1:
                print(f"There exist only a single treatment {treatments} in {filepath.stem}. "
                      f"Skipping this csv file!!!")
                continue
            roc_curves, roc_aucs, roc_diff_curves, roc_diff_aucs = self.compute_roccurve_and_rocauc_np(distmap_df)
            # roc_curves, roc_diff_curves, roc_aucs, roc_diff_aucs = self.compute_roccurve_and_rocauc_torch(
            #     distmap_df, self.feature_types, self.channels, batch_size=384)
            assert roc_curves.shape[2] == roc_aucs.shape[2] == distmap_df.shape[0]
            print(distmap_df.shape, roc_curves.shape, roc_diff_curves.shape)
            #
            # # TODO: Fix partI and partII using derivative and TBV as well.
            # # calculate roc_curves, and roc_aucss for visualization as well as hit calling
            # get median values per well
            roc_curves_median = groups.apply(lambda x: np.median(roc_curves[:, :, x.index, :], axis=2))
            roc_diff_curves_median = groups.apply(lambda x: np.median(roc_diff_curves[:, :, x.index, :], axis=2))
            roc_aucs_median = groups.apply(lambda x: np.median(roc_aucs[:, :, x.index], axis=2))
            roc_diff_aucs_median = groups.apply(lambda x: np.median(roc_diff_aucs[:, :, x.index], axis=2))
            print(roc_curves_median.shape, roc_curves_median[0].shape)

            roc_curves_median = np.concatenate([it[:, :, np.newaxis, :] for it in roc_curves_median], axis=2)
            roc_diff_curves_median = np.concatenate([it[:, :, np.newaxis, :] for it in roc_diff_curves_median], axis=2)
            roc_aucs_median = np.concatenate([it[:, :, np.newaxis] for it in roc_aucs_median], axis=2)
            roc_diff_aucs_median = np.concatenate([it[:, :, np.newaxis] for it in roc_diff_aucs_median], axis=2)

            print(
                  roc_curves_median.shape,
                  roc_diff_curves_median.shape,
                  roc_aucs_median.shape,
                  roc_diff_aucs_median.shape)

            # print(distmap_df.shape, unix_df.shape, roc_curves_median.shape)
            # print(
            #     f"distmap_df: {distmap_df.shape}   "
            #     f"unix_df: {unix_df.shape}    "
            #     f"roc_curves: {roc_curves.shape}   "
            #     f"roc_aucs: {roc_curves.shape}   "
            #     f"roc_curves_median: {roc_curves_median.shape}   "
            #     f"roc_aucs_median: {roc_aucs_median.shape}")
            # calculate roc_curves, and roc_aucss for visualization as well as hit calling
            distmap_df[self.hit_col_auc] = self.get_well_level_hits(
                roc_aucs, distmap_df, self.args.hit_compartment_count, self.args.hit_mad_mult)
            unix_df[self.hit_col_auc] = self.get_treatment_level_hits(
                unix_df, distmap_df, distmap_df[self.hit_col_auc].to_numpy())

            distmap_df[self.hit_col_diff_auc] = self.get_well_level_hits(
                roc_diff_aucs, distmap_df, self.args.hit_compartment_count, self.args.hit_mad_mult)
            unix_df[self.hit_col_diff_auc] = self.get_treatment_level_hits(
                unix_df, distmap_df, distmap_df[self.hit_col_diff_auc].to_numpy())

            # I have to fix column names
            N1, N2, N3, N4 = roc_aucs.shape[0], roc_aucs.shape[1], roc_aucs.shape[2], roc_aucs_median.shape[2]
            df1 = distmap_df[["cell-count", "well-id", "treatment", self.hit_col_auc, self.hit_col_diff_auc,
                              "dosage", "cell-line", "density"]]
            df2 = unix_df[self.aggs_new+["treatment", self.hit_col_auc, self.hit_col_diff_auc,
                                         "dosage", "cell-line", "density"]]

            cols3 = [f"roc-auc_{it}" for it in self.xlabels]
            cols4 = [f"median-roc-auc_{it}" for it in self.xlabels]
            cols5 = [f"roc-derivative-auc_{it}" for it in self.xlabels]
            cols6 = [f"median-roc-derivative-auc_{it}" for it in self.xlabels]
            df3 = pd.DataFrame(np.transpose(np.reshape(roc_aucs, (N1 * N2, N3)), (1, 0)), columns=cols3)
            df4 = pd.DataFrame(np.transpose(np.reshape(roc_aucs_median, (N1 * N2, N4)), (1, 0)), columns=cols4)
            df5 = pd.DataFrame(np.transpose(np.reshape(roc_diff_aucs, (N1 * N2, N3)), (1, 0)), columns=cols5)
            df6 = pd.DataFrame(np.transpose(np.reshape(roc_diff_aucs_median, (N1 * N2, N4)), (1, 0)), columns=cols6)
            # print(df1.shape, df3.shape, df2.shape, df4.shape)
            well_level_metadata.append(pd.concat([df1, df3, df5], axis=1))
            treatment_level_metadata.append(pd.concat([df2, df4, df6], axis=1))
            pass_wells += distmap_df["well-id"].to_list()

            if self.make_plots:
                if self.is_multi_dose:
                    for jj, dos in enumerate(
                            unix_df.loc[unix_df["treatment"] != self.args.anchor_treatment]["dosage"].unique()):
                        cond = (unix_df["dosage"].to_numpy() == dos) | \
                               (unix_df["treatment"].to_numpy() == self.args.anchor_treatment)
                        savename1, savename2, savename3, savename4, title1, title2, title3, title4 = \
                            self.get_partI_savenames_and_titles(
                                filepath, ii, self.is_multi_dose, controls_only, jj, dos)
                        meta_df = (unix_df.loc[cond]).reset_index(drop=True)

                        self.plot_roccurves(
                            roc_curves_median[:, :, cond, :],
                            meta_df,
                            self.plot_type,
                            self.is_multi_dose,
                            title=title1,
                            savename=savename1,
                            hit_col=self.hit_col_auc)
                        self.plot_rocauc_heatmap(
                            roc_aucs_median[:, :, cond],
                            meta_df,
                            self.aggs_new[0],
                            self.plot_type,
                            self.is_multi_dose,
                            title=title2,
                            savename=savename2,
                            hit_col=self.hit_col_auc)

                        self.plot_roccurves(
                            roc_diff_curves_median[:, :, cond, :],
                            meta_df,
                            self.plot_type,
                            self.is_multi_dose,
                            title=title3,
                            savename=savename3,
                            hit_col=self.hit_col_diff_auc)
                        self.plot_rocauc_heatmap(
                            roc_diff_aucs_median[:, :, cond],
                            meta_df,
                            self.aggs_new[0],
                            self.plot_type,
                            self.is_multi_dose,
                            title=title4,
                            savename=savename4,
                            hit_col=self.hit_col_diff_auc)

                        # print(f"roc_aucs_median:{roc_aucs_median[:, :, cond].shape}   meta_df:{meta_df.shape}")
                else:
                    savename1, savename2, savename3, savename4, title1, title2, title3, title4 = \
                        self.get_partI_savenames_and_titles(filepath, ii, self.is_multi_dose, controls_only)

                    self.plot_roccurves(
                        roc_curves_median[:, :, :, :],
                        unix_df,
                        self.plot_type,
                        self.is_multi_dose,
                        title=title1,
                        savename=savename1,
                        hit_col=self.hit_col_auc)
                    self.plot_rocauc_heatmap(
                        roc_aucs_median[:, :, :],
                        unix_df,
                        self.aggs_new[0],
                        self.plot_type,
                        self.is_multi_dose,
                        title=title2,
                        savename=savename2,
                        hit_col=self.hit_col_auc)

                    self.plot_roccurves(
                        roc_diff_curves_median[:, :, :, :],
                        unix_df,
                        self.plot_type,
                        self.is_multi_dose,
                        title=title3,
                        savename=savename3,
                        hit_col=self.hit_col_diff_auc)
                    self.plot_rocauc_heatmap(
                        roc_diff_aucs_median[:, :, :],
                        unix_df,
                        self.aggs_new[0],
                        self.plot_type,
                        self.is_multi_dose,
                        title=title4,
                        savename=savename4,
                        hit_col=self.hit_col_diff_auc)

        n_wells = len(well_level_metadata)
        if (n_wells > 0) & ((self.n_tr > self.n_ctrl_tr and not controls_only) or (self.n_tr == self.n_ctrl_tr)):
            # print("create Hits platemap and save roc_aucss together with their metadata as csv files ...")
            well_level_metadata = pd.concat(well_level_metadata, axis=0, ignore_index=True)
            treatment_level_metadata = pd.concat(treatment_level_metadata, axis=0, ignore_index=True)
            well_level_metadata.sort_values(
                by=["cell-line", "density", "treatment", "dosage", self.hit_col_auc, self.hit_col_diff_auc],
                inplace=True)
            treatment_level_metadata.sort_values(
                by=["cell-line", "density", "treatment", "dosage", self.hit_col_auc, self.hit_col_diff_auc],
                inplace=True)
            # print(well_level_metadata.columns)

            well_level_metadata.to_csv(
                self.args.step7_save_path / self.excel_savename1,
                index=False,
                float_format="%.2f")
            treatment_level_metadata.to_csv(
                self.args.step7_save_path / self.excel_savename2,
                index=False,
                float_format="%.2f")
            self.plot_hits(well_level_metadata, pass_wells, hit_col=self.hit_col_auc)
            self.plot_hits(well_level_metadata, pass_wells, hit_col=self.hit_col_diff_auc)

    def partII_get_graphs(self, ):

        if len(self.args.celllines) == 1:
            print(f"There is only one cell-line={self.args.celllines[0]} found in the entire plate! "
                  "No DMSO comparison across cell-lines can be made! Skipping Cellpaint Step 7 part II ...")
            print(f"Finished Cellpaint step 7 in: {(time.time() - self.s_time) / 3600} hours")
            print("***********************")
            return None
        self.plot_type = 3
        self.is_multi_dose = False

        treat_save_name = self.args.anchor_treatment.upper()
        csv_files = list(self.args.step5_save_path.rglob(f"{treat_save_name}_*_wasserstein.csv"))
        if len(csv_files) == 0:
            raise ValueError("Distance-maps are missing and have not been generated!!! "
                             "Make sure to run Cellpaint step 5 first!")
        elif len(csv_files) > 1:
            raise ValueError(f"Multiple DMSO wasserstein Distance maps were found in {self.args.step5_save_path}!!!"
                             f"Make sure there is only 1!!!")
        # loading csv file
        distmap_df = self.compatibility_fn(csv_files[0])
        distmap_df = distmap_df.loc[distmap_df["well-status"] == "pass"]
        # This is a must for the rest to work properly!!!!
        distmap_df.reset_index(inplace=True, drop=True)
        #######################################################
        # using unix_df (unique meta-data) to make plots average per unique value
        unix_df = distmap_df.groupby(self.part_2_group_cols)["cell-count"].agg(self.aggs).reset_index()
        unix_df.rename(dict(zip(self.aggs, self.aggs_new)), axis=1, inplace=True)
        # print("unix_df: ", unix_df.shape, list(unix_df.columns))
        roc_curves, roc_aucs, roc_diff_curves, roc_diff_aucs = \
            self.compute_roccurve_and_rocauc(distmap_df, distmap_df, self.plot_type)
        assert roc_curves.shape[2] == roc_aucs.shape[2] == distmap_df.shape[0]
        # moving controls to the top positions for better visibility
        distmap_df, unix_df, roc_curves, roc_aucs, roc_diff_curves, roc_diff_aucs = \
            self.move_controls_up(distmap_df, unix_df, roc_curves, roc_aucs, roc_diff_curves, roc_diff_aucs)
        roc_curves_median, roc_diff_curves_median, _, _, roc_aucs_median, roc_diff_aucs_median, _, _ = \
            self.get_median_and_mad_maps_per_well(
                distmap_df, unix_df, roc_curves, roc_aucs, roc_diff_curves, roc_diff_aucs)

        title = f"{self.args.experiment}: {self.args.anchor_treatment} Comparison across Cell-lines"
        savename1 = f"{treat_save_name}_0_ROC-Curves"
        savename2 = f"{treat_save_name}_0_ROC-AUCs"
        savename3 = f"{treat_save_name}_1_ROC-Derivative-Curves"
        savename4 = f"{treat_save_name}_1_ROC-Derivative-AUCs"

        if self.make_plots:
            self.plot_roccurves(
                roc_curves_median,
                unix_df,
                self.plot_type,
                self.is_multi_dose,
                title=f"{treat_save_name} ROC-Curves\n{title}",
                savename=savename1,
                hit_col="")
            self.plot_rocauc_heatmap(
                roc_aucs_median,
                unix_df,
                self.aggs_new[0],
                self.plot_type,
                self.is_multi_dose,
                title=f"{treat_save_name} ROC-AUCs\n{title}",
                savename=savename2,
                hit_col="")
            self.plot_roccurves(
                roc_diff_curves_median,
                unix_df,
                self.plot_type,
                self.is_multi_dose,
                title=f"{treat_save_name} ROC-Derivative-Curves\n{title}",
                savename=savename3,
                hit_col="")
            self.plot_rocauc_heatmap(
                roc_diff_aucs_median,
                unix_df,
                self.aggs_new[0],
                self.plot_type,
                self.is_multi_dose,
                title=f"{treat_save_name} ROC-Derivative-AUCs\n{title}",
                savename=savename4,
                hit_col="")

        # SAVING
        self.save_auc_df_partII(distmap_df, unix_df, roc_aucs, roc_aucs_median, treat_save_name,
                                auc_name="ROC-AUCs")
        self.save_auc_df_partII(distmap_df, unix_df, roc_diff_aucs, roc_diff_aucs_median, treat_save_name,
                                auc_name="RUC-AUCs-Derivative")

    def get_partI_savenames_and_titles(self, filepath, ii, is_multi_dose, controls_only, jj=0, dos=0):
        den = filepath.stem.split("_")[0].replace("density=", "")
        cl = str(filepath.stem.split("_")[1].replace("cellline=", ""))
        exp_id = filepath.parents[1].stem

        name1 = "Median-ROC-Curves"
        name2 = "Median-ROC-AUCs"
        name3 = "Median-ROC-Derivative-Curves"
        name4 = "Median-ROC-Derivative-AUCs"

        title1 = f"{exp_id}\n{name1}: Density={den}  Cell-line={cl}"
        title2 = f"{exp_id}\n{name2}: Density={den}  Cell-line={cl}"
        title3 = f"{exp_id}\n{name3}: Density={den}  Cell-line={cl}"
        title4 = f"{exp_id}\n{name4}: Density={den}  Cell-line={cl}"

        if not is_multi_dose:
            print(f"case-index={ii}   Density={den}   Cell-line={cl}")
            savename1 = f"c{ii}_{name1}_Density={den}-Cellline={cl}"
            savename2 = f"c{ii}_{name2}_Density={den}-Cellline={cl}"
            savename3 = f"c{ii}_{name3}_Density={den}-Cellline={cl}"
            savename4 = f"c{ii}_{name4}_Density={den}-Cellline={cl}"
        else:
            print(f"case-index={ii}-{jj}   Density={den}   Cell-line={cl}   Dose={dos}")
            savename1 = f"c{ii}-{jj}_{name1}_Density={den}-Cellline={cl}-Dose={dos}"
            savename2 = f"c{ii}-{jj}_{name2}_Density={den}-Cellline={cl}-Dose={dos}"
            savename3 = f"c{ii}-{jj}_{name3}_Density={den}-Cellline={cl}-Dose={dos}"
            savename4 = f"c{ii}-{jj}_{name4}_Density={den}-Cellline={cl}-Dose={dos}"
            title1 += f"  Dose={dos}"
            title2 += f"  Dose={dos}"
            title3 += f"  Dose={dos}"
            title4 += f"  Dose={dos}"

        if controls_only:
            savename1 = "1-Controls_" + savename1
            savename2 = "1-Controls_" + savename2
            savename3 = "1-Controls_" + savename3
            savename4 = "1-Controls_" + savename4

        return savename1, savename2, savename3, savename4, title1, title2, title3, title4

    def save_auc_df_partII(self, distmap_df, unix_df, roc_aucs_np, roc_aucs_np_median, treat_save_name, auc_name):
        # print(f"roc_aucss:{roc_aucss.shape}   distmap_df:{unix_df.shape}")
        N1, N2, N3, N4 = roc_aucs_np.shape[0], roc_aucs_np.shape[1], roc_aucs_np.shape[2], roc_aucs_np_median.shape[2]
        well_level_metadata = pd.concat([
            distmap_df[["cell-count", "well-id", "cell-line", "density", "treatment", "dosage"]],
            pd.DataFrame(np.transpose(np.reshape(roc_aucs_np, (N1 * N2, N3)), (1, 0)), columns=self.xlabels)],
            axis=1)
        treatment_level_metadata = pd.concat([
            unix_df[self.aggs_new + ["cell-line", "density", "treatment", "dosage"]],
            pd.DataFrame(np.transpose(np.reshape(roc_aucs_np_median, (N1 * N2, N4)), (1, 0)), columns=self.xlabels)],
            axis=1)
        well_level_metadata.sort_values(by=["treatment", "dosage", "density", "cell-line"], inplace=True)
        treatment_level_metadata.sort_values(by=["treatment", "dosage", "density", "cell-line"], inplace=True)

        well_level_metadata.to_csv(
            self.args.step7_save_path / f"{auc_name}_{treat_save_name}_avg-well-level.csv",
            index=False, float_format="%.2f")
        treatment_level_metadata.to_csv(
            self.args.step7_save_path / f"{auc_name}_{treat_save_name}_avg-treatment-level.csv",
            index=False, float_format="%.2f")
        print(f"Finished Cellpaint step 7 in: {(time.time() - self.s_time) / 3600} hours")
        print("***********************")

    def get_well_level_hits(self, roc_aucs_arr, distmap_df, hit_compartment_count, hit_mad_mult):
        """
            roc_auc_arr: numpy array of shape (self.num_feat_cat, self.num_cin, num_curves)
            distmap_df: pandas dataframe of shape (num_curves, self.num_cols)
        """
        mad_multipliers = np.repeat(np.array([[hit_mad_mult] * self.num_feat_cat]),
                                    repeats=self.num_cin, axis=0).T
        anchor_idxs = distmap_df.index[
            (distmap_df["treatment"] == self.args.anchor_treatment) &
            (distmap_df["dosage"] == self.args.anchor_dosage)].to_list()
        N1, N2 = distmap_df.shape[0], roc_aucs_arr.shape[2]
        assert N1 == N2
        # assuming cell-line, and density are fixed and we only have a single experiment
        assert len(anchor_idxs) > 1, "There has to be more than 1 DMSO well!!!"

        anchor_median = np.median(np.take(roc_aucs_arr, anchor_idxs, axis=2), axis=2)
        anchor_median = np.repeat(anchor_median[:, :, np.newaxis], repeats=N1, axis=2)
        anchor_mad = median_abs_deviation(np.take(roc_aucs_arr, anchor_idxs, axis=2), axis=2)
        anchor_mad = np.repeat(anchor_mad[:, :, np.newaxis], repeats=N1, axis=2)
        coeff = np.repeat(mad_multipliers[:, :, np.newaxis], repeats=N1, axis=2)

        # print(anchor_median.shape, anchor_mad.shape, roc_auc_arr.shape, self.args.hit_mad_multipliers.shape)
        difference, radius = roc_aucs_arr - anchor_median, coeff * anchor_mad
        hits = (difference - radius > 0) | (difference + radius < 0)
        # hits = hits01 | hits12 | hits20
        # hits_shape = np.sum(hits[0, :, :], axis=0) >= 1
        # hits_intensity = np.sum(hits[1, :, :], axis=0) >= 1
        # hits_haralick = np.sum(hits[2, :, :], axis=0) >= 1
        # hits = ((hits_shape & hits_intensity) | (hits_haralick & hits_intensity)).astype(np.uint8)
        # print(np.sum(hits[:, :, :], axis=(0, 1))[0:20])
        hits = (np.sum(hits[:, :, :], axis=(0, 1)) >= hit_compartment_count).astype(np.uint8)
        return hits

    def get_treatment_level_hits(self, unix_df, distmap_df, well_hits):
        """
            distmap_df: pandas dataframe of shape (num_curves, self.num_cols)
            unix_df: pandas dataframe of shape (N1, self.num_cols) where N1<=num_curves
        """
        treat_hits = np.zeros((unix_df.shape[0], ), dtype=np.uint8)
        for ii, row in unix_df.iterrows():
            indices = self.find_row_indices(distmap_df, row)
            count = np.sum(well_hits[indices])
            total = len(indices)  # has to be positive
            if total <= 2:
                treat_hits[ii] = count == total
            elif total < 3:
                treat_hits[ii] = count >= 2
            else:
                treat_hits[ii] = count/total >= .75
        return treat_hits

    def plot_hits(self, well_level_unix_df, pass_wells, hit_col):
        # create plate-maps
        title = f"Hit Calls: {self.args.experiment}"
        annot_save_names = ["treatment", "cell-line", "dosage", "density"]
        annot_font_size = [6.5, 8, 9, 9]
        colors_meta = [
            ("blue", 0, "Not Hit"),
            ("red", 1, "Hit"),
            ("black", 2, "Ignore"),
            # ("green", 1, "Partial Hit"),
        ]
        # print(self.rows, self.cols)
        annot = np.zeros((4, self.nrows, self.ncols), dtype=object)
        data = pd.DataFrame(np.zeros((self.nrows, self.ncols), dtype=np.float32), columns=self.cols, index=self.rows)
        # if np.isin(self.args.anchor_treatment, well_level_unix_df["treatment"]):
        #     well_level_unix_df.drop(
        #     well_level_unix_df[well_level_unix_df["treatment"] == self.args.anchor_treatment].index, inplace=True)
        for ii in range(self.nrows):
            for jj in range(self.ncols):
                well_id = f"{self.rows[ii]}{self.cols[jj]}"
                if self.args.wellid2treatment.get(well_id) is None:  # it could be a well-id is missing
                    annot[:, ii, jj] = ""
                    hit = 2
                    # tr, cl = "", ""
                else:
                    tr = self.args.wellid2treatment[well_id]
                    cl = self.args.wellid2cellline[well_id]
                    den = self.args.wellid2density[well_id]
                    dos = self.args.wellid2dosage[well_id]
                    # print(type(cl))
                    cond = (well_level_unix_df["well-id"] == well_id) & \
                           (well_level_unix_df["cell-line"] == cl) & \
                           (well_level_unix_df["density"] == den) & \
                           (well_level_unix_df["treatment"] == tr) & \
                           (well_level_unix_df["dosage"] == dos)
                    # print(np.sum(cond),)
                    if (np.sum(cond) == 1) & np.isin(well_id, pass_wells):
                        # print(well_id, cl, den, tr, dos)
                        hit = well_level_unix_df[cond][hit_col].to_numpy()[0]
                    else:
                        hit = 2
                    # annot[0, ii, jj] = '\n'.join(treat.lower()[0:8].split('-'))
                    tr_abv = Args.shorten_str(tr, "treatment", self.args.max_chars)
                    cl_abv = Args.shorten_str(cl, "cell-line", self.args.max_chars)
                    annot[0, ii, jj] = tr_abv
                    annot[1, ii, jj] = cl_abv
                    annot[2, ii, jj] = dos
                    annot[3, ii, jj] = den
                # print(f"{well_id:10} {tr_abv:35} {cl_abv:20} {hit}")
                #     print(f"{well_id:8} {tr:40} "
                #           f"pass-well={np.isin(well_id, pass_wells)}  "
                #           f"tr-eq={np.sum(well_level_unix_df['treatment'] == tr)}  "
                #           f"cl-eq={np.sum(well_level_unix_df['cell-line'] == cl)}  "
                #           f"dos-eq={np.sum(well_level_unix_df['dosage'] == dos)}   "
                #           f"den-eq={np.sum(well_level_unix_df['density'] == den)}  "
                #           f"all-eq{np.sum(cond):7}  "
                #           f"{np.isin(well_id, pass_wells):7}  {hit}")
                data.loc[self.rows[ii], self.cols[jj]] = hit

        for kk in range(4):
            self.create_discretized_heatmap(
                data=data,
                annotation=annot[kk],
                annotation_font_size=annot_font_size[kk],
                title=title,
                colors_meta=colors_meta,
                xtick_labels=self.cols,
                ytick_labels=self.rows,
                save_path=self.save_path,
                save_name=f"Platemap_{hit_col}_{kk}_{annot_save_names[kk]}")

    def is_exp_multi_dose(self, unix_df):
        """Check whether the experiment has wells with multiple dosages (the same exact ones)
         for all none-DMSO treatments in its platemap."""
        # TODO: Fix multi-dose: when there is only a single exact dose for all None-DMSO treatments,
        #  the is_multi_dose variable must equal False. But this is not necessary at this point.
        treatments = np.unique(unix_df["treatment"].to_numpy())
        if len(treatments) == 1:
            is_multi_dose = False
        else:
            is_multi_dose = True
            treatments = np.setdiff1d(treatments, [self.args.anchor_treatment])
            pairs = {it: np.unique(unix_df.loc[unix_df["treatment"] == it, "dosage"].to_numpy())
                     for it in treatments}
            # determine whether the array of all dosages is the same for all the different treatments
            vals = list(pairs.values())

            for ii in range(0, len(vals) - 1):
                check1 = np.array_equal(vals[ii], vals[ii + 1])
                if not check1:
                    is_multi_dose = False
                    break
            print("self.is_multi_dose = ", is_multi_dose)
        return is_multi_dose

    def get_median_and_mad_maps_per_well(self,
                                         distmap_df, unix_df,
                                         roc_curves, roc_aucs,
                                         roc_diff_curves, roc_diff_aucs,
                                         ):
        """The order of rows in distmap_df, roc_curves, and roc_aucs must match!!!!
        distmap_df: pandas dataframe of shape (N, M)
        unix_df: pandas dataframe of shape (N1, M) where N1<=N
        roc_curves: numpy array of shape (self.num_feat_cat, self.num_cin, N, self.num_thresholds)
        roc_curves: numpy array of shape (self.num_feat_cat, self.num_cin, N)
        """
        N = len(unix_df)
        assert unix_df.index.to_list() == pd.RangeIndex(start=0, stop=N, step=1).to_list()
        roc_curves_median, roc_diff_curves_median, roc_curves_mad, roc_diff_curves_mad,  \
            roc_aucs_median, roc_diff_aucs_median, roc_aucs_mad, roc_diff_aucs_mad = \
            np.zeros((self.num_feat_cat, self.num_cin, N, self.num_thresholds), dtype=np.float32), \
            np.zeros((self.num_feat_cat, self.num_cin, N, self.num_thresholds), dtype=np.float32), \
            np.zeros((self.num_feat_cat, self.num_cin, N, self.num_thresholds), dtype=np.float32), \
            np.zeros((self.num_feat_cat, self.num_cin, N, self.num_thresholds), dtype=np.float32), \
            np.zeros((self.num_feat_cat, self.num_cin, N), dtype=np.float32), \
            np.zeros((self.num_feat_cat, self.num_cin, N), dtype=np.float32), \
            np.zeros((self.num_feat_cat, self.num_cin, N), dtype=np.float32), \
            np.zeros((self.num_feat_cat, self.num_cin, N), dtype=np.float32)

        # Iterate over each unique (cell-line, density, treatment, dosage) and
        # average AUC_MAP and DERIV_MAP over its corresponding wells
        for ii, row in unix_df.iterrows():
            indices = self.find_row_indices(distmap_df, row)

            roc_curves_median[:, :, ii] = np.median(roc_curves[:, :, indices], axis=2)
            roc_diff_curves_median[:, :, ii] = np.median(roc_diff_curves[:, :, indices], axis=2)
            roc_aucs_median[:, :, ii] = np.median(roc_aucs[:, :, indices], axis=2)
            roc_diff_aucs_median[:, :, ii] = np.median(roc_diff_aucs[:, :, indices], axis=2)

            roc_curves_mad[:, :, ii] = median_abs_deviation(roc_curves[:, :, indices], axis=2)
            roc_diff_curves_mad[:, :, ii] = median_abs_deviation(roc_diff_curves[:, :, indices], axis=2)
            roc_aucs_mad[:, :, ii] = median_abs_deviation(roc_aucs[:, :, indices], axis=2)
            roc_diff_aucs_mad[:, :, ii] = median_abs_deviation(roc_diff_aucs[:, :, indices], axis=2)

        return roc_curves_median, roc_diff_curves_median, roc_curves_mad, roc_diff_curves_mad,  \
            roc_aucs_median, roc_diff_aucs_median, roc_aucs_mad, roc_diff_aucs_mad

    def split_into_anchor_control_test(self, df, plot_type):
        if plot_type == 2:
            # get number of control treatments, all treatments
            N1, N2, = len(self.args.control_treatments), len(np.unique(df["treatment"]))
            cond0 = df["treatment"].to_numpy() == self.args.anchor_treatment
            if N1 == N2 and N1 == 1:
                # There is only a single group: anchor-treatment (DMSO)
                conds = [cond0]
            elif N1 == N2 and N1 > 1:
                # divide into 2 ygroups in order:
                # 1) anchor-treatment, 2) control-treatments except anchor
                conds = [cond0, ~cond0]
            elif N1 != N2:
                # divide into 3 ygroups in order:
                # 1) anchor-treatment 2) control-treatments except anchor 3) test-treatments (Not control treatments)
                cond1 = np.isin(df["treatment"].to_numpy(), self.args.control_treatments)
                conds = [cond0, (~cond0) & cond1, ~cond1]
            else:
                raise ValueError(f"It must be the case that {N1}>=1")

        elif plot_type == 3:  # DMSO cell-line comparison
            # divide into 2 ygroups in order:
            # 1) anchor-cellline 2) rest of cell-line
            N2 = len(np.unique(df["cell-line"]))
            assert N2 > 1, f"There has to be more than one cell-line for comparison, found {N2}"
            cond0 = df["cell-line"].to_numpy() == self.args.anchor_cellline
            conds = [cond0, ~cond0]
        else:
            raise NotImplementedError()
        return conds

    def move_controls_up(self, distmap_df, unix_df, roc_curves, roc_aucs, roc_diff_curves, roc_diff_aucs):
        # moving controls to the top positions for better visibility
        conds0 = self.split_into_anchor_control_test(distmap_df, self.plot_type)
        conds1 = self.split_into_anchor_control_test(unix_df, self.plot_type)
        M = len(conds0)
        distmap_df = pd.concat([distmap_df[conds0[ii]] for ii in range(M)], axis=0, ignore_index=True)
        unix_df = pd.concat([unix_df[conds1[ii]] for ii in range(M)], axis=0, ignore_index=True)

        roc_curves = np.concatenate([roc_curves[:, :, conds0[ii]] for ii in range(M)], axis=2)
        roc_aucs = np.concatenate([roc_aucs[:, :, conds0[ii]] for ii in range(M)], axis=2)
        roc_diff_curves = np.concatenate([roc_diff_curves[:, :, conds0[ii]] for ii in range(M)], axis=2)
        roc_diff_aucs = np.concatenate([roc_diff_aucs[:, :, conds0[ii]] for ii in range(M)], axis=2)
        return distmap_df, unix_df, roc_curves, roc_aucs, roc_diff_curves, roc_diff_aucs

    @staticmethod
    def find_row_indices(distmap_df, row):
        """
        The order of rows in distmap_df, roc_curves, and roc_aucs must match!!!!
         distmap_df:
            pandas dataframe of shape (N, M)
         row:
            pandas Series of shape (1, M), which is a row from unix_df pandas dataframe of shape (N1, M) where N1 <= N

         plot_type (== 2 or == 3): int
             a switch flag to determine whether we are inside the
             partII_get_graphs function (for TREATMENT COMPARISON) or
             partII_get_graphs function (for DMSO COMPARISON across cell-lines)
         """
        mask = (distmap_df["exp-id"] == row["exp-id"]) & \
               (distmap_df["cell-line"] == row["cell-line"]) & \
               (distmap_df["density"] == row["density"]) & \
               (distmap_df["treatment"] == row["treatment"]) & \
               (distmap_df["dosage"] == row["dosage"])
        indices = distmap_df.index[mask].to_list()
        return indices

    @staticmethod
    def get_df_unix_df(df, keys):
        unix_df = []
        grp = df.groupby(list(keys)) if len(keys) > 1 else df.groupby(keys[0])
        for key, val in grp:
            print(key)
            unix_df.append(key)
        print('\n')
        # print(unix_df, keys, unix_df[0])
        # print(df[get_df_at_cond(df, keys, unix_df[0])])
        return unix_df

    @staticmethod
    def get_df_at_cond(df, keys, vals):
        assert isinstance(vals, np.ndarray)
        N, M = len(df), len(keys)
        cond = np.ones((N,), dtype=bool)
        for ii in range(M):
            cond &= (df[keys[ii]] == vals[ii])
        return cond
