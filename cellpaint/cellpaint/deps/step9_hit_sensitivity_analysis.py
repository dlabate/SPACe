import time
from tqdm import tqdm
from pathlib import WindowsPath

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from cellpaint.steps_single_plate.step8_rocauc import ROCAUCAnalysis


class HitSensitivityAnalysis(ROCAUCAnalysis):
    is_multi_dose = False
    plot_type = 2
    vmin = 0
    vmax = 150

    hit_compartment_counts = np.arange(1, 16)
    hit_mad_mults = np.arange(1, 20)
    analysis_step = 8

    def __init__(self, args):
        ROCAUCAnalysis.__init__(self, args)

        self.n_hcc = len(self.hit_compartment_counts)
        self.n_mm = len(self.hit_mad_mults)
        self.hit_count_thresholds = [(it0, it1)
                                     for it0 in self.hit_compartment_counts
                                     for it1 in self.hit_mad_mults]
        self.s_time = time.time()

    def get_hit_counts_sensitivity_analysis_heatmap(self,):
        print(
            "Cellpaint Step 8: Generating a Heatmap Displaying the # of Hits by changing the values of\n, "
            "(1) hit_compartment_counts: "
            "# of (feature_category, input_image_channel) compartments (?/15) that are *-MAD away"
            "from the DMSO-median-ROC-AUC 15 (feature_category, input_image_channel) values\n"
            "(2) mad_mult: The MAD Multiplier Value * in (1)....")
        # loading csv file
        csv_files = self.args.step5_save_path.rglob("*_wasserstein.csv")
        csv_files = [it for it in csv_files if f"{self.args.anchor_treatment.upper()}" not in it.stem]
        if len(csv_files) == 0:
            raise ValueError("Distance maps are not generated!!! Make sure to run the 5th Cellpaint step first!")

        for ii, filepath in enumerate(csv_files):
            den = filepath.stem.split("_")[0].replace("density=", "")
            cl = str(filepath.stem.split("_")[1].replace("cellline=", ""))
            exp_id = filepath.parents[1].stem

            fig_title1 = f"{exp_id}\nROC AUC Hit Count Sensitivity Analysis: Density={den} Cell-line={cl}"
            fig_save_name1 = f"ROC-AUC-Hit-Sensitivity-Analysis_density={den}-cellline={cl}"
            fig_title2 = f"{exp_id}\nROC Derivative AUC Hit Count Sensitivity Analysis: Density={den} Cell-line={cl}"
            fig_save_name2 = f"ROC-Derivative-AUC-Hit-Sensitivity-Analysis_density={den}-cellline={cl}"

            distmap_df = self.compatibility_fn(filepath)
            distmap_df = distmap_df.loc[distmap_df["well-status"] == "pass"]
            # This is a must for the rest to work properly!!!!
            distmap_df.reset_index(inplace=True, drop=True)
            # treatment level unique values from distmap_df
            unix_df = distmap_df.groupby(self.part_1_group_cols)["cell-count"].agg(self.aggs).reset_index()
            unix_df.rename(dict(zip(self.aggs, self.aggs_new)), axis=1, inplace=True)
            treatments = np.unique(unix_df["treatment"])
            if len(treatments) <= 1:
                print(f"There exist only a single treatment {treatments} in {filepath.stem}. "
                      f"Skipping this csv file!!!")
                return

            # calculate roc_curves, and roc_aucs for visualization as well as hit calling
            roc_curves, roc_aucs, roc_diff_curves, roc_diff_aucs = \
                self.compute_roccurve_and_rocauc(distmap_df, distmap_df, self.plot_type)
            assert roc_curves.shape[2] == roc_aucs.shape[2] == distmap_df.shape[0]
            hit_counts_arr1 = self.calculate_hit_count_sensitivity_arr(distmap_df, unix_df, roc_auc_arr=roc_aucs)
            hit_counts_arr2 = self.calculate_hit_count_sensitivity_arr(distmap_df, unix_df, roc_auc_arr=roc_diff_aucs)
            self.plot_hit_count_heatmap(hit_counts_arr1, fig_title1, fig_save_name1)
            self.plot_hit_count_heatmap(hit_counts_arr2, fig_title2, fig_save_name2)

        print(f"Finished Cellpaint step 8 in: {(time.time() - self.s_time) / 3600} hours")
        print("***********************")

    def calculate_hit_count_sensitivity_arr(self, distmap_df, unix_df, roc_auc_arr):
        # calculating hit_counts
        hit_counts = np.zeros(self.n_hcc * self.n_mm, dtype=np.uint64)
        for kk, (hit_compartment_count, hit_mad_mult) in \
                tqdm(enumerate(self.hit_count_thresholds), total=self.n_hcc * self.n_mm):
            # print(self.hit_compartment_count, self.mad_mult)
            hits_well_level = self.get_well_level_hits(roc_auc_arr, distmap_df, hit_compartment_count, hit_mad_mult)
            hit_treat_level = self.get_treatment_level_hits(unix_df, distmap_df, hits_well_level)
            hit_counts[kk] = np.sum(hit_treat_level)
        hit_counts = hit_counts.reshape((self.n_hcc, self.n_mm))
        hit_counts = pd.DataFrame(hit_counts, columns=self.hit_mad_mults, index=self.hit_compartment_counts)
        return hit_counts

    def plot_hit_count_heatmap(self, hit_counts_arr, fig_title, fig_save_name):
        # Creating a heatmap for hit_counts
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(21.3, 13.3)
        fig.suptitle(fig_title, fontname='Comic Sans MS', fontsize=17)
        sns.heatmap(
            hit_counts_arr,
            cmap=self.colormap,
            annot=True, fmt='g',
            annot_kws={"fontsize": 9},
            linecolor='gray', linewidth=1,
            vmin=self.vmin, vmax=self.vmax,
            cbar_kws={
                "orientation": "horizontal",
                "use_gridspec": False,
                "pad": .1, }
        )
        # Set common labels
        xlabel = "*: The MAD Multiplier Value"
        ylabel = "?: (?/15) Compartments that Are *-MAD Away\nfrom DSMO-Median of the Same Compartment"
        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)

        # Set y-axis ticks
        ax.set_yticks(np.arange(.5, len(self.hit_compartment_counts) + .5, 1))
        ax.set_yticklabels(self.hit_compartment_counts)
        ax.set_yticklabels(ax.get_ymajorticklabels(), rotation=0, fontsize=12)

        # Set x-axis ticks
        ax.set_xticks(np.arange(.5, len(self.hit_mad_mults) + .5, 1))
        ax.set_xticklabels(self.hit_mad_mults)
        ax.set_xticklabels(ax.get_xmajorticklabels(), rotation=0, fontsize=12)

        # set cbar ticks fonts ize
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=12)

        # saving
        plt.savefig(self.args.step8_save_path / f"{fig_save_name}.png", bbox_inches='tight', dpi=400)
        plt.close(fig)
        plt.cla()
        # plt.show()

