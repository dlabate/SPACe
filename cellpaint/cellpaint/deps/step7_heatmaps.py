import time
import types
from tqdm import tqdm

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import multiprocessing as mp
from functools import partial

from cellpaint.utils.shared_memory import MyBaseManager, TestProxy
from cellpaint.utils.img_files_dep import unique_with_preservered_order, find_first_occurance
from cellpaint.utils.post_feature_extraction import FeaturePreprocessing

pd.options.mode.chained_assignment = None

import cmcrameri as cmc


class FeatureHeatMap(FeaturePreprocessing):
    # if self.dist_type == "bimodality-index":
    #     vmin = 0
    #     vmax = 1
    # # elif self.dist_type == "alpha-geodes-div-symm":
    # #     vmin = .004
    # #     vmax = .02
    # #     # vmax = np.percentile(self.distmap.values, 99)
    # # elif self.dist_type in ["kurtosis", "skewness"]:
    # #     vmin = -2
    # #     vmax = 2
    #     # vmin = np.percentile(self.distmap.values, 1)
    #     # vmax = np.percentile(self.distmap.values, 99)
    # elif self.dist_type in ["mean", "wasserstein", "median", "mode"]:
    #     vmin = -1
    #     vmax = 1
    # else:
    #     raise ValueError(f"{self.dist_type} does not exist or has not been implemented yet!!!")
    analysis_step = 6
    num_meta = 9
    cmap = "cmc.batlow"

    color_pass_control = "blue"
    color_pass_test = "black"
    color_fail = "red"

    # font sizes that do not depend on the number of wells
    title_fontsize = 15
    xtitle_top_fontsize = 13
    xsubtitle_top_fontsize = 10
    xlabel_bottom_fontsize = 2

    nrow_col = "#rows"
    nrow_font_col = "font_size"

    def __init__(self, args):
        FeaturePreprocessing.__init__(self)
        self.args = args

        # self.heatmap_threshold = heatmap_threshold
        self.csv_paths = list(args.step5_save_path.glob("*.csv"))
        self.N = len(self.csv_paths)
        assert self.N >= 1, "Distmaps are missing!!! Please Run Step 4 and Step 5 of cellpaint first!!!"
        self.reset_index_param_dict = {"inplace": False, "drop": True, }

    def get_heatmap(self, ii):
        """Main function
         The functions have to be executed in order, that is why I used steps_single_plate!!!"""
        self.step1_load_heatmap_csv_file(ii)
        self.step2_choose_ylabel_left_font_size()
        self.step3_handle_columns()
        self.step4_create_ylabels_right()
        self.step5_apply_threshold()
        self.step6_create_heatmap_figure_object()

        self.step7_create_ylabels_left()
        self.step8_add_ylabels_right()

        self.step9_create_xlabels_bottom()
        self.step10_create_xlabels_top()

        self.step11_rotate_tick_labels()
        self.step12_adjust_colorbar_location()
        self.step13_save_heatmap_figure(ii)

    @staticmethod
    def join_str_fn(x, delimiter='_'):
        return f"{delimiter}".join(x.astype(str))

    def step1_load_heatmap_csv_file(self, ii):
        # load distance map
        self.distmap = self.compatibility_fn(self.csv_paths[ii])
        ##############################################################################
        self.filename = self.csv_paths[ii].stem

        split = self.filename.split('_')
        # print("filename: ", filename)
        prefix, self.dist_type = '_'.join(split[0:-2]), split[-1]
        if f"{self.args.anchor_treatment}" not in self.filename.lower():
            self.plot_type = 2
            self.main_col = "treatment"
        else:  # DMSO cell-line comparison
            self.plot_type = 3
            self.main_col = "cell-line"

        self.num_wells = len(self.distmap)
        self.plot_type_error = NotImplementedError(f"self.plot_type={self.plot_type} is not implemented yet!!!")

    def step2_choose_ylabel_left_font_size(self, ):
        # print(f"number wells: {self.num_wells}")

        if self.num_wells <= 10:
            self.ylabel_left_fontsize = 11

        elif 10 < self.num_wells <= 20:
            self.ylabel_left_fontsize = 9

        elif 20 < self.num_wells <= 40:
            self.ylabel_left_fontsize = 7

        elif 40 < self.num_wells <= 80:
            self.ylabel_left_fontsize = 5

        elif 80 < self.num_wells <= 144:
            self.ylabel_left_fontsize = 3

        else:
            self.ylabel_left_fontsize = 1

        # self.ylabel_right_fontsize = 4

    def step3_handle_columns(self):
        """get column names which is divided into to sets:
        meta data columns, self.distmap, and feature distance columns, self.feat_cols"""
        # get meta-data columns as well as none-meta/feature-distance columns
        # "cell-count", "exp-id", "well-id", "cell-line", "density", "treatment", "other", "dosage", "well-status"
        self.meta_cols = list(self.distmap.columns)[0:self.num_meta]
        self.feat_cols = list(self.distmap.columns)[self.num_meta:]
        # get cell count for unique treatment, density, dosage triplets
        # exclude the exp-id and well-id columns
        self.vals_in_title = []
        # removing/dropping meta columns (excluding cell-count) that have only a single unique value
        self.yright_cols = ["treatment", "cell-line", "dosage", "density", "other", "well-status", "cell-count"]
        for col in np.setdiff1d(self.yright_cols, ["treatment", "well-status", "cell-count"]):
            unix = np.unique(self.distmap[col].values)
            if len(unix) == 1:
                self.yright_cols.remove(col)
                if col.lower() != "other":
                    self.vals_in_title.append(f"{col}={unix[0]}")
        """This part is a MUST!!!"""
        # resorting the entire dist-map dataframe has to happen here!
        # basically each dataframe has to be sorted by the column used
        # to obtain the self.ylabel_right_w_count
        self.other_cols = list(np.setdiff1d(self.yright_cols, ["cell-count"]))
        if self.main_col in self.other_cols:  # putting treatment or cell-line first
            self.other_cols.remove(self.main_col)
            self.other_cols.insert(0, self.main_col)
        # assuming controls are not empty (There should be at least DMSO there!!!
        assert len(self.args.control_treatments) > 0, \
            "There has to be at least one control treatment like DMSO, but self.args.control_treatments is empty!!!"

    @staticmethod
    def choose_ylabel_right_font_size(num_rows, total_rows, num_cases):
        if total_rows <= 96:
            if num_rows == 1:
                font_size = 4
            elif num_rows == 2:
                font_size = 5
            elif num_rows == 3:
                font_size = 6
            elif 2 < num_rows <= 4:
                font_size = 7
            elif 4 < num_rows <= 10:
                font_size = 9
            else:
                font_size = 12
        else:
            if num_rows == 1:
                font_size = 1
            elif num_rows == 2:
                font_size = 2
            elif num_rows == 3:
                font_size = 3
            elif 2 < num_rows <= 4:
                font_size = 4
            elif 4 < num_rows <= 10:
                font_size = 6
            else:
                font_size = 9

        return font_size

    def step4_create_ylabels_right(self, ):
        self.distmap.sort_values(by=self.other_cols, ascending=[True] * len(self.other_cols), inplace=True)
        ygroups = []
        distmaps = []
        nrows_per_ygroup = []
        # Make sure Control treatments take in top rows when grouping based on cell-count
        if self.plot_type == 2:
            # get number of control treatments, all treatments
            N1, N2, = len(self.args.control_treatments), len(np.unique(self.distmap["treatment"]))
            cond0 = self.distmap["treatment"].to_numpy() == self.args.anchor_treatment
            if N1 == N2 and N1 == 1:
                # There is only a single group: anchor-treatment (DMSO)
                conds = [cond0]
            elif N1 == N2 and N1 > 1:
                # divide into 2 ygroups in order:
                # 1) anchor-treatment
                # 2) control-treatments except anchor
                conds = [cond0, ~cond0]
            elif N1 != N2:
                # divide into 3 ygroups in order:
                # 1) anchor-treatment
                # 2) control-treatments except anchor
                # 3) test-treatments (Not control treatments)
                cond1 = np.isin(self.distmap["treatment"].to_numpy(), self.args.control_treatments)
                conds = [cond0, (~cond0) & cond1, ~cond1]
            else:
                raise ValueError(f"It must be the case that {N1}>=1")

        elif self.plot_type == 3:  # DMSO cell-line comparison
            # divide into 2 ygroups in order:
            # 1) anchor-cellline and 2) rest of cell-line
            N2 = len(np.unique(self.distmap["cell-line"]))
            assert N2 > 1, f"There has to be more than one cell-line for comparison, found {N2}"
            cond0 = self.distmap["cell-line"].to_numpy() == self.args.anchor_cellline
            conds = [cond0, ~cond0]
        else:
            raise self.plot_type_error

        for ii in range(len(conds)):
            distmap_section = self.distmap[conds[ii]].reset_index(drop=True)
            ygroup = distmap_section[self.yright_cols].groupby(self.other_cols)
            ygroups.append(ygroup.agg({"cell-count": "sum"}, axis="rows").reset_index())
            nrows_per_ygroup.append(ygroup.size().reset_index())
            distmaps.append(distmap_section)
            # print(ygroups[ii].shape, nrows_per_ygroup[ii].shape)
            # print(ygroups[ii].head(10))
            # print(nrows_per_ygroup[ii].head(10))
            # print(distmap_section[self.yright_cols].head(20))
            # print('\n')

        self.distmap = pd.concat(distmaps, axis=0).reset_index(drop=True)
        self.ylabel_right_groups = pd.concat(ygroups, axis=0).reset_index(drop=True)

        # We need two things:
        # 1) Total number of rows in self.distmap!!!
        # 2) number of rows in each you_label_right_group
        self.nrows_per_ygroup = pd.concat(nrows_per_ygroup, axis=0).reset_index(drop=True)
        self.nrows_per_ygroup.rename(columns={0: self.nrow_col}, inplace=True)

        self.nrows_per_ygroup[self.nrow_font_col] = self.nrows_per_ygroup[self.nrow_col].apply(
            lambda x: self.choose_ylabel_right_font_size(x, self.num_wells, N2))

    def step5_apply_threshold(self):
        """We apply the intensity based upper and lower
        thresholds for better visibility/readability/ biological interpretability"""

        # specify the thresholds here!
        if self.dist_type == "bimodality-index":
            self.heatmap_threshold = .56
        # elif self.dist_type == "alpha-geodes-div-symm":
        #     self.heatmap_threshold = .0009
        # elif self.dist_type in ["kurtosis", "skewness"]:
        #     self.heatmap_threshold = 2
        elif self.dist_type in ["mean", "wasserstein", "median", "mode"]:
            self.heatmap_threshold = .1
        else:
            raise ValueError(f"{self.dist_type} does not exist or has not been implemented yet!!!")

        # print("self.heatmap_threshold: ", self.dist_type, self.heatmap_threshold)
        # self.distmap only contains the feature distance columns, not any of the meta data columns
        self.distmap[self.feat_cols] = self.distmap[self.feat_cols][
            (self.distmap[self.feat_cols] <= -self.heatmap_threshold) |
            (self.distmap[self.feat_cols] >= self.heatmap_threshold)]

    def step6_create_heatmap_figure_object(self):
        """
        create the matplotlib/sns figure object. We set global values vmin=-1, vmax=1.
        This gives a universal and uniform color gradient to all generated heatmaps, and
        allows for apple to apple comparison of all figure maps."""

        self.fig, (self.ax1, axcb) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 0.06]}, )
        self.fig.set_size_inches(21.3, 13.3)
        # self.fig.set_size_inches(54.1, 33.8 )
        # TODO: Fix the title
        self.fig.suptitle(
            f"{self.args.experiment}:{self.dist_type} {'  '.join(self.vals_in_title)}\n"
            f"fail wells color={self.color_fail}      control wells color={self.color_pass_control}",
            fontname='Comic Sans MS',
            fontsize=self.title_fontsize)
        # # set lower and upper thresholds for heatmap values for better display
        vmin = -1
        vmax = 1

        # print(self.dist_type, vmin, vmax)
        self.heatmap = sns.heatmap(
            self.distmap[self.feat_cols].to_numpy(),
            # linewidths=.05,
            # linecolor='black',
            cmap=self.cmap,
            ax=self.ax1,
            cbar_ax=axcb,
            vmin=vmin, vmax=vmax,
            cbar_kws={"shrink": 0.4})

    def step7_create_ylabels_left(self):
        # predefine/preset y-axis labels
        # print("index inside step6_create_ylabels_left func: ", self.distmap.index)
        self.ylabels_left = list(self.distmap['well-id'] + "_#" + self.distmap['cell-count'].astype(str))
        # set y ticks positions and values
        self.heatmap.set_yticks(np.arange(.5, len(self.ylabels_left) + .5, 1))
        self.heatmap.set_yticklabels([it.lower() for ii, it in enumerate(self.ylabels_left) if (ii % 1) == 0])
        self.heatmap.set_yticklabels(self.heatmap.get_ymajorticklabels(), fontsize=self.ylabel_left_fontsize)

        for jj, tick_label in enumerate(self.ax1.axes.get_yticklabels()):
            tick_label.set_color("black" if self.distmap.iloc[jj]["well-status"] == "pass" else "red")
            # tick_label.set_fontsize("15")

    def step8_add_ylabels_right(self):
        """
        Set treatment/cell-line labels on right side of the plot for better readability.
        Group over all metacols with more than 1 unique value, except the "cell-count" column itself.
        """
        assert "cell-count" in self.yright_cols

        # df_groups = self.distmap[self.yright_cols].groupby(self.other_cols).agg(['sum']).reset_index()
        # excluding the "cell-count" column, because it got aggregated and can't be used as reference
        ylabel_right_wo_count = self.ylabel_right_groups[self.other_cols].apply(self.join_str_fn, axis=1).to_list()
        ylabel_right_all_rows = self.distmap[self.other_cols].apply(self.join_str_fn, axis=1).to_list()
        # horizontal lines/stops locations
        self.hstops = [ylabel_right_all_rows.index(it) for it in ylabel_right_wo_count] + [len(ylabel_right_all_rows)]
        # for it in ylabel_right_wo_count:
        #     print(f"{it}      {ylabel_right_all_rows.index(it)}")
        #     print('\n')
        ##########################################################################
        # # labels for the bands, list of places where the bands start and stop
        # create a loop using the begin and end of each band, the colors and the labels
        cols = list(self.ylabel_right_groups.columns)
        # We won't include well-status in the text_line bc it will look busy!!!
        # rather we would use it's value to color the text_line.
        cols.remove("well-status")
        M = len(self.ylabel_right_groups.index.to_list())
        assert self.ylabel_right_groups.index.to_list() == pd.RangeIndex(start=0, stop=M, step=1).to_list()
        for ii, row1 in self.ylabel_right_groups.iterrows():
            # for ii, (beg, end, label) in enumerate(zip(self.hstops[:-1], self.hstops[1:], self.ylabel_right_w_count)):
            font_size = self.nrows_per_ygroup.iloc[ii][self.nrow_font_col]
            if self.plot_type == 2:
                # specify text color
                if row1["well-status"] == "fail":
                    color = self.color_fail
                elif (row1["well-status"] == "pass") & np.isin(row1["treatment"], self.args.control_treatments):
                    color = self.color_pass_control
                else:
                    color = self.color_pass_test

                # specify text
                cond = self.ylabel_right_groups["treatment"] == row1["treatment"]
                row1["treatment"] = row1["treatment"][0:self.args.max_chars]
                if ("dosage" in self.ylabel_right_groups.columns) and \
                        (len(np.unique(self.ylabel_right_groups.loc[cond, "dosage"])) == 1):
                    text_line = row1[np.setdiff1d(cols, "dosage")].to_list()
                    text_line = '_'.join([str(it) for it in text_line])
                else:
                    # row1["treatment"] = row1["treatment"][0:self.args.max_chars]
                    text_line = row1[cols].to_list()
                    text_line = '_'.join([str(it) for it in text_line])

            elif self.plot_type == 3:  # DMSO cell-line comparison
                if row1["well-status"] == "fail":
                    color = self.color_fail
                elif (row1["well-status"] == "pass") & (row1["cell-line"] == self.args.anchor_cellline):
                    color = self.color_pass_control
                else:
                    color = self.color_pass_test
                text_line = row1[cols].to_list()
                text_line = '_'.join([str(it) for it in text_line])
            else:
                raise self.plot_type_error

            # add some text_line to the center right of this band
            nrows = self.nrows_per_ygroup.iloc[ii][self.nrow_col]
            y_beg = self.hstops[ii]
            y_end = self.hstops[ii + 1]
            if self.num_wells <= 120 and nrows < 4:
                adjust = -.1
            elif self.num_wells > 120 and nrows < 4:
                adjust = -.2
            else:
                adjust = 0
            yloc = (y_beg + y_end) / 2 + adjust
            self.heatmap.text(
                1.01, yloc,
                text_line,
                ha='left', va='center',
                transform=self.heatmap.get_yaxis_transform(),
                size=font_size,
                color=color)
        # add_horizontal_gridlines
        for stop in self.hstops:
            # y specifies where the line's starting point on the y-axis
            # xmin and xmax specify how long the line is (from where to where)
            self.heatmap.hlines(y=stop, xmin=0, xmax=len(self.feat_cols),
                                linestyle='dashed', linewidth=.4, color="black")

    def step9_create_xlabels_bottom(self):
        """
        self.xlabels are basically all feature names put in a list:
        shape feature names,
        followed by intensity feature names,
        followed by moment feature names,
        followed by haralick feature names,
        """

        # predefine/preset x-axis labels
        # self.xtitles = ["All", "Nucleus", "Cyto", "Nucleoli", "Actin", "Mito"]
        self.xtitles = unique_with_preservered_order([it.split('_')[0] for it in self.feat_cols])
        self.xsubtitles = unique_with_preservered_order([
            '_'.join(it.split('_')[0:2])
            for it in self.feat_cols
            if it.split('_')[0] in self.xtitles[1:]])
        # in ["Nucleus", "Cyto", "Nucleoli", "Actin", "Mito"] not including "All"

        # remove channel name and feature category from the feature-distance column names
        # self.xlabels_bottom = []
        # sub_feature_compartments = ["All_"]+[f"{it}_" for it in self.args.organelles]
        # for cin in sub_feature_compartments:
        #     self.xlabels_bottom += ["".join(it.split('_')[2:]) for it in self.feat_cols if cin in it]
        self.xlabels_bottom = ["".join(it.split('_')[2:]) for it in self.feat_cols]

        # set x ticks positions and values
        self.heatmap.set_xticks(np.arange(.5, len(self.xlabels_bottom) + .5, 1))
        self.heatmap.set_xticklabels([it for ii, it in enumerate(self.xlabels_bottom) if ii % 1 == 0])
        self.heatmap.set_xticklabels(self.heatmap.get_xmajorticklabels(), fontsize=self.xlabel_bottom_fontsize)

    def step10_create_xlabels_top(self):
        self.xtitle_locs = [find_first_occurance(self.feat_cols, cin) for cin in self.xtitles]
        self.xsubtitle_locs = [find_first_occurance(self.feat_cols, cin) for cin in self.xsubtitles]
        ############################################################################################
        # add vertical lines
        hh = list(np.arange(0, len(self.ylabels_left), 1))
        for x_ in self.xsubtitle_locs:
            # x specifies where the line's starting point on the y-axis
            # xmin and xmax specify how long the line is (from where to where)
            self.heatmap.vlines(x=x_, ymin=0, ymax=len(hh), linestyle='dashed', linewidth=.4, color="gray")
        for x_ in self.xtitle_locs:
            # x specifies where the line's starting point on the y-axis
            # xmin and xmax specify how long the line is (from where to where)
            self.heatmap.vlines(x=x_, ymin=0, ymax=len(hh), linestyle='solid', linewidth=.6, color="black")
        ###############################################################################################
        self.xtitle_locs.append(len(self.feat_cols))
        # add text for sub-categories of features above the figure object
        for beg, end, label in zip(self.xtitle_locs[:-1], self.xtitle_locs[1:], self.xtitles):
            # add some text to the center right of this band
            self.heatmap.text((beg + end) / 2, 1.08,
                              # '\n'.join(label.split())
                              label,
                              va='top', ha='center',
                              transform=self.heatmap.get_xaxis_transform(), size=self.xtitle_top_fontsize, )
        ###################################################################################################
        self.xsubtitle_locs.append(len(self.feat_cols))
        # add text for sub-categories of features above the figure object
        for beg, end, label in zip(self.xsubtitle_locs[:-1], self.xsubtitle_locs[1:], self.xsubtitles):
            # add some text to the center right of this band
            self.heatmap.text((beg + end) / 2, 1.05,
                              # '\n'.join(label.split())
                              label.split('_')[1],
                              va='top', ha='center',
                              transform=self.heatmap.get_xaxis_transform(), size=self.xsubtitle_top_fontsize, )

    def step11_rotate_tick_labels(self, ):
        """rotate the tick labels 90 degrees (correctly)"""
        self.heatmap.set_xticklabels(self.heatmap.get_xticklabels(), rotation=90)
        self.heatmap.set_yticklabels(self.heatmap.get_yticklabels(), rotation=0)

    def step12_adjust_colorbar_location(self, ):
        """move the colorbar to the right to allow more space for ylabels_right"""
        plt.subplots_adjust(wspace=.32)

    def step13_save_heatmap_figure(self, ii):
        """save the matplotlib.pyplot figure object to disk"""
        plt.savefig(self.args.step6_save_path / f"{self.csv_paths[ii].stem}_{self.heatmap_threshold}.png",
                    bbox_inches='tight', dpi=400)
        # plt.show()
        plt.close(self.fig)


def step6_run_single_process_for_loop(args):
    """heatmap generation loop:
    Using a single process when there are less than 6 heatmaps in the self.args.step6_save_path  folder."""
    inst = FeatureHeatMap(args)
    N = len(inst.csv_paths)
    for ii in range(0, N):
        print(ii)
        inst.get_heatmap(ii)


def step6_run_multi_process_for_loop(args):
    """heatmap generation loop:
    Using a multi-processing for loop when there are more than 6 heatmaps in the self.args.step6_save_path  folder."""
    MyManager = MyBaseManager()
    # register the custom class on the custom manager

    # test_proxy = create_proxy(HeatMap)
    # MyManager.register('HeatMap', HeatMap, test_proxy)

    # MyManager.register('HeatMap', HeatMap, TestProxy2(HeatMap))

    MyManager.register('FeatureHeatMap', FeatureHeatMap, TestProxy)
    # create a new manager instance
    with MyManager as manager:
        inst = manager.FeatureHeatMap(args)
        num_proc = min(mp.cpu_count(), inst.N)
        with mp.Pool(processes=num_proc) as pool:
            for _ in tqdm(pool.imap(inst.get_heatmap, np.arange(inst.N)), total=inst.N):
                pass


def step6_main_run_loop(args):
    """
    Main function for cellpaint step V (currently it is the last step).
        It generate heatmaps using a loop over an instance of the HeatMap class.
        If there are more than 6 heatmaps, it uses a multi-processing for loop.

        It saves each resulting heat-maps into separate png files as:
            self.args.step6_save_path / f"{self.csv_paths[ii].stem}_{self.heatmap_threshold}.png"

        where

        if args.mode.lower() == "debug":
            self.args.step6_save_path = args.main_path / args.experiment / "Debug" / "HeatMaps"
        elif args.mode.lower() == "test":
            self.args.step6_save_path = args.main_path / args.experiment / "Test" / "HeatMaps"
        elif args.mode.lower() == "full":
            self.args.step6_save_path = args.main_path / args.experiment / "HeatMaps"

    """
    s_time = time.time()
    print("Cellpaint Step 6: Generating Heat-maps for all DistanceMap Generated in Cellpaint Step 5...")
    # stepV_run_single_process_for_loop(args)
    if args.mode == "debug":
        step6_run_single_process_for_loop(args)
    else:
        step6_run_multi_process_for_loop(args)
    print(f"Finished Cellpaint step 6 in: {(time.time() - s_time) / 3600} hours")
    print("***********************")
