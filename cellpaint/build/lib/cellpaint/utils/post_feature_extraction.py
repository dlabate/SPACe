import re
import sys
import time
import string

import numpy as np
import pandas as pd
import scipy.stats as scstats
from scipy.integrate import simpson
from functools import partial, lru_cache
from sklearn.preprocessing import robust_scale, RobustScaler

import torch


import seaborn as sns
import cmcrameri as cmc
import matplotlib.pyplot as plt
import matplotlib.colors as matcolors

from cellpaint.steps_single_plate.step0_args import Args


class FeaturePreprocessing:
    # handling a single gpu, gpu 0, memory
    device_id = 0
    device = torch.device(f"cuda:{device_id}") if torch.cuda.is_available() else torch.device(f"cpu")
    num_groups_ub = 384  # maximum number of groups possible in a 16*24=384-wells plate

    @staticmethod
    @lru_cache(maxsize=1)
    def load_and_preprocess_features(load_path, min_well_cell_count):
        # TODO: Get number of features at each stage to get some sort of summary stats for quality control.
        s_time = time.time()
        for ii in range(5):
            if not (load_path / f"w{ii}_features.csv").is_file():
                raise ValueError(f"{load_path}/w{ii}_features.csv file does not exist! Make sure"
                                 f"to run Step 3 of the Cellpaint first!!!")

        w0_features = pd.read_csv(load_path / "w0_features.csv")
        w1_features = pd.read_csv(load_path / "w1_features.csv")
        w2_features = pd.read_csv(load_path / "w2_features.csv")
        w3_features = pd.read_csv(load_path / "w3_features.csv")
        w4_features = pd.read_csv(load_path / "w4_features.csv")
        misc_features = pd.read_csv(load_path / "misc_features.csv")
        metadata = pd.read_csv(load_path / "metadata_of_features.csv")
        ##########################################################################################################

        # automatically calculates when meta_cols end and feature cols starts, i.e. self.start_index
        # The first 4 feature columns for each channel are bounding boxes.
        features = pd.concat([
            metadata,
            misc_features,
            w0_features[w0_features.columns[4:]],
            w1_features[w1_features.columns[4:]],
            w2_features[w2_features.columns[4:]],
            w3_features[w3_features.columns[4:]],
            w4_features[w4_features.columns[4:]]],
            axis=1)
        # removing rows/cells with no nucleoli
        features = features.loc[features["has-nucleoli"] == 1]
        # removing Nans
        features.dropna(axis=0, inplace=True)
        #########################################################
        # If I changed the name of the experiment folder after feature extraction
        if "exp-id" in list(features.columns):
            features["exp-id"] = load_path.parents[0].stem
        #############################################################################################################
        # For compatibility with the older versions of the code
        # print("load_path: ", load_path, load_path.parents[0], load_path.parents[1])
        if "exp-id" not in list(features.columns):
            experiment = load_path.parents[0].stem
            features.insert(0, "exp-id", experiment)
        features.drop([it for it in list(features.columns) if "Haralicks-Global" in it or "Moments" in it],
                      axis=1,
                      inplace=True)
        for it in features.columns:
            if "Haralicks-Local" in it:
                features.rename(columns={it: it.replace("Haralicks-Local", "Haralicks")}, inplace=True)
        features["treatment"] = features["treatment"].astype(str)
        features["cell-line"] = features["cell-line"].astype(str)
        ##############################################################################################################
        # remove wells with not enough cells
        cell_counts = features["well-id"].value_counts().reset_index()
        cell_counts.rename(columns={"count": "cell-count"}, inplace=True)
        low_cell_count_well_ids = (cell_counts["well-id"].loc[
            cell_counts["cell-count"] < min_well_cell_count]).to_list()
        features = features.loc[~np.isin(features["well-id"], low_cell_count_well_ids)]
        # for it in features.columns:
        #     print(it)
        start_index = metadata.shape[1]
        # size_mb = np.round(sys.getsizeof(features) / 1024 / 1024, 2)
        # print(f"features size in MB={size_mb}  shape={features.shape}")
        e_time = time.time()
        return features, cell_counts, start_index

    @staticmethod
    def normalize_features(features, features_start_index, quantile_range):
        """Normalizes the feature columns of the features dataframe.
        features is a pandas dataframe of shape (Num_cells, NumCols)
        features[:, 0:features_start_index] has the metadata associated with the experiment
        features[:, features_start_index:] has the feature values associated with the experiment
        """
        # normalize each feature column (NOT metadata columns)
        feature_cols = list(features.columns[features_start_index:])
        features[feature_cols] = robust_scale(features[feature_cols].to_numpy(), quantile_range=quantile_range)
        # clip outlier feature values to -1 and 1
        features[feature_cols] = features[feature_cols].clip(lower=-1, upper=1)
        return features

    @staticmethod
    def normalize_features_transformer(feats_train, feats_test, features_start_index, quantile_range):
        """Normalizes the feature columns of the features dataframe.
        features is a pandas dataframe of shape (Num_cells, NumCols)
        features[:, 0:features_start_index] has the metadata associated with the experiment
        features[:, features_start_index:] has the feature values associated with the experiment
        """
        # normalize each feature column (NOT metadata columns)
        feature_cols = list(feats_test.columns[features_start_index:])
        transformer = RobustScaler(quantile_range=quantile_range)
        transformer.fit(feats_train[feature_cols].to_numpy())
        feats_test[feature_cols] = transformer.transform(feats_test[feature_cols].to_numpy())
        # clip outlier feature values to -1 and 1
        feats_test[feature_cols] = feats_test[feature_cols].clip(lower=-1, upper=1)
        return feats_test

    @staticmethod
    def get_anchor_features(features, expid, anchor_treatment, anchor_other, anchor_col, anchor_col_val):
        """anchor_treatment: anchor treatment column value"""
        # cond0 = (features["exp-id"] == expid)
        # cond1 = (features["treatment"] == anchor_treatment)
        # cond2 = (features["other"] == anchor_other)
        # cond3 = (features[anchor_col] == anchor_col_val)
        # print(f"exp: {np.sum(cond0)} "
        #       f"treatment: {np.sum(cond1)} "
        #       f"other: {np.sum(cond2)} "
        #       f"{anchor_col}: {np.sum(cond3)} "
        #       f"exp & treatment: {np.sum(cond0&cond1)} "
        #       f"exp & treatment & other: {np.sum(cond0&cond1&cond2)} "
        #       f"exp & treatment & {anchor_col}: {np.sum(cond0&cond1&cond3)} "
        #       )
        return features[
            (features["exp-id"] == expid) &
            (features["treatment"] == anchor_treatment) &  # 99% of the time it is equal to DMSO
            (features["other"] == anchor_other) &
            (features[anchor_col] == anchor_col_val)]

    @staticmethod
    def compatibility_fn(distmap_csv_file_name):
        # for compatibility with previous versions
        distmap = pd.read_csv(distmap_csv_file_name)
        if "has-nucleoli" in distmap.columns:
            distmap.drop("has-nucleoli", axis=1, inplace=True)
        if "exp-id" in list(distmap.columns):
            distmap["exp-id"] = distmap_csv_file_name.parents[1].stem
        distmap["treatment"] = distmap["treatment"].astype(str)
        distmap["cell-line"] = distmap["cell-line"].astype(str)
        return distmap

    @staticmethod
    def get_cell_count_per_well(features):
        groups = features["well-id"].value_counts().reset_index()
        groups.rename(columns={"well-id": "cell-count"}, inplace=True)
        groups.rename(columns={"index": "well-id"}, inplace=True)
        return groups

    @staticmethod
    def get_cell_count_per_fov(features):
        raise NotImplementedError("W8 for it!!! Maybe!!! In the long distance future .... HAHA HAHA ...")
        # groups = features["well-id"].value_counts().reset_index()
        # groups.rename(columns={"well-id": "cell-count"}, inplace=True)
        # groups.rename(columns={"index": "well-id"}, inplace=True)
        # return groups

    @staticmethod
    def swap_df_anchor_row(df, old_idx, new_idx):
        """
        https://stackoverflow.com/questions/46890972/swapping-rows-within-the-same-pandas-dataframe
        https://stackoverflow.com/questions/21800169/python-pandas-get-index-of-rows-where-column-matches-certain-value

        df: if df=metadata_unix then shape=(num_unix, 7) elif df=distmap then shape=(num_unix, num_meta+num_features)
        """
        # anchor_idx = df.index[df[anchor_col] == anchor_val].tolist()[0]
        b, c = df.iloc[old_idx], df.iloc[new_idx]

        temp = b.copy()
        df.iloc[old_idx] = c
        df.iloc[new_idx] = temp

        return df

    @staticmethod
    def swap_np_anchor_row(np_arr, old_idx, new_idx, axis):
        """
        https://stackoverflow.com/questions/54069863/swap-two-rows-in-a-numpy-array-in-python

        np_arr: if roc_curves: shape=(3, 5, num_unix, 100) elif rocaucs shape=(3, 5, num_unix)
        """
        if axis == 0:
            np_arr[[old_idx, new_idx]] = np_arr[[new_idx, old_idx]]
        elif axis == 1:
            np_arr[:, [old_idx, new_idx]] = np_arr[:, [new_idx, old_idx]]
        elif axis == 2:
            np_arr[:, :, [old_idx, new_idx]] = np_arr[:, :, [new_idx, old_idx]]
        elif axis == 3:
            np_arr[:, :, :, [old_idx, new_idx]] = np_arr[:, :, :, [new_idx, old_idx]]
        elif axis == 4:
            np_arr[:, :, :, :, [old_idx, new_idx]] = np_arr[:, :, :, :, [new_idx, old_idx]]
        else:
            NotImplementedError()
        return np_arr

    @staticmethod
    def shift_up_control_treatments(df, control_treatments, anchor_treatment):
        if "treatment" in list(df.columns):
            cond0 = np.isin(df["treatment"].to_numpy(), control_treatments)
            if len(cond0) > 0:
                assert np.isin(anchor_treatment, control_treatments)
                df_test = df.loc[~cond0]
                df_ctrl = df.loc[cond0]
                cond1 = df_ctrl["treatment"] == anchor_treatment
                df_ctrl_anchor = df_ctrl.loc[cond1]
                df_ctrl_not_anchor = df_ctrl.loc[~cond1]
                # Clear the existing index and reset it in the result by setting the ignore_index option to True
                df = pd.concat([df_ctrl_anchor, df_ctrl_not_anchor, df_test], axis=0, ignore_index=True)
        return df


class ROCAUC:
    """Receiving Operator Characteristic Curve (ROC Curve) and ROCAUC (ROC Area Under the Curve)"""

    colormap = "cmc.batlow"  # "magma", "jet", "rocket_r", "tab20"
    # colors = ["red", "green", "blue", "orange", "purple", "pink", "gray", "black"]

    thresholds = np.linspace(0, 1, 100)
    num_thresholds = len(thresholds)

    device_id = 0
    device = torch.device(f"cuda:{device_id}") if torch.cuda.is_available() else torch.device("cpu:0")

    # these two arguments should be specified in the child classes!!!
    analyis_type = None
    save_path = None
    qc_label_col = "QC-label"

    # if analysis_type == 1:  # "Quality_Control"
    #     print("Quality Control Module Begins... ")
    #     save_path = args.step4_save_path
    # elif analysis_type == 2:  # "Final_ROC_CURVE"
    #     save_path = self.args.step7_save_path
    #     print("Final Derivative Map and AUC Map Module Begins... ")
    # elif analysis_type == 3:  # "Final_ROC_CURVE"
    #     save_path = self.args.step7_save_path
    #     print("Final Derivative Map and AUC Map Module Begins... ")
    # else:
    #     raise NotImplementedError()

    def __init__(self, args):
        # Unfortunately, we can't know the plot_type variable value in advance
        self.args = args

        self.feature_types = self.args.heatmap_feature_groups[1:]
        self.channels = self.args.organelles
        self.num_feat_cat = len(self.feature_types)
        self.num_cin = len(self.channels)

        self.xlabels = [f"{it0}_{it1}" for it0 in self.feature_types for it1 in self.channels]

    def compute_roccurve_and_rocauc(self, distance_map, metadata_df, plot_type):
        num_curves = metadata_df.shape[0]
        assert metadata_df.index.to_list() == pd.RangeIndex(start=0, stop=num_curves, step=1).to_list()
        assert distance_map.shape[0] == metadata_df.shape[0]
        # print(distance_map.shape, metadata_df.shape, plot_type, hit_calling)

        roc_curves = np.zeros((self.num_feat_cat, self.num_cin, num_curves, self.num_thresholds), dtype=np.float32)
        roc_aucs = np.zeros((self.num_feat_cat, self.num_cin, num_curves), dtype=np.float32)

        roc_diff_curves = np.zeros((self.num_feat_cat, self.num_cin, num_curves, self.num_thresholds), dtype=np.float32)
        roc_diff_aucs = np.zeros((self.num_feat_cat, self.num_cin, num_curves), dtype=np.float32)
        for ii, it1 in enumerate(self.feature_types):
            for jj, it2 in enumerate(self.channels):
                feat_cols = [cc for cc in distance_map.columns if it1 in cc and it2 in cc]
                assert len(feat_cols) > 0

                for kk, row in metadata_df.iterrows():  # restrict to a specific (exp-id, treatment, dosage) triplet
                    # vals = metadata_df[kk] if isinstance(metadata_df[kk], np.ndarray)
                    # else np.array([metadata_df[kk]], dtype=object)
                    # tmp = distance_map.loc[self.get_df_at_cond(distance_map, meta_cols, vals), feat_cols].values
                    if plot_type == 1:
                        tmp = np.abs(distance_map.iloc[kk][feat_cols].to_numpy())
                    elif plot_type == 2:
                        cond = (distance_map["exp-id"] == row["exp-id"]) & \
                               (distance_map["treatment"] == row["treatment"]) & \
                               (distance_map["dosage"] == row["dosage"]) & \
                               (distance_map["well-id"] == row["well-id"])
                        tmp = np.abs(distance_map.loc[cond, feat_cols].to_numpy())

                    elif plot_type == 3:
                        cond = (distance_map["exp-id"] == row["exp-id"]) & \
                               (distance_map["cell-line"] == row["cell-line"]) & \
                               (distance_map["well-id"] == row["well-id"])
                        tmp = np.abs(distance_map.loc[cond, feat_cols].to_numpy())

                    elif plot_type == 4:
                        cond = (distance_map["exp-id"] == row["exp-id"]) & \
                               (distance_map["cell-line"] == row["cell-line"]) & \
                               (distance_map["treatment"] == row["treatment"]) & \
                               (distance_map["well-id"] == row["well-id"])
                        tmp = np.abs(distance_map.loc[cond, feat_cols].to_numpy())

                    else:
                        raise NotImplementedError()
                    if len(tmp) == 0:
                        continue

                    for ll, it4 in enumerate(self.thresholds):
                        tmp1 = tmp.copy()
                        tmp1[tmp <= it4] = 0
                        roc_curves[ii, jj, kk, ll] = np.nansum(tmp1) / tmp1.shape[0] if tmp1.size > 0 else 0
                        # roc_curves[0, ii, jj, kk, ll] = np.sum((tmp1 != 0))/tmp1.size if tmp1.size>0 else 0
                    roc_aucs[ii, jj, kk] = simpson(roc_curves[ii, jj, kk, :], dx=1)
                    roc_diff_curves[ii, jj, kk] = np.abs(np.gradient(roc_curves[ii, jj, kk, :]))
                    roc_diff_aucs[ii, jj, kk] = simpson(roc_diff_curves[ii, jj, kk, :], dx=1)
        return roc_curves, roc_aucs, roc_diff_curves, roc_diff_aucs

    def plot_rocauc_heatmap(self, rocaucs, metadata_df, cell_count_key,
                            plot_type, is_multi_dose, title, savename, hit_col):
        """
        rocaucs.shape = (self.num_feat_cat, self.num_cin, num_curves/num_well_ids)

        https://stackoverflow.com/questions/29647749/seaborn-showing-scientific-
        notation-in-heatmap-for-3-digit-numbers
        https://stackoverflow.com/questions/10388462/matplotlib-different-size-subplots
        https://stackoverflow.com/questions/69347136/how-to-remove-color-bar-in-seaborn-heatmap
        https://stackoverflow.com/questions/69409099/move-colorbar-closer-to-heatmap-seaborn
        https://stackoverflow.com/questions/16419670/increase-distance-between-title-and-plot-in-matplolib
        https://stackoverflow.com/questions/6541123/improve-subplot-size-spacing-with-many-subplots
        https://stackoverflow.com/questions/37233108/seaborn-change-font-size-of-the-colorbar
        https://stackoverflow.com/questions/72660993/
        change-seaborn-heatmap-y-ticklabels-font-color-for-alternate-labels
        """
        # print(rocaucs.shape, metadata_df.shape)
        assert rocaucs.shape[2] == metadata_df.shape[0]
        assert self.num_feat_cat == rocaucs.shape[0]

        title_fs, subtitle_fs, annot_fs, \
        ylabel_fs, xlabel_fs, cbar_label_fs, \
        subplots_width_ratios, subplot_wspace, subplot_top_space, cbar_pad, linewidth = \
            self.plot_props_part2(len(metadata_df), self.num_feat_cat)
        _, _, _, _, triplets = self.plot_props_part1(len(metadata_df))
        props_df, title = self.plot_props_part3(metadata_df, title, triplets, plot_type, is_multi_dose, hit_col)
        # cols = ["legend_or_ylabel_text", "legend_or_ylabel_color", "curve_color", "curve_zorder"]

        min_cval, max_cval = np.nanmin(rocaucs, axis=(1, 2)), np.nanmax(rocaucs, axis=(1, 2))
        fig, axes = plt.subplots(1, self.num_feat_cat + 1, width_ratios=subplots_width_ratios)
        fig.set_size_inches(21.3, 13.3)
        fig.subplots_adjust(wspace=subplot_wspace, top=subplot_top_space)

        fig.suptitle(title, fontname='Comic Sans MS', fontsize=title_fs)
        cbar_props = {
            "orientation": "horizontal",
            "shrink": 0.8,
            "use_gridspec": False,
            "pad": cbar_pad,}
        ######################################################################
        cell_count = metadata_df[cell_count_key].to_numpy()[:, np.newaxis].astype(np.int64)
        # print(metadata_df.columns)
        sns.heatmap(
            cell_count, cmap=self.colormap,
            annot=True, fmt='g', annot_kws={"fontsize": annot_fs},
            ax=axes[0],
            vmin=np.nanmin(cell_count), vmax=np.nanmax(cell_count),
            linecolor='gray', linewidth=linewidth,
            cbar_kws=cbar_props)

        axes[0].set_title(cell_count_key, fontname='Comic Sans MS', fontsize=subtitle_fs-1, pad=1)
        axes[0].set_yticks(np.arange(.5, len(metadata_df) + .5, 1))
        axes[0].set_yticklabels(props_df["legend_or_ylabel_text"].to_list())
        axes[0].set_yticklabels(axes[0].get_ymajorticklabels(), rotation=0, fontsize=ylabel_fs)

        axes[0].set_xticks([])
        axes[0].set_xticklabels([])
        # set cbar ticks font-size
        cbar = axes[0].collections[0].colorbar
        cbar.ax.tick_params(labelsize=cbar_label_fs - 1)
        # change y-axis tick colors depending on outlier/control_treatment/anchor_cellline
        for jj, tick_label in enumerate(axes[0].axes.get_yticklabels()):
            tick_label.set_color(props_df.iloc[jj]["legend_or_ylabel_color"])
            # tick_label.set_fontsize("15")

        for ii in range(self.num_feat_cat):
            sns.heatmap(
                rocaucs[ii].T, cmap=self.colormap,
                annot=True, annot_kws={"fontsize": annot_fs},
                ax=axes[ii+1],
                vmin=min_cval[ii], vmax=max_cval[ii],
                linecolor='gray', linewidth=linewidth,
                cbar_kws=cbar_props)
            axes[ii+1].set_title(self.feature_types[ii], fontname='Comic Sans MS', fontsize=subtitle_fs, pad=1)
            axes[ii+1].set_yticks([])
            axes[ii+1].set_xticks(np.arange(.5, self.num_cin + .5, 1))
            axes[ii+1].set_xticklabels(self.channels)
            axes[ii+1].set_xticklabels(axes[ii+1].get_xmajorticklabels(), rotation=90, fontsize=xlabel_fs)

            # set cbar ticks fontsize
            cbar = axes[ii+1].collections[0].colorbar
            cbar.ax.tick_params(labelsize=cbar_label_fs)

        plt.savefig(self.save_path / f"{savename}.png", bbox_inches='tight', dpi=400)
        # plt.show()
        plt.close(fig)
        plt.cla()

    def plot_roccurves(self, roc_curves, metadata_df, plot_type, is_multi_dose, title, savename, hit_col):
        """min_yaxis, max_yaxis, and max_nonz_xval are calculated per feature category, which are:
        shapes, intensities, and haralick"""
        assert roc_curves.shape[2] == metadata_df.shape[0]

        markersize, ncols, bbox_to_anchor, legend_font_size, triplets = self.plot_props_part1(len(metadata_df))
        # print(type(metadata_df), type(title), type(triplets), type(plot_type), type(is_multi_dose))
        props_df, title = self.plot_props_part3(metadata_df, title, triplets, plot_type, is_multi_dose, hit_col)
        # cols = ["legend_or_ylabel_text", "legend_or_ylabel_color", "curve_color", "curve_zorder"]

        min_yaxis, max_yaxis = np.nanmin(roc_curves, axis=(1, 2, 3)), np.nanmax(roc_curves, axis=(1, 2, 3))
        M, N = len(self.feature_types), len(self.channels)
        max_nonz_xval = np.zeros((M,), dtype=object)
        fig, axes = plt.subplots(M, N)
        fig.set_size_inches(21.3, 13.3)
        fig.suptitle(title, fontname="Comic Sans MS", fontsize=16)

        for ii, it1 in enumerate(self.feature_types):  # rows
            max_nonz_xval[ii] = []
            axes[ii, 0].set_ylabel(it1, fontname="Comic Sans MS", fontsize=16)
            for jj, it2 in enumerate(self.channels):  # columns
                if ii == 0:
                    axes[ii, jj].set_title(f"{it2}", **self.args.csfont)
                for kk, it3 in metadata_df.iterrows():  # (curves or well-ids)
                    axes[ii, jj].plot(
                        self.thresholds,
                        roc_curves[ii, jj, kk],
                        label=props_df.iloc[kk]["legend_or_ylabel_text"],
                        marker=triplets[kk][0],
                        linestyle=triplets[kk][1],
                        color=props_df.iloc[kk]["curve_color"],
                        markersize=markersize,
                        zorder=props_df.iloc[kk]["curve_zorder"],
                    )

                    axes[ii, jj].set_ylim([min_yaxis[ii], max_yaxis[ii]])
                    zids = np.where(np.abs(roc_curves[ii, jj, kk]) < 1e-5)[0]
                    if len(zids) > 0:
                        max_nonz_xval[ii].append(zids[0])

        # max_nonz_xval is the index of x after which all entries for all curves are zero!!!
        for ll, it1 in enumerate(self.feature_types):
            x_max_id = np.nanmax(max_nonz_xval[ll]) if len(max_nonz_xval[ll]) > 0 else -1
            for kk, it2 in enumerate(self.channels):
                axes[ll, kk].set_xlim([0, self.thresholds[x_max_id]])

        leg = plt.legend(
            loc="lower center",
            bbox_to_anchor=bbox_to_anchor,
            ncols=ncols,
            borderaxespad=0.1,
            fancybox=False,
            shadow=False,
            prop={'size': legend_font_size, 'family': 'Consolas'})
        # set the color and text of legend
        for ii, leg_text in enumerate(leg.get_texts()):
            plt.setp(leg_text, color=props_df.iloc[ii]["legend_or_ylabel_color"])

        # Create a big subplot that contains the main plot, to add a single x-axis title and a single y-axis title,
        # to all subplots simultaneously!!!!
        ax = fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axes
        ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        # Use argument `labelpad` to move label downwards or leftward.
        ax.set_xlabel(f"Threshold below which entries are zeroed out", labelpad=9, fontname="Cambria", fontsize=16)
        ax.set_ylabel(f"Sum of none-zero entries in Wasserstein distance-map",
                      labelpad=40, rotation="vertical", fontname="Cambria", fontsize=16)
        plt.savefig(self.save_path / f"{savename}.png", bbox_inches='tight', dpi=300)

        # plt.show()
        plt.close(fig)
        plt.cla()

    def plot_props_part3(self, metadata_df, title, triplets, plot_type, is_multi_dose, hit_col):
        """
        "roc_curves" props extracted from "metadata_df" that depend on the "plot_type" value:
            curve_color
            curve_zorder
            curve_legend_color
            curve_legend_text
        "rocaucs" props extracted from "metadata_df" that depend on the "plot_type" value:
            row_ylabel_color
            row_ylabel_texts
        curve_zorder:"https://stackoverflow.com/questions/35781612/matplotlib-control-which-plot-is-on-top"
        """
        N = len(metadata_df)
        # print(f"metadata_df.shape:{metadata_df.shape}    triplets={len(triplets)}")
        assert len(triplets) >= N
        # TODO: Figure out how to add bad-well-ids to this class!!!
        counter = 0
        props = np.zeros((N, 4), dtype=object)
        cols = ["legend_or_ylabel_text", "legend_or_ylabel_color", "curve_color", "curve_zorder"]
        black, red, blue = "black", "red", "blue"
        z10, z6, z1,  = 10, 6, 1
        ###############################################################
        # Add information about legend color to the title
        if plot_type == 1:
            title += f"\n QC-Pass={black}   QC-Fail={red}"
        elif plot_type == 2 and (not self.args.hit_calling):
            title += f"\n Control={blue}    Not-Control={black}"
        elif plot_type == 2 and self.args.hit_calling:
            title += f"\n Hit={red}    Not-Hit={black}"
        elif plot_type == 3:
            title += f"\n Anchor={blue}   Rest={black}"
        elif plot_type == 4:
            title += f"  treatment=DMSO"
        ###########################################################
        for _, it in metadata_df.iterrows():  # (curves or well-ids)
            c_color = triplets[counter][2]
            if plot_type == 1:
                # different legend color and z-order for outlier wells
                lg_color = red if it[self.qc_label_col] == 1 else black
                props[counter, 0] = it["well-id"]
                props[counter, 1] = lg_color
                props[counter, 2] = c_color
                props[counter, 3] = z1 if it[self.qc_label_col] else z10

            elif plot_type == 2:
                treat = it["treatment"]
                treatments = metadata_df["treatment"].to_list()
                # different legend color and z-order for control treatments and test treatments
                # the black color is reserved for the anchor treatment
                cond0 = treat == self.args.anchor_treatment
                cond1 = np.isin(treat, self.args.control_treatments)
                txt = treat[:self.args.max_chars]
                if (not is_multi_dose) and (treatments.count(treat) != 1):
                    txt += f"-dose={it['dosage']}uM"
                if self.args.hit_calling:
                    lg_color = red if it[hit_col] else black
                else:
                    lg_color = blue if cond1 else black
                props[counter, 0] = txt
                props[counter, 1] = lg_color
                props[counter, 2] = black if cond0 else c_color
                props[counter, 3] = z10 if cond0 else (z6 if cond1 else z1)

            elif plot_type == 3:
                # the black color is reserved for the anchor cell-line
                cond0 = it["cell-line"] == self.args.anchor_cellline
                props[counter, 0] = it["cell-line"]
                props[counter, 1] = blue if cond0 else black
                props[counter, 2] = black if cond0 else c_color
                props[counter, 3] = z10 if cond0 else z1

            elif plot_type == 4:
                # the black color is reserved for the anchor cell-line
                split = it["exp-id"].split("_")
                date, barcode = split[0].split('-')[0], split[-1]
                props[counter, 0] = f"{date}_{barcode}"
                props[counter, 1] = black
                props[counter, 2] = c_color
                props[counter, 3] = z1
            else:
                raise NotImplementedError(
                    f"the plot_type param can only be 1, or 2, or 3, or 4, but was set to {plot_type}!!!")
            counter += 1

        props = pd.DataFrame(props, columns=cols, index=np.arange(N))
        return props, title

    @staticmethod
    @lru_cache(maxsize=1)
    def plot_props_part1(num_curves):
        colors, line_styles, markers, markersize, ncols, bbox_to_anchor, legend_font_size = \
            [], [], [], 0, 0, (0, 0), 0

        if num_curves <= 10:
            colors = list(plt.cm.tab10(np.linspace(0, 1, 10)))
            line_styles = ["-"]
            markers = ["o"]
            markersize = 4
            ncols = 5
            bbox_to_anchor = (-2, -.8)
            legend_font_size = 25

        elif 10 < num_curves <= 20:
            colors = list(plt.cm.tab10(np.linspace(0, 1, 10)))
            line_styles = ["-", "--"]
            markers = ["o"]
            markersize = 3
            ncols = 5
            bbox_to_anchor = (-2, -1)
            legend_font_size = 23

        elif 20 < num_curves <= 40:
            colors = list(plt.cm.tab10(np.linspace(0, 1, 10)))
            line_styles = ["-", "--"]
            markers = ["o", "2"]
            markersize = 3
            ncols = 7
            bbox_to_anchor = (-2, -1.2)
            legend_font_size = 20

        elif 40 < num_curves <= 80:
            colors = list(plt.cm.tab20(np.linspace(0, 1, 20)))
            line_styles = ["-", "--"]
            markers = ["o", "2"]
            markersize = 3
            ncols = 8
            bbox_to_anchor = (-2, -1.5)
            legend_font_size = 18

        elif 80 < num_curves <= 120:
            colors = list(plt.cm.tab20(np.linspace(0, 1, 20)))
            line_styles = ["-", "--"]
            markers = ["o", "2", "3"]
            markersize = 2
            ncols = 8
            bbox_to_anchor = (-2, -1.7)
            legend_font_size = 16

        elif 120 < num_curves <= 160:
            colors = list(plt.cm.tab20(np.linspace(0, 1, 20)))
            line_styles = ["-", "--"]
            markers = ["o", "2", "3", "x"]
            markersize = 2
            ncols = 8
            bbox_to_anchor = (-2, -2)
            legend_font_size = 13

        elif 160 < num_curves <= 200:
            colors = list(plt.cm.tab20(np.linspace(0, 1, 20)))
            line_styles = ["-", "--"]
            markers = ["o", "2", "3", "x", 7]
            markersize = 2
            ncols = 8
            bbox_to_anchor = (-2, -2.5)
            legend_font_size = 12
        else:
            NotImplementedError()

        triplets = [
            (it0, it1, it2)
            for it0 in markers
            for it1 in line_styles
            for it2 in colors
        ]

        return markersize, ncols, bbox_to_anchor, legend_font_size, triplets

    @staticmethod
    def plot_props_part2(num_cases, num_feat_cat):
        title_font_size, subtitle_font_size, annot_font_size, ylabel_font_size, xlabel_font_size, cbar_label_font_size = \
            0, 0, 0, 0, 0, 0
        subplots_width_ratios, subplot_wspace, subplot_top_space, cbar_pad, linewidth = \
            [.1] + [1] * num_feat_cat, 0, 0, 0, 0

        if num_cases <= 20:
            title_font_size = 18
            subtitle_font_size = 17
            annot_font_size = 12
            ylabel_font_size = 14
            xlabel_font_size = 15
            cbar_label_font_size = 12

            subplot_wspace = .09
            subplot_top_space = .85
            cbar_pad = .2
            linewidth = .5
            subplots_width_ratios = [.5] + [1] * num_feat_cat

        elif 20 < num_cases <= 40:
            title_font_size = 16
            subtitle_font_size = 15
            annot_font_size = 10
            ylabel_font_size = 12
            xlabel_font_size = 14
            cbar_label_font_size = 10

            subplot_wspace = .07
            subplot_top_space = .88
            cbar_pad = .18
            linewidth = .4
            subplots_width_ratios = [.5] + [1] * num_feat_cat

        elif 40 < num_cases <= 80:
            title_font_size = 14
            subtitle_font_size = 14
            annot_font_size = 8
            ylabel_font_size = 8
            xlabel_font_size = 12
            cbar_label_font_size = 8

            subplot_wspace = .03
            subplot_top_space = .91
            cbar_pad = .1
            linewidth = .2
            subplots_width_ratios = [.3] + [1] * num_feat_cat

        elif num_cases > 80:
            title_font_size = 10
            subtitle_font_size = 9
            annot_font_size = 4
            ylabel_font_size = 5
            xlabel_font_size = 8
            cbar_label_font_size = 6

            subplot_wspace = .02
            subplot_top_space = .93
            cbar_pad = .08
            linewidth = .1
            subplots_width_ratios = [.11] + [1] * num_feat_cat

        return title_font_size, subtitle_font_size, annot_font_size, \
               ylabel_font_size, xlabel_font_size, cbar_label_font_size, \
               subplots_width_ratios, subplot_wspace, subplot_top_space, cbar_pad, linewidth


class PlateMapAnnot:

    def __init__(self, args):
        self.args = args
        self.nrows = 16
        self.ncols = 24
        self.rows = list(string.ascii_uppercase[:self.nrows])
        self.cols = [str(ii).zfill(2) for ii in np.arange(1, self.ncols+1)]

        self.annot_save_names = ["cell-count", "treatment", "cell-line", "dosage", "density"]
        self.annot_font_size = [10, 7, 8, 9, 9]
        self.num_heatmaps = len(self.annot_font_size)

    def create_annotation(self, sheetname):
        annot = np.zeros((self.nrows, self.ncols), dtype=object)
        for ii in range(self.nrows):
            for jj in range(self.ncols):
                well_id = f"{self.rows[ii]}{self.cols[jj]}"
                if self.args.wellid2treatment.get(well_id) is None:  # it could be a well-id is missing
                    annot[ii, jj] = ""
                else:
                    treatment = Args.shorten_str(self.args.wellid2treatment[well_id], "treatment", self.args.max_chars)
                    cellline = Args.shorten_str(self.args.wellid2cellline[well_id], "cell-line", self.args.max_chars)
                    if sheetname == "treatment":
                        annot[ii, jj] = treatment
                    elif sheetname == "cell-line":
                        annot[ii, jj] = cellline
                    elif sheetname == "dosage":
                        annot[ii, jj] = self.args.wellid2dosage[well_id]
                    elif sheetname == "density":
                        annot[ii, jj] = self.args.wellid2density[well_id]
                    elif sheetname == "other":
                        annot[ii, jj] = self.args.wellid2other[well_id]
                    else:
                        raise FileExistsError(f"Sheet {sheetname} does not exist!!!")
        return annot

    @staticmethod
    def create_discretized_heatmap(
            data,
            annotation,
            annotation_font_size,
            title,
            colors_meta,
            xtick_labels,
            ytick_labels,
            save_path,
            save_name
    ):
        fig, axes = plt.subplots(1, 1, sharex=True, sharey=True)
        fig.set_size_inches(21.3, 13.3)
        fig.suptitle(title, fontsize=25, fontname='Comic Sans MS', )
        N = len(colors_meta)
        colors, values, names = list(zip(*colors_meta))
        cmap = matcolors.ListedColormap(colors)
        vmin, vmax = np.amin(values), np.amax(values)

        ax = sns.heatmap(
            data,
            linecolor='gray',
            linewidth=.5,
            cmap=cmap,
            annot=annotation,
            fmt="",
            annot_kws={
                'fontsize': annotation_font_size,
                'fontstyle': 'italic',
                'color': 'white',
                # 'alpha': 0.6,
                # 'rotation': 'vertical',
                # 'verticalalignment': 'center',
                # 'backgroundcolor': 'w'
            },
            vmin=vmin,
            vmax=vmax
        )
        # Manually specify colorbar labelling after it's been generated
        step = (vmax - vmin) / N
        color_boundary_vals = list(np.arange(vmin, vmax + step, step))
        tick_locs = [(it0 + it1) / 2
                     for it0, it1 in zip(color_boundary_vals[0:-1], color_boundary_vals[1:])]
        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks(tick_locs)
        colorbar.set_ticklabels(names, fontsize=20)

        axes.set_yticks(np.arange(.5, len(list(ytick_labels)) + .5, 1))
        axes.set_yticklabels([it for ii, it in enumerate(list(ytick_labels))])

        axes.set_xticks(np.arange(.5, len(xtick_labels) + .5, 1))
        axes.set_xticklabels([it for ii, it in enumerate(xtick_labels)])

        axes.set_xticklabels(axes.get_xmajorticklabels(), rotation=90, fontsize=15)
        axes.set_yticklabels(axes.get_ymajorticklabels(), rotation=0, fontsize=15)

        plt.savefig(save_path / f"{save_name}.png", bbox_inches='tight', dpi=300)
        # plt.show()
        plt.close(fig)
        plt.cla()

