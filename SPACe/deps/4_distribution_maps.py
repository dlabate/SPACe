import os
from pathlib import WindowsPath
import time

import scipy.stats
from tqdm import tqdm

import numpy as np
import pandas as pd

from scipy import stats
from sklearn.preprocessing import minmax_scale, robust_scale

import seaborn as sns
import matplotlib.pyplot as plt

# from plotly import express as px
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go

import multiprocessing as mp
from functools import partial

"""https://stackoverflow.com/questions/73380537/
how-to-add-multiple-labels-for-multiple-groups-of-rows-in-sns-heatmap-on-right-s"""


class FoundItem(Exception):
    pass


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.default'] = 'regular'




def plot_heatmap(full_csv_path, thresh=.1, ):
    dist_df = pd.read_csv(full_csv_path, index_col=0).reset_index()
    prefix, dist_type, channel = full_csv_path.stem.split('_')
    # print(prefix, dist_type, channel)
    # print(dist_df.head())
    ########################################################
    # from bounding-box cols
    # "well-id", "cell-count", "treatment", "density", "dosage"
    meta_df = dist_df[dist_df.columns[:5]]
    meta_df['treatment'] = meta_df["treatment"].apply(lambda x: x[:np.min([len(x) - 2, 10])] + x[-2:])
    cols = list(dist_df.columns)[5:]
    #############################################################
    # apply the upper and lower thresholds for better visualization
    df_plot = dist_df[(dist_df[cols] <= -thresh) | (dist_df[cols] >= thresh)][cols]
    ##################################################################
    # # remove bad wells
    # valids = filter_index(list(df_plot.index))
    # df_plot = df_plot[df_plot.index.isin(valids)]
    ##################################################################
    # predefine/preset x axis and yaxis labels
    xlabels = df_plot.columns.tolist()
    # remove channel name from the feature names
    xlabels = ["".join(it.split('-')[1:]) for it in xlabels]
    ylabels_left = list(meta_df['well-id'] + "_#" + meta_df['cell-count'].astype(str))
    # get cell count for unique treatment, density, dosage triplets
    tmp_df = meta_df.groupby(['treatment', 'density', 'dosage']).sum().reset_index()
    y_label_right_with_count = tmp_df[tmp_df.columns].apply(lambda x: '_'.join(x.astype(str)), axis=1).to_list()
    y_label_right_without_count = tmp_df[['treatment', 'density', 'dosage']] \
        .apply(lambda x: '_'.join(x.astype(str)), axis=1).to_list()
    ylabel_right_all = meta_df[["treatment", "density", "dosage"]].apply(
        lambda x: '_'.join(x.astype(str)), axis=1).to_list()

    stops = [ylabel_right_all.index(it) for it in y_label_right_without_count] + [len(ylabel_right_all)]
    ###################################################################################
    # create the matplotlib/sns figure object
    fig, (ax1, axcb) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 0.08]}, )
    fig.set_size_inches(21.3, 13.3)
    # fig.set_size_inches(54.1, 33.8 )
    fig.suptitle(
        f"{args.expid}: {dist_type} + {channel}",
        # fontname='Comic Sans MS',
        fontsize=18)
    #######################################################################
    # create the heatmap
    # set lower and upper thresholds for heatmap values for better display
    # ymin = np.percentile(df_plot.values, 5)
    # ymax = np.percentile(df_plot.values, 95)
    g1 = sns.heatmap(
        df_plot,
        # linewidths=.05,
        # linecolor='black',
        cmap="jet",
        ax=ax1,
        cbar_ax=axcb,
        vmin=-1, vmax=1,
        cbar_kws={"shrink": 0.4})
    ##########################################################################
    # set x ticks positions and values
    g1.set_xticks(np.arange(.5, len(xlabels) + .5, 1))
    g1.set_xticklabels([it for ii, it in enumerate(xlabels) if ii % 1 == 0])
    g1.set_xticklabels(g1.get_xmajorticklabels(), fontsize=5)
    # set y ticks positions and values
    g1.set_yticks(np.arange(.5, len(ylabels_left) + .5, 1))
    g1.set_yticklabels([it.lower() for ii, it in enumerate(ylabels_left) if ii % 1 == 0])
    g1.set_yticklabels(g1.get_ymajorticklabels(), fontsize=3)
    ##########################################################################
    # set treatment label on right side of the plot for better readability
    #######################################################################
    # # TODO: Add cell count
    # # labels for the bands, list of places where the bands start and stop
    # create a loop using the begin and end of each band, the colors and the labels
    for beg, end, label in zip(stops[:-1], stops[1:], y_label_right_with_count):
        # add some text to the center right of this band
        g1.text(1.01, (beg + end) / 2,
                # '\n'.join(label.split())
                label,
                ha='left', va='center',
                transform=g1.get_yaxis_transform(), size=9)
    # sns.set_context(font_scale=0.7)  # scale factor for all fonts
    ##############################################################################
    # customize horizontal and vertical gridlines
    ww = list(np.arange(0, len(xlabels), 1))
    hh = list(np.arange(0, len(ylabels_left), 1))
    for it in stops[1:-1]:
        g1.hlines(y=it, xmin=0, xmax=len(ww), linestyle='dashed', linewidth=.4, color="black")
    x = [0]
    y = ['Shape']
    # get the first moment key column index
    try:
        for cc in args.moment_keys:
            for it in cols:
                if cc in it:
                    x.append(cols.index(it))
                    y.append(it)
                    raise FoundItem
    except:
        pass
    # get the first intensity key column index
    try:
        for cc in args.intensity_keys:
            for it in cols:
                if cc in it:
                    x.append(cols.index(it))
                    y.append(it)
                    raise FoundItem
    except:
        pass

    # get haralick features column index
    haralick_names = []
    for cc in args.haralick_features:
        for it in cols:
            if cc in it:
                x.append(cols.index(it))
                y.append(it)
                haralick_names.append(cc)
                break
    # print(haralick_names)
    for it in x:
        g1.vlines(x=it, ymin=0, ymax=len(hh), linestyle='dashed', linewidth=.4, color="black")
    # add text for sub-categories of features as well
    x.append(len(cols))
    # print(x)
    # print(y)
    for beg, end, label in zip(x[:-1], x[1:], ["Shape", "Moments", "Intensity"] + haralick_names):
        # add some text to the center right of this band
        g1.text((beg + end) / 2, 1.05,
                # '\n'.join(label.split())
                label,
                va='top', ha='center',
                transform=g1.get_xaxis_transform(), size=14, )
    ############################################################################################################
    # rotate the tick labels correctly
    tl = g1.get_xticklabels()
    g1.set_xticklabels(tl, rotation=90)
    tly = g1.get_yticklabels()
    g1.set_yticklabels(tly, rotation=0)
    ###############################################################################
    plt.subplots_adjust(wspace=.32)
    # plt.tight_layout()
    # plt.savefig(csv_path/ f"{prefix}-{cin}-{channel_name}-{key}-{thresh}.png",
    plt.savefig(str(full_csv_path).replace(".csv", f"_{thresh}.png"),
                bbox_inches='tight', dpi=300)
    # plt.show()
    plt.close(fig)


def main():
    thresh = .1
    axis_font = {'fontname': 'sans-serif', 'size': 20}
    save_path = args.main_path / args.experiment / "results"
    list_csvs = list(save_path.glob("*.csv"))
    print(len(list_csvs))

    # for full_csv_path in list_csvs:
    #     plot_heatmap(full_csv_path, thresh=.1)

    with mp.Pool(processes=mp.cpu_count()-10) as pool:
        for _ in pool.imap(partial(plot_heatmap, thresh=.1), list_csvs):
            pass


if __name__ == "__main__":
    st_time = time.time()
    main()
    print(f"Time taken {time.time() - st_time} seconds.")
