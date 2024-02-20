import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import scipy.stats as scstats
# from scipy.integrate import simpson
# from sklearn.ensemble import IsolationForest
# from sklearn.preprocessing import robust_scale
from sklearn.cluster import AffinityPropagation
from cellpaint.utils.shared_memory import MyBaseManager, TestProxy
from cellpaint.utils.torch_api import DistCalcDataset, DistCalcModel, dist_calc_fn
from cellpaint.utils.post_feature_extraction import FeaturePreprocessing, ROCAUC, PlateMapAnnot
# from torch.profiler import profile, record_function, ProfilerActivity


import time
from tqdm import tqdm
import torch.multiprocessing as tmp

import string
import seaborn as sns
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None


class QualityControl(FeaturePreprocessing, ROCAUC, PlateMapAnnot):
    analysis_step = 4
    plot_type = 1
    is_multi_dose = False
    normalize_quartile_range = (2, 98)
    qc_label_col = "QC-label"

    # This FLAG has to always to be true to perform the QC
    calc = True

    # fail_count_lb = 4
    # pass_dynamic_range_thresh = .1
    # mad_multiplier = 7
    # min_feat_rows = 450  # minimum number of cells which is the number of rows in the features dataframe
    # batch_size = 8

    group_cols = ["exp-id", "cell-line", "density", "dosage", "well-id", "treatment"]
    calc_cols = ["exp-id", "well-id"]
    save_maps = True

    def __init__(self, args):
        """https://stackoverflow.com/questions/9575409/
        calling-parent-class-init-with-multiple-inheritance-whats-the-right-way"""
        FeaturePreprocessing.__init__(self)
        ROCAUC.__init__(self, args)
        PlateMapAnnot.__init__(self, args)

        self.args = args
        self.analyis_type = 1

        self.save_path = self.args.step4_save_path
        self.sort_types = [True] * len(self.group_cols)

        self.all_features, self.cell_count, self.start_index = \
            self.load_and_preprocess_features(
                self.args.step3_save_path, self.args.step4_save_path, self.args.min_well_cell_count, self.analysis_step)
        # print(self.all_features, self.cell_count.shape, self.cell_count.columns,
        # '\n', self.cell_count.head(3))
        self.feat_cols = list(self.all_features.columns[self.start_index:])
        self.M = len(self.feat_cols)
        ############################################
        # Only for seema
        # self.device = "cpu"
        # self.batch_size = 384
        # torch.set_num_threads(tmp.cpu_count())
        ###################################################

        # self.cell_count.to_csv(self.save_path)

        self.R = len(self.args.celllines)

    def calculate_single(self, quartet):
        treatment, cellline, density, index = quartet
        treat_name = treatment.upper().replace("_", "-").replace(" ", "-")[0:self.args.max_chars]
        cellline_name = str(cellline).upper().replace("_", "-").replace(" ", "-")
        # treat_name = self.args.shorten_str(treatment, key="treatment").upper()
        # cellline_name = self.args.shorten_str(cellline, key="cell-line").upper()
        cond0 = ((self.all_features["cell-line"] == cellline) &
                 (self.all_features["density"] == density) &
                 (self.all_features["treatment"] == treatment))
        dosages = np.unique(self.all_features[cond0]["dosage"].to_numpy())
        all_outliers = []
        for jj, dose in enumerate(dosages):
            features = self.all_features[cond0 & (self.all_features["dosage"] == dose)]
            features = self.normalize_features(features, self.start_index, self.normalize_quartile_range)
            # print(jj, treat_name, cellline_name, density, dose, len(features))
            if len(features) < self.args.qc_min_feat_rows:  # not enough rows/cells
                print(f"case={jj} treatment={treat_name} cell-line={cellline_name} density={density} dosage={dose} "
                      f"has ncells={len(features)} < well-cell-count-threshold={self.args.qc_min_feat_rows}!"
                      f"Skipping!!!")
                continue
            # print(f"counter={index}-{jj} {cellline_name} {density} {treat_name} {dose} features {features.shape}")
            features.reset_index(inplace=True, drop=True)
            features.sort_values(by=self.group_cols, ascending=[True] * len(self.group_cols), inplace=True)
            dataset = DistCalcDataset(
                features, self.group_cols, self.start_index, self.args.min_well_cell_count, self.analysis_step)
            if len(dataset.unique_indices) < 1:  # no groups with enough cells/rows
                print(f"case={jj} treatment={treat_name} cell-line={cellline_name} density={density} dosage={dose} "
                      f"has all the wells already removed because of low per well cell-count!"
                      f"Skipping!!!")
                continue
            data_loader = DataLoader(
                dataset,
                batch_size=self.args.qc_batch_size,
                shuffle=False,
                pin_memory=False,
                num_workers=0,
                collate_fn=DistCalcDataset.custom_collate_fn)
            model = DistCalcModel(features, self.feat_cols, self.analysis_step)
            model.eval()

            # Profile the fucking memory usage!!!!
            # for tt in range(20):
            #     inputs = torch.randn(244, 1000*tt)
            #     with profile(activities=[ProfilerActivity.CPU],
            #                  profile_memory=True, record_shapes=True) as prof:
            #         model(inputs)
            #     print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=20))
            #     print('\n')

            # if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            #     model = torch.nn.DataParallel(model)
            model = model.to(self.device)
            dist_map = dist_calc_fn(model, data_loader, device=self.device, analysis_step=self.analysis_step)

            """groups_meta columns are (cell_count, exp-id, cell-line, treatment, well-id)"""
            # well_ids, cell_counts = groups_meta[:, -1], groups_meta[:, 0]
            # dist_map = self.get_distances_from_ref_distribution(features, groups_index, unix)
            dist_map = pd.DataFrame(dist_map,
                                    columns=self.feat_cols,
                                    index=dataset.groups_metadata["well-id"].to_list())
            # roccurves, rocaucs = self.compute_roccurve_and_rocauc(dist_map, dataset.groups_metadata, self.plot_type)
            roc_curves, roc_aucs, roc_diff_curves, roc_aucs = self.compute_roccurve_and_rocauc_np(dist_map)

            assert roc_aucs.shape[2] == dataset.groups_metadata.shape[0]
            dataset.groups_metadata[self.qc_label_col] = self.find_outliers(roc_aucs)
            all_outliers.append(dataset.groups_metadata)

            if self.save_maps:
                common = f"{treat_name}  {cellline_name}  density={density}  dosage={dose}uM  l1-Norm"
                title1 = f"{common}\nROCCurve QC"
                title2 = f"{common}\nROCAUC QC"
                savename1 = f"{treat_name}_ROC-Curves_l1_{index}-{jj}_{cellline_name}_{density}_{dose}"
                savename2 = f"{treat_name}_ROC-Derivative-AUC_l1_{index}-{jj}_{cellline_name}_{density}_{dose}"
                self.plot_roccurves(
                    roc_curves, dataset.groups_metadata,
                    self.plot_type, self.is_multi_dose, title1, savename1, "")
                self.plot_rocauc_heatmap(
                    roc_aucs, dataset.groups_metadata, "cell-count",
                    self.plot_type, self.is_multi_dose, title2, savename2, "")

        if len(all_outliers) > 0:
            all_outliers = pd.concat(all_outliers, axis=0)
            all_outliers.to_csv(
                self.args.step4_save_path /
                f"{treat_name}_outliers_{index}_{cellline_name}_{density}.csv",
                index=False)

    def plot_platemap(self, ):
        """
        https://stackoverflow.com/questions/38836154/discrete-legend-in-seaborn-heatmap-plot
        https://stackoverflow.com/questions/57892473/how-to-map-discrerte-values-to-a-heatmap-in-seaborn
        https://stackoverflow.com/questions/33158075/custom-annotation-seaborn-heatmap
        """
        # 16 X 24 = 384 wells plate-map
        nrow, ncol = 16, 24
        rows = list(string.ascii_uppercase[:nrow])
        cols = [str(ii).zfill(2) for ii in np.arange(1, ncol+1)]
        # TODO: Figure out a way to make it work in general, so that the missing wells are white
        # rows = [f"{it[0]}" for it in self.args.wellids]
        # cols = [f"{it[1:]}" for it in self.args.wellids]

        # step 0) Create the pass/fail/low cell #/test/ignore platemap
        # (It is a colormap/heatmap with green/red/black/blue/white colors)
        pdf = pd.DataFrame(np.ones((nrow, ncol), dtype=np.float32), columns=cols, index=rows)
        ##############################################
        # step 1) combine outlier csv files
        df = []
        csv_files = list(self.args.step4_save_path.rglob("*_outliers_*.csv"))
        if len(csv_files) == 0:
            raise ValueError("You have not created the outlier CSV files!!!!")

        for ii, it in enumerate(csv_files):
            df.append(pd.read_csv(it, index_col=0))
        df = pd.concat(df)
        df.reset_index(inplace=True)
        df["QC-label"] = df["QC-label"].astype(np.uint8)
        df["QC-label"][df["QC-label"] == 0] = 4   # 0 are good/pass wells
        df["QC-label"][df["QC-label"] == 1] = 3   # 1 are bad/fail wells
        bad_wellids = df["well-id"][df["QC-label"] == 3].to_numpy()
        np.save(self.args.step4_save_path / "bad_wellids.npy", bad_wellids)

        for ii, row in df.iterrows():
            label, well_id = row["QC-label"], row["well-id"]
            row, col = well_id[0], well_id[1:]
            pdf.loc[row, col] = label
        #######################################
        # step 2) Get per wells cell-count as well as flag cells with not enough cell count as "low cell #"
        if (self.args.step3_save_path / "cell_count.csv").is_file():
            cell_counts = pd.read_csv(self.args.step3_save_path / "cell_count.csv")
            cell_counts = cell_counts[["well-id", "cell-count"]].groupby(
                ["well-id"], group_keys=False).sum().reset_index()
        else:
            cell_counts = self.get_cell_count_per_well(self.all_features)

        not_enough_cells_well_ids = \
            cell_counts.loc[cell_counts["cell-count"] < self.args.min_well_cell_count]["well-id"].to_list()
        for ii, well_id in enumerate(not_enough_cells_well_ids):
            row, col = well_id[0], well_id[1:]
            pdf.loc[row, col] = 2
        #########################
        # step 3)  the cell-count platemap
        cell_counts_arr = np.zeros((nrow, ncol), dtype=int)
        for ii in range(nrow):
            for jj in range(ncol):
                well_id = f"{rows[ii]}{cols[jj]}"
                cc = cell_counts["cell-count"].loc[cell_counts["well-id"] == well_id].to_numpy()
                if len(cc) != 0 and len(cc) == 1:
                    cell_counts_arr[ii, jj] = cc[0]
                elif len(cc) == 0:  # wells that do not have corresponding images, or are not used at all!
                    cell_counts_arr[ii, jj] = 0
                    row, col = well_id[0], well_id[1:]
                    pdf.loc[row, col] = 0
                else:
                    raise ValueError("Can't have a well with two different cell-counts!!!")

        ##########################
        # step 4) plot two heatmaps using the pdf dataframe
        # one with treatment names, the other with cell_count_arr as annotations
        colors_meta = [
                ("gray", 0, "Ignore"),
                ("blue", 1, "Test"),
                ("black", 2, "Low\nCell\n#"),
                ("red", 3, "Fail"),
                ("darkgreen", 4, "Pass")]
        for ii in range(self.num_heatmaps):
            self.create_discretized_heatmap(
                data=pdf,
                annotation=self.create_annotation(self.annot_save_names[ii]) if ii > 0 else cell_counts_arr,
                annotation_font_size=self.annot_font_size[ii],
                title=f"{self.args.experiment}\n QC on Control Treatments",
                colors_meta=colors_meta,
                xtick_labels=pdf.columns,
                ytick_labels=list(pdf.index),
                save_path=self.args.step4_save_path,
                save_name=f"Platemap_{ii}_{self.annot_save_names[ii]}")

    def find_outliers(self, rocaucs):
        """
        rocaucs.shape = (self.num_feat_cat, self.num_cin, num_curves)
        Get one (median, mad) value per (feature_category, channel_in) compartment
        for all the curves (wells) to use for outlier flagging per compartment.

        returns:
            outliers numpy array of shape (num_curves, self.num_feat_cat*self.num_cin)
            The entries are 0 or 1, where 0 means a pass/good well, 1 means a fail/bad well.
        """
        num_feat_cat, num_cin, num_curves = rocaucs.shape[0], rocaucs.shape[1], rocaucs.shape[2]

        # median and mad per (feature category, channel) aggregated over all curves
        median = np.repeat(np.median(rocaucs, axis=2)[:, :, np.newaxis], repeats=num_curves, axis=2)
        mad = np.repeat(scstats.median_abs_deviation(rocaucs, axis=2)[:, :, np.newaxis], repeats=num_curves, axis=2)
        # minimum and maximum per feature category aggregated over all curves and all channels
        min_ = np.nanmin(rocaucs, (1, 2))[:, np.newaxis, np.newaxis]
        max_ = np.nanmax(rocaucs, (1, 2))[:, np.newaxis, np.newaxis]
        min_ = np.repeat(np.repeat(min_, repeats=num_cin, axis=1), repeats=num_curves, axis=2)
        max_ = np.repeat(np.repeat(max_, repeats=num_cin, axis=1), repeats=num_curves, axis=2)
        assert rocaucs.shape == median.shape == mad.shape == min_.shape == max_.shape

        # Outliers (0 is a pass/good well, 1 is a fail/bad well) are detected based on two conditions:
        # 1) dynamic_range: Hard thresholding, not data driven
        # 2) fail: Adaptive thresholding, data driven
        dynamic_range = (max_ - min_) > self.args.qc_pass_dynamic_range_thresh
        fail = (median - self.args.qc_mad_multiplier * mad > rocaucs) | \
               (median + self.args.qc_mad_multiplier * mad < rocaucs)
        outliers = (np.sum(dynamic_range & fail, axis=(0, 1)) >=
                    self.args.qc_fail_compartment_count).astype(np.uint8)
        return outliers

    def plot_feature_heatmap(self, distances, groups_meta, title, savename):
        min_, max_ = np.nanmin(distances), np.nanmax(distances)
        """groups_meta columns are (cell_count, exp-id, cell-line, treatment, well-id)"""
        meta_labels = [f"{it[4]}_{it[0]}" for it in groups_meta]

        fig, axes = plt.subplots(1, 1, sharex=True, sharey=True)
        fig.set_size_inches(21.3, 13.3)
        fig.suptitle(title, fontname='Comic Sans MS', fontsize=20)
        sns.heatmap(distances, cmap=self.args.cmap, vmin=min_, vmax=max_, )

        axes.set_xticks(np.arange(.5, len(self.feat_cols) + .5, 1))
        axes.set_xticklabels([it for ii, it in enumerate(self.feat_cols)])
        axes.set_yticks(np.arange(.5, len(meta_labels) + .5, 1))
        axes.set_yticklabels([it for ii, it in enumerate(meta_labels)])
        axes.set_xticklabels(axes.get_xmajorticklabels(), rotation=90, fontsize=6)
        axes.set_yticklabels(axes.get_ymajorticklabels(), rotation=0, fontsize=6)

        plt.savefig(self.args.step4_save_path / f"{savename}.png", bbox_inches='tight', dpi=300)
        plt.close(fig)

    def get_clusters(self, distances, groups_meta):
        clustering = AffinityPropagation(random_state=100, ).fit(distances)
        num_labels = clustering.labels_
        median_ = np.median(distances, axis=1)
        unix = np.unique(num_labels)
        for ii, it in enumerate(unix):
            cond = (num_labels == it)
            print(it, groups_meta[cond, 2:4], median_[cond])


def step4_run_for_loop(args):
    s_time = time.time()
    print("Cellpaint Step 4 ... Quality Control begins ... ")

    inst = QualityControl(args)
    if inst.calc:
        celllines = np.unique(inst.all_features["cell-line"])
        densities = np.unique(inst.all_features["density"])
        cases = [(it0, it1, it2)
                 for it0 in inst.args.control_treatments
                 for it1 in celllines
                 for it2 in densities]
        cases = [it + (ii,) for ii, it in enumerate(cases)]
        M = len(cases)
        # for ii, quartet in tqdm(enumerate(cases), total=M):
        for ii, quartet in tqdm(enumerate(cases), total=M):
            inst.calculate_single(quartet)
    # plot the quality control result as a heatmap the same shape as the 384 well platemap
    inst.plot_platemap()
    torch.cuda.empty_cache()
    print(f"Cellpaint Step 4 took {time.time() - s_time} seconds to finish ...")
    print("***********************")


def step_4_run_multi_processing_for_loop(args):
    s_time = time.time()
    #############################################
    # There is a hidden race condition here:
    # DO NOT USE THIS FUNCTION!!!!
    ##############################################
    MyManager = MyBaseManager()
    # register the custom class on the custom manager
    MyManager.register("QualityControl", QualityControl, TestProxy)
    # create a new manager instance
    with MyManager as manager:
        inst = getattr(manager, "QualityControl")(args)
        if inst.calc:

            celllines = np.unique(inst.all_features["cell-line"])
            densities = np.unique(inst.all_features["density"])
            cases = [(it0, it1, it2)
                     for it0 in inst.args.control_treatments
                     for it1 in celllines
                     for it2 in densities]
            cases = [it + (ii,) for ii, it in enumerate(cases)]
            M = len(cases)
            num_processes = M

            # # for ii, quartet in tqdm(enumerate(cases), total=M):
            # for ii, quartet in tqdm(enumerate(cases), total=M):
            #     inst.calculate_single(quartet)

            processes = []
            for ii, rank in tqdm(enumerate(range(num_processes)), total=M):
                p = tmp.Process(target=inst.calculate_single, args=(cases[ii],))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        # plot the quality control result as a heatmap the same shape as the 384 well platemap
        inst.plot_platemap()
        print(f"Cellpaint Step 4 took {time.time() - s_time} seconds to finish ...")
        print("***********************")
