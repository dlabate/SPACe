import sys
import time
from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.stats import median_abs_deviation
from functools import partial, lru_cache
from sklearn.preprocessing import robust_scale, RobustScaler

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from cellpaint.steps_single_plate.step0_args import Args
import matplotlib.pyplot as plt

"""
The following torch api will help calculate summary aggregates per well, after the cellpaint step 3, feature extraction.
It allows for the calculation of the summaries to be done at once for the entire plate:
    0) load the features dataframe
    1) sort the dataframe in the desired manner, usually based on 
    (density, cell-line, treatment, dosage, well-id) columns
    2) Drop wells with not enough cells.
    3) Define how to get the anchor dataframe per (density, cell-line) pair
    4) Calculate the per well aggregates
    5) group the results of the calculation into different (density, cell-line) pairs
    6) save the results.
    
Note: The computational flow is well-based, meaning each dataset example is the rows of the features dataframe, 
cell-level measurements, that belong to the same well.

This is a rough outline of how it is done for dose response.
However, for quality control and DMSO comparison the steps may differ.  
But the hope is to generalize this API well enough to capture all kinds of scenarios, and different edge cases.


# For ease of understanding, reading, implementation, and etc it is best to write a separate torch class for
each case, step5 quality control, step5-1 treatment distance map from anchor.

Cellpaint Step 5:
    It calculates the distmap, the ROC-Curve, ROC-Derivative-Curve, ROC-AUC, and ROC-Derivative-AUC,
    per (density, cell-line, treatment, dosage) quartet where the treatment
    belongs to control treatments. The anchor treatment will depend on the quartet.
"""


class CellFeatureDataset(TensorDataset):
    shuffle = False
    meta_cols = ["exp-id", "density", "cell-line", "treatment", "dosage", "other"]

    def __init__(self, features, start_index, transform=None):
        super().__init__()

        self.features = features
        self.feat_cols = self.features.columns[start_index:]
        self.transform = transform

    def __getitem__(self, idx):
        features = torch.as_tensor(np.float32(self.features.iloc[idx][self.feat_cols].to_numpy()))
        if self.transform:
            features = self.transform(features)
        return features

    def __len__(self):
        return len(self.features)


class OutlierDetectionModel(nn.Module):
    """
    Calculates mean-distance, wassertein-distance, and median-distance from reference distribution (DMSO)
    https://stackoverflow.com/questions/57540745/
    what-is-the-difference-between-register-parameter-and-register-buffer-in-pytorch
    """

    def __init__(self, n_features):
        super().__init__()
        self.lin1 = nn.Linear(n_features, 50)
        self.bn1 = nn.BatchNorm1d(50)
        self.nl1 = nn.Tanh()

    def forward(self, x):
        x = self.lin1(x)
        x = self.bn1(x)
        x = self.nl1(x)
        return x


def outlier_detection(anchor_features, start_index, device):
    n_epochs = 20
    lr = 1e-3
    batch_size = 8*4096
    is_cuda = torch.cuda.is_available()
    meta_cols = anchor_features.columns[:start_index]
    feat_cols = anchor_features.columns[start_index:]
    n_data, n_features = anchor_features.shape[0], len(feat_cols)
    target = torch.as_tensor(np.float32(np.repeat(np.median(
        anchor_features[feat_cols].to_numpy(), axis=0)[np.newaxis], repeats=len(anchor_features), axis=0)))

    # Define dataset and data loader
    dataset = CellFeatureDataset(anchor_features, start_index)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=is_cuda, num_workers=3)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=is_cuda, num_workers=3)
    print(len(train_loader), len(test_loader))
    # Define model and optimizer
    model = OutlierDetectionModel(n_features)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.HuberLoss()
    # Define training loop
    model.train()
    for epoch in range(n_epochs):
        for ii, batch_data in tqdm(enumerate(train_loader)):
            batch_data = batch_data.to(device)
            m = len(batch_data)
            batch_target = target[0:m].to(device)
            optimizer.zero_grad()
            outs = model(batch_data)
            outs_target = model(batch_target)
            loss = loss_fn(outs, outs_target)
            loss.backward()
            optimizer.step()
            print(epoch, ii, batch_data.size(), loss.item())

    model.eval()
    scores = np.zeros((n_data, ), dtype=np.float32)
    with torch.no_grad():
        for jj, batch_data in tqdm(enumerate(test_loader)):
            batch_data = batch_data.to(device)
            outs = model(batch_data)
            score = loss_fn(outs)
            scores[jj*batch_size, jj*batch_size+len(batch_data)] = score.cpu().numpy()
            del batch_data, outs, score

    # define a percentile and a hard threshold for cell outliers
    anchor_features["outlier-score"] = scores
    percentiles = [2, 5, 15, 25, 75, 90, 95, 98]
    percentile_scores = np.percentile(scores, percentiles)
    for prc, score in zip(percentiles, percentile_scores):
        print(prc, score)
    print('\n')
    for kk, row in anchor_features.iterrows():
        print(row["well-id"], row["outlier-score"])
    # # Define threshold for outlier detection
    # threshold = np.percentile(outlier_scores, q=95)

    # # Identify outliers
    # outliers = data[outlier_scores > threshold]


class WellAggFeaturesDataset(TensorDataset):
    grp_col = "well-id"
    shuffle = False
    # start_feat_col_index=10
    # min_fov_cell_count=20
    # n_fovs_per_well=9
    """
    Getting measurement array and saving as dataframe steps:
        1) Get features groups based on the well-id column
        2) Get the well-id keys and sort them
        3) Extract the desired measurement from each well, using torch api
        4) Reorder the features dataframe the way you desire. (controls first!!!)
        5) Find the index of each well-id key in the ordered features
        6) Evaluate the measurements on those indices
        7) Concatenate and save as a dataframe
    """
    meta_cols = ["exp-id", "density", "cell-line", "treatment", "dosage", "other"]

    def __init__(self, features, feat_cols, transform=None):
        super().__init__()

        self.features = features
        self.feat_cols = feat_cols

        # get metadata value for each well
        # preserving the original order of the group elements with sort=False,
        self.meta_data = self.features.groupby(self.grp_col, sort=False)[self.meta_cols].first().reset_index()
        self.cell_counts = self.features[[self.grp_col, self.meta_cols[0]]].groupby(
            self.grp_col, sort=False).count().reset_index()
        self.cell_counts = self.cell_counts.rename(columns={self.meta_cols[0]: "cell-count"})["cell-count"]
        self.meta_data = pd.concat((self.meta_data, self.cell_counts), axis=1)
        self.transform = transform

    def __getitem__(self, grp_idx):
        grp_val = self.meta_data.iloc[grp_idx][self.grp_col]  # well_id
        features = self.features.loc[self.features[self.grp_col] == grp_val, self.feat_cols].to_numpy()
        features = torch.as_tensor(np.float32(features))
        if self.transform:
            features = self.transform(features)
        return features
        # print(features)

    def __len__(self):
        return len(self.meta_data)

    @staticmethod
    def collate_fn_wellwise(batch):
        """
        batch: list of tensors each of shape (M, Ni) where
            M is the number of feature columns, and
            Ni is the number of cells belonging to a specific well/well-id.
        """
        # print(type(batch), len(batch), batch[0].shape, batch[1].shape)
        return batch


class DistanceMapsFromAnchorModel(nn.Module):
    """
    Calculates mean-distance, wassertein-distance, and median-distance from reference distribution (DMSO)
    https://stackoverflow.com/questions/57540745/
    what-is-the-difference-between-register-parameter-and-register-buffer-in-pytorch
    """

    def __init__(self, anchor_features, feat_cols):
        super().__init__()
        # Distance is calculated from the anchor features
        self.metrics = ("roc-curve", "median-distance", "wasserstein-distance", "mean-distance")
        thresholds = np.linspace(0, 1, 100)
        self.n_thresholds = len(thresholds)
        self.register_buffer("thresholds", torch.as_tensor(np.float32(thresholds)))
        self.register_buffer("anchor_features", torch.as_tensor(np.float32(anchor_features[feat_cols].to_numpy().T)))

    def forward(self, x):
        # make sure you calculate the distance metrics in order of appearance
        print(x)
        median_dist = torch.median(x, dim=1)[0] - torch.median(self.anchor_features, dim=1)[0]
        wasser_dist = self.wassertein_distance_2d(x, self.anchor_features)
        mean_dist = torch.mean(x, dim=1) - torch.mean(self.anchor_features, dim=1)
        median_sign = torch.sign(median_dist)
        # use median sign as for defining the positive direction or negative direction
        wasser_dist = torch.mul(wasser_dist, median_sign)
        mean_dist = torch.mul(mean_dist, median_sign)
        roc_curves = torch.zeros((x.size(0), self.n_thresholds), dtype=torch.float32).to(x.device)
        for ii in range(0, self.n_thresholds - 1):
            roc_curves[:, ii] = torch.sum(torch.where(torch.abs(x) > self.thresholds[ii], x, 0), dim=1)
        # average it for the cells/population
        roc_curves /= x.size(1)
        # return torch.stack((mean_dist, wasser_dist, median_dist), dim=0)
        # mean_reg = torch.mean(x, dim=1)
        # median_reg = torch.median(x, dim=1)[0]
        # std_reg = torch.std(x, dim=1)
        # mad_reg = torch.mean(x, dim=1)
        return roc_curves, median_dist, wasser_dist, mean_dist #, mean_reg, median_reg, std_reg

    @staticmethod
    def wassertein_distance_2d(u_values, v_values):
        """
        pytorch implementation of the wassertein distance between 1d slices of two 2d-distributions:

        Adapted from the scipy.stats.wasserstein_distance implementation:
        https://github.com/scipy/scipy/blob/v1.10.1/scipy/stats/_stats_py.py#L9002-L9076

        u_values: MxN1 tensor
        where M is the number of columns/features and N1 is the number of rows/cells

        v_values: MxN2 anchor tensor where
        M is the number of columns/features and N2 is the number of rows/cells in the
        anchor condition

        Compute, between two one-dimensional distributions :math:`u` and
        :math:`v`, whose respective CDFs are :math:`U` and :math:`V`, the
        statistical distance that is defined as:
        .. math::
            l_p(u, v) = \left( \int_{-\infty}^{+\infty} |U-V|^p \right)^{1/p}
        p is a positive parameter; p = 1 gives the Wasserstein distance, p = 2
        gives the energy distance.
        Parameters
        ----------
        u_values, v_values : array_like
            Values observed in the (empirical) distribution.
        u_weights, v_weights : array_like, optional
            Weight for each value. If unspecified, each value is assigned the same
            weight.
            `u_weights` (resp. `v_weights`) must have the same length as
            `u_values` (resp. `v_values`). If the weight sum differs from 1, it
            must still be positive and finite so that the weights can be normalized
            to sum to 1.
        Returns
        -------
        distance : float
            The computed distance between the distributions.
        Notes
        -----
        The input distributions can be empirical, therefore coming from samples
        whose values are effectively inputs of the function, or they can be seen as
        generalized functions, in which case they are weighted sums of Dirac delta
        functions located at the specified values.
        References
        ----------
        .. [1] Bellemare, Danihelka, Dabney, Mohamed, Lakshminarayanan, Hoyer,
               Munos "The Cramer Distance as a Solution to Biased Wasserstein
               Gradients" (2017). :arXiv:`1705.10743`.
        """
        all_values = torch.cat((u_values, v_values), dim=1)
        all_values, _ = torch.sort(all_values, dim=1)

        u_sorted, _ = torch.sort(u_values, dim=1)
        v_sorted, _ = torch.sort(v_values, dim=1)

        # Compute the differences between pairs of successive values of u and v.
        deltas = torch.diff(all_values, dim=1)
        # Get the respective positions of the values of u and v among the values of
        # both distributions.
        # all_values = all_values[:, :-1].contiguous()
        u_cdf_indices = torch.searchsorted(u_sorted, all_values[:, :-1].contiguous(), right=True)
        v_cdf_indices = torch.searchsorted(v_sorted, all_values[:, :-1].contiguous(), right=True)
        # Calculate the CDFs of u and v using their weights, if specified.
        u_cdf = u_cdf_indices / u_values.size(1)
        v_cdf = v_cdf_indices / v_values.size(1)
        # Compute the value of the integral based on the CDFs.
        # If p = 1 or p = 2, we avoid using np.power, which introduces an overhead
        # of about 15%.
        return torch.sum(torch.mul(torch.abs(u_cdf - v_cdf), deltas), dim=1)


class WellAggFeatureDistanceMetrics:

    analysis_step = 5
    # TODO: Get the feature types from args so that it is automated
    feature_types = ["Misc", "Shape", "Intensity", "Texture"]

    mad_mults = [3, 4, 5, 6, 7]
    roc_auc_keys = ("roc-aucs", "roc-diff-aucs")
    roc_curve_keys = ("roc-curves", "roc-diff-curves")
    distrib_metric_keys = ("wasserstein-distances", "median-distances", "mean-distances",)
    regular_keys = ("mean","median","standard deviation","median absolute deviation")
    distance_keys = distrib_metric_keys + roc_auc_keys + regular_keys

    device_id = 0
    quantile_range = (2, 98)
    qc_dmso_dynamic_range_thresh = .1
    batch_size = 384  # maximum number of groups/wells possible in a 16*24=384-wells plate
    device = torch.device(f"cuda:{device_id}") if torch.cuda.is_available() else torch.device(f"cpu")
    colors = ("red", "black", "orange", "green", "blue")

    def __init__(self, args):
        self.args = args
        self.load_path = self.args.main_path/self.args.experiment/f"Step{self.analysis_step-1}_Features"
        self.save_path = self.args.main_path/self.args.experiment/f"Step{self.analysis_step}_DistanceMaps"
        self.save_path.mkdir(exist_ok=True, parents=True)

        self.min_well_cell_count = self.args.min_fov_cell_count * self.args.n_fovs_per_well // 3
        self.features, self.cell_count_wellwise, self.start_index = self.load_and_preprocess_features()
        self.meta_cols = self.features.columns[:self.start_index]
        self.feat_cols = self.features.columns[self.start_index:]

        # sort the features dataframe properly and move control treatments to the top
        self.features = self.sort_rows_fn(self.features)
        # Feature normalization has to happen before the anchor features are extracted!
        self.normalize_features()

        self.anchor_features = self.get_anchor_features(self.features)
        self.dataset = WellAggFeaturesDataset(self.features, self.feat_cols, transform=None)
        # print(self.dataset)
        self.data_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=self.dataset.collate_fn_wellwise)

    def step5_main_run_loop(self):
        start_time = time.time()
        print("Cellpaint Step 5: Distance Map calculation...")
        counter = 1
        # Calculating the distance of everything from the DMSO wells, and see how many mads away it is,
        # and saving with metadata
        wellwise_summ_metrics = self.get_well_based_feature_summary_distance_metrics()
        writer = pd.ExcelWriter(self.save_path/"wellwise-summary-distance-metrics.xlsx", engine="xlsxwriter")
        for _, key in enumerate(self.distance_keys):
            val = pd.DataFrame(wellwise_summ_metrics[key], columns=self.feat_cols)
            val = pd.concat([self.dataset.meta_data, val], axis=1)
            val.to_excel(writer, sheet_name=f"{counter}_{key}", index=False)
            counter += 1
        writer.close()
        for key in self.roc_curve_keys:
            np.save(self.save_path/f"{counter}_{key}.npy", wellwise_summ_metrics[key])
            counter += 1
        print(f"Finished Cellpaint step 5 in: {(time.time() - start_time)} seconds \n")
        # Hit calling using Euclidean Distance
        # cond = ~np.isin(self.dataset.meta_data["treatment"], self.args.control_treatments)
        # # num_wells = len(wellwise_summ_metrics[self.distance_keys[0]])
        # n_keys = len(self.distance_keys)
        # n_mad_mults = len(self.mad_mults)
        # num_wells = np.sum(cond)
        # res = np.zeros((n_keys, n_mad_mults, num_wells), dtype=np.float32)
        # for ii, key in enumerate(self.distance_keys):
        #     val = wellwise_summ_metrics[key]
        #     anchor_cond = val["treatment"] == self.args.anchor_treatment
        #     anchor_median = val.loc[anchor_cond, self.feat_cols].apply(np.median, axis=0)
        #     anchor_mad = val.loc[anchor_cond, self.feat_cols].apply(median_abs_deviation, axis=0)
        #     diff = np.abs(val.loc[cond] - anchor_median)
        #     for jj, mad_mult in enumerate(self.mad_mults):
        #         tmp = diff.sub(mad_mult*anchor_mad, axis=1)
        #         for c_in in self.channels:
        #             for feat_cat in self.feature_types:
        #                 cols = [col for col in self.feat_cols if c_in in col and feat_cat in col]
        #                 res[ii, jj, :] += np.mean(tmp[cols].to_numpy(), axis=1)
        #
        # hits = np.sum((res > 0).astype(np.uint8), axis=2)
        # print(hits.shape)
        # fig, axes = plt.subplots(1, 1)
        # fig.suptitle("Wellwise Hit Count")
        # for ii, (key, cc) in enumerate(zip(self.distance_keys, self.colors)):
        #     axes.plot(self.mad_mults, hits[ii], label=key, color=cc)
        # axes.set_title("Hits")
        # axes.set_xlabel("MAD Multiplier")
        # axes.set_ylabel("# Hit Wells")
        # plt.legend()
        # plt.show()
        # # hits = res > 0
        # # # do QC on control compounds here!!!
        # # # do hit-calling here!!!

    def get_anchor_features(self, features):
        anchor_cond = (features["density"] == self.args.anchor_density) & \
                      (features["cell-line"] == self.args.anchor_cellline) & \
                      (features["treatment"] == self.args.anchor_treatment) & \
                      (features["dosage"] == self.args.anchor_dosage)
        check_series = (features["cell-line"] == self.args.anchor_cellline)
        print(type(self.args.anchor_cellline),self.args.anchor_cellline)
        # print(anchor_cond)
        # print("Features",features["cell-line"][0],type(features["cell-line"][0]))
        # print("Anchor",int(self.args.anchor_cellline),type(int(self.args.anchor_cellline)))

        # print("Features", features["cell-line"], type(features["cell-line"]))
        # print("Anchor", self.args.anchor_cellline, type(self.args.anchor_cellline))
        # if 'True' in check_series:
        #     print('Yes')
        # else:
        #     print('No')
        # print()
        return features.loc[anchor_cond]
        # print(features.loc[anchor_cond])

    def get_controls(self, features):
        return features.loc[np.isin(features["treatment"].to_numpy(), self.args.control_treatments)]

    def normalize_features(self, ):
        """Normalize features per plate by centering the feature values on the median of
        anchor cellline's anchor treatment (DMSO)."""
        # get normalization factors per plate, by centering on anchor (cell-line, treatment) median
        print(self.features)
        controls = self.get_controls(self.features)
        anchor = self.get_anchor_features(self.features)
        # print(controls)
        # print(anchor)
        # print(anchor[self.feat_cols])
        medians = np.nanmedian(anchor[self.feat_cols].to_numpy(), axis=0)
        quantiles = np.percentile(controls[self.feat_cols].to_numpy(), self.quantile_range, axis=0)
        scales = self.handle_zeros_in_scale(quantiles[1] - quantiles[0])
        self.features[self.feat_cols] -= medians
        print(medians)
        self.features[self.feat_cols] /= scales
        # print(scales)
        # print(self.features)
        # for jj, col in enumerate(self.feat_cols):
        #     print(jj, col, np.percentile(self.features[col], [2, 50, 98]), medians[jj], scales[jj])
        # # clip outlier feature values to -1 and 1
        # features[self.feat_cols] = features[self.feat_cols].clip(lower=-1, upper=1)

    def sort_rows_fn(self, df):
        df.sort_values(
            by=["exp-id"], ascending=[True], ).sort_values(
            by=["density"], ascending=[True], key=lambda x: x == self.args.anchor_density).sort_values(
            by=["cell-line"], ascending=[True], key=lambda x: x == self.args.anchor_cellline).sort_values(
            by=["treatment"], ascending=[True], key=lambda x: np.isin(x, self.args.control_treatments), ).sort_values(
            by=["treatment"], ascending=[True], key=lambda x: x == self.args.anchor_treatment, ).sort_values(
            by=["dosage", "well-id"], inplace=True)
        return df
        # print(df)
    def get_well_based_feature_summary_distance_metrics(self, ):
        # print(self.anchor_features)
        # print(self.feat_cols)
        dist_model = DistanceMapsFromAnchorModel(self.anchor_features, self.feat_cols)

        dist_model.to(self.device)
        dist_model.eval()

        batch_size = self.data_loader.batch_size
        N, M1, M2 = self.dataset.__len__(), len(self.dataset.feat_cols), dist_model.n_thresholds
        # print(N, M1, M2)
        summs_wellwise = {key: np.zeros((N, M1), dtype=np.float32) for key in self.distance_keys}
        summs_wellwise.update({key: np.zeros((N, M1, M2), dtype=np.float32) for key in self.roc_curve_keys})
        with torch.no_grad():
            # for jj, (feats, sids, eids) in tqdm(enumerate(data_loader), total=len(data_loader)):
            for ii, features_batch in enumerate(self.data_loader):
                count = len(features_batch)
                start = ii * batch_size
                end = start + count
                index = start
                for jj in tqdm(range(count)):
                    features = torch.transpose(features_batch[jj], 0, 1).to(self.device)
                    roc_curve, median_dist, wass_dist, mean_dist = dist_model(features)
                    roc_curve, median_dist, wass_dist, mean_dist = \
                        roc_curve.cpu().numpy(), \
                            median_dist.cpu().numpy(), \
                            wass_dist.cpu().numpy(), \
                            mean_dist.cpu().numpy(), \

                    # print(ii, jj, start, end, features.size(), wass_dist.shape)
                    # print(roc_curve.shape, wass_dist.shape, mean_dist.shape, median_dist.shape)
                    summs_wellwise["roc-curves"][index] = roc_curve
                    summs_wellwise["median-distances"][index] = median_dist
                    summs_wellwise["wasserstein-distances"][index] = wass_dist
                    summs_wellwise["mean-distances"][index] = mean_dist
                    # summs_wellwise["mean"][index] = mean_reg
                    # summs_wellwise["median"][index] = median_reg
                    # summs_wellwise["standard deviation"][index] = std_reg
                    index += 1
        torch.cuda.empty_cache()
        # it is easier to calculate the rest receiver operating characteristic curves type metrics
        # with numpy rather than torch
        summs_wellwise["roc-aucs"] = simpson(summs_wellwise["roc-curves"], dx=1, axis=2)
        summs_wellwise["roc-diff-curves"] = np.abs(np.gradient(summs_wellwise["roc-curves"], axis=2))
        summs_wellwise["roc-diff-aucs"] = simpson(summs_wellwise["roc-diff-curves"], dx=1, axis=2)
        return summs_wellwise

    @lru_cache(maxsize=2)
    def load_and_preprocess_features(self, ):
        start_time = time.time()
        metadata_features = pd.read_csv(self.load_path / "metadata_features.csv")
        misc_features = pd.read_csv(self.load_path / "misc_features.csv")
        shape_features = pd.read_csv(self.load_path / "shape_features.csv")
        intensity_features = pd.read_csv(self.load_path / "intensity_features.csv")
        texture_features = pd.read_csv(self.load_path / "texture_features.csv")
        features = pd.concat(
            [metadata_features, misc_features, shape_features, intensity_features, texture_features],
            axis=1)
        # print(features)
        # removing Nans
        features.dropna(axis=0, inplace=True)
        #########################################################
        # If I changed the name of the experiment folder after feature extraction
        if "exp-id" in list(features.columns):
            features["exp-id"] = self.load_path.parents[0].stem
        ##############################################################################################################
        # remove wells with not enough cells
        #########################################################################
        # all computations from here onward are well based.
        # So if we remove wells with low cell count here, then we don't have to worry about it anymore!!!
        cell_count_wellwise = features["well-id"].value_counts().reset_index()
        cell_count_wellwise.rename(columns={"count": "cell-count"}, inplace=True)
        # enough_cell_count_well_ids = (cell_count_wellwise["well-id"].loc[
        #     cell_count_wellwise["cell-count"] >= self.args.min_well_cell_count]).to_list()
        enough_cell_count_well_ids = (cell_count_wellwise["well-id"].loc[
            cell_count_wellwise["cell-count"] >= self.min_well_cell_count]).to_list()
        features = features.iloc[np.isin(features["well-id"], enough_cell_count_well_ids)]

        ############################################################################################################
        start_index_feat_cols = metadata_features.shape[1]
        size_mb = np.round(sys.getsizeof(features) / 1024 / 1024, 2)
        print(f"loading took  {time.time()-start_time:.4f}  seconds ....")
        print(f"features size in MB={size_mb}  shape={features.shape}")
        # print(features)

        return features, cell_count_wellwise, start_index_feat_cols

    def normalize_arr(self, arr):
        """Normalize the group by centering the feature values on the median of
        anchor cellline's anchor treatment (DMSO)."""
        medians = np.nanmedian(arr, axis=0)
        # print(medians)
        quantiles = np.percentile(arr, (2, 98), axis=0)
        scales = self.handle_zeros_in_scale(quantiles[1] - quantiles[0])
        arr -= medians
        arr /= scales
        return arr

    @staticmethod
    def handle_zeros_in_scale(scale, ):

        """Taken from https://github.com/scikit-learn/scikit-learn/blob/
        364c77e047ca08a95862becf40a04fe9d4cd2c98/sklearn/preprocessing/_data.py#L90

        Set scales of near constant features to 1.
        The goal is to avoid division by very small or zero values.
        Near constant features are detected automatically by identifying
        scales close to machine precision unless they are precomputed by
        the caller and passed with the `constant_mask` kwarg.
        Typically for standard scaling, the scales are the standard
        deviation while near constant features are better detected on the
        computed variances which are closer to machine precision by
        construction.
        """
        # if we are fitting on 1D arrays, scale might be a scalar
        if np.isscalar(scale):
            if scale == 0.0:
                scale = 1.0
            return scale
        elif isinstance(scale, np.ndarray):
            # Detect near constant values to avoid dividing by a very small
            # value that could lead to surprising results and numerical
            # stability issues.
            constant_mask = scale < 10 * np.finfo(scale.dtype).eps
            scale[constant_mask] = 1.0
            return scale

