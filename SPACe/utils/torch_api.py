import sys
import itertools
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset


class DistCalcDataset(TensorDataset):
    num_groups_ub = 384  # for 16*24 = 384 well plates

    def __init__(self, features, group_cols, start_index, min_well_cell_count, analysis_step):
        super().__init__()
        self.analysis_step = analysis_step
        self.min_well_cell_count = min_well_cell_count
        self.calc_cols = ["exp-id", "well-id"]
        self.group_cols = group_cols
        self.cols = list(features.columns)
        for col in self.calc_cols+group_cols:
            assert col in self.cols
        # features must be sorted based on (exp-id, well-id), so that later,we can slice it as a tensor correctly.
        if self.analysis_step == 5:
            features.reset_index(inplace=True, drop=True)
            features.sort_values(by=self.calc_cols, ascending=[True, ] * len(self.calc_cols), inplace=True)

        # groups_index is necessary for dist_map gpu-calculation, as we can't pass string tensors to gpus,
        # so we need to map the metadata to a set of numeric indices for reference.
        self.groups_metadata, self.groups_indices = self.get_groups(features)
        self.unique_indices = np.unique(self.groups_indices[self.groups_indices != -1])

        self.feat_cols = list(features.columns[start_index:])
        self.groups_indices = torch.as_tensor(self.groups_indices).contiguous()
        self.features_tensor = torch.as_tensor(np.float32(features[self.feat_cols].to_numpy().T)).contiguous()
        self.M = len(self.feat_cols)

    def __getitem__(self, index):
        # index starts at 0, that why we have index!!!
        return self.features_tensor[:, self.groups_indices == index]

    def __len__(self):
        return len(self.groups_metadata)  # usually number of unique (exp-id, well-id) pairs

    @staticmethod
    def custom_collate_fn(batch):
        """
        batch: list of tensors each of shape (M, Ni) where
            M is the number of feature columns, and
            Ni is the number of cells belonging to an (exp-id, well-id) pair.
        """
        N = len(batch)
        numels = [0] + [batch[ii].shape[1] for ii in range(N)]
        start_indices = np.cumsum(numels[0:-1])
        end_indices = np.cumsum(numels[1:])
        features_tensor = torch.concat(batch, dim=1)
        return features_tensor, start_indices, end_indices

    def get_groups(self, features):
        """Assuming all relevant metadata cols, except exp-id and well-id,
         have been alread fixed properly!

         We want to get feature distance metrics per (exp-id, well-id) pair, but can't
         use strings as tensors in pytorch. so we want to extract the metadata per pair
         as well as get the corresponding index to as a numeric value to be used as reference
         when calculating distances with pytorch.

         groups_meta columns are (group_id, cell_count, exp-id, cell-line, treatment, well-id)
         """
        grp_id = 0
        groups = features[self.group_cols].groupby(self.group_cols, group_keys=False)
        groups_meta = np.zeros((self.num_groups_ub, len(self.group_cols) + 1), dtype=object)
        groups_index = -1 * np.ones((len(features),), dtype=np.int64)
        # loop over the (expid, wellid) unique pairs in the features dataframe
        for ii, (key, slice_) in enumerate(groups):
            # print(grp_id, slice_.index, key)
            cell_count = len(slice_)
            ##########################################################################################
            # Already taken care of in the PostFeatureExtraction.load_and_preprocess_features function
            if cell_count < self.min_well_cell_count:
                continue
            #########################################################################################

            groups_meta[grp_id] = (cell_count,) + tuple(key)
            groups_index[slice_.index] = grp_id  # valid group index starts at 1, not zero
            grp_id += 1

        groups_meta = groups_meta[0:grp_id]
        groups_meta = pd.DataFrame(groups_meta, columns=["cell-count"]+self.group_cols)
        return groups_meta, groups_index


class DistCalcModel(nn.Module):
    """
    if self.step == 4:
        Calculates wassertein-distance from reference distribution (control treatment)
    elif self.step == 5:
        Calculates mean-distance, wassertein-distance, and median-distance from reference distribution (DMSO)
    https://stackoverflow.com/questions/57540745/
    what-is-the-difference-between-register-parameter-and-register-buffer-in-pytorch
    """
    def __init__(self, anchor_features, feat_cols, analysis_step):
        super(DistCalcModel, self).__init__()
        assert analysis_step == 4 or analysis_step == 5, \
            "DistanceMap Calculation only happens in step 4 or step 5 of cellpaint!!!"
        self.step = analysis_step
        self.M = anchor_features.shape[1]
        anchor_features = torch.as_tensor(np.float32(anchor_features[feat_cols].to_numpy()).T)
        self.register_buffer("anchor_features", anchor_features)
        # divisor = 1024*1024
        # size_mb = np.round(sys.getsizeof(self.anchor_features.storage()) / divisor, 2)
        # print(f"anchor_features size in MB inside DistCalcModel: {size_mb}  shape={anchor_features.size()}")
        if self.step == 4:
            self.metrics = ("wasserstein",)
        else:
            self.metrics = ("mean", "wasserstein", "median")

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

    def forward(self, x):
        if self.step == 4:
            return self.wassertein_distance_2d(x, self.anchor_features)
        else:
            # make sure you calculate the distance metrics in order of appearance
            median_dist = torch.median(x, dim=1)[0] - torch.median(self.anchor_features, dim=1)[0]
            median_sign = torch.sign(median_dist)
            mean_dist = torch.mul(torch.mean(x, dim=1) - torch.mean(self.anchor_features, dim=1), median_sign)
            wassertein_dist = torch.mul(self.wassertein_distance_2d(x, self.anchor_features), median_sign)
            return torch.stack((mean_dist, wassertein_dist, median_dist), dim=0)


def dist_calc_fn(model, data_loader, device, analysis_step):
    # divisor = 1024 * 1024
    # nvmlInit()
    N, M = data_loader.dataset.__len__(), data_loader.dataset.M

    if analysis_step == 4:
        # metrics = ("wasserstein",)
        distances = np.zeros((1, N, M), dtype=np.float32)
    elif analysis_step == 5:
        # metrics = ("mean", "wasserstein", "median")
        distances = np.zeros((3, N, M), dtype=np.float32)
    else:
        raise NotImplementedError()

    model.eval()
    with torch.no_grad():
        # for jj, (feats, sids, eids) in tqdm(enumerate(data_loader), total=len(data_loader)):
        for jj, (feats, sids, eids) in enumerate(data_loader):
            feats = feats.to(device)
            # size_mb = np.round(sys.getsizeof(feats.storage()) / divisor, 2)
            # h = nvmlDeviceGetHandleByIndex(0)
            # info = nvmlDeviceGetMemoryInfo(h)
            # print(f"anchor_features size in MB inside DistCalcModel: {size_mb}  shape={feats.size()} "
            #       f"total: {info.total} free: {info.free}   used: {info.used}")

            for ii in range(len(sids)):
                distances[:, jj * data_loader.batch_size + ii, :] = model(feats[:, sids[ii]:eids[ii]]).cpu().numpy()

    if analysis_step == 4:
        distances = distances[0]
    torch.cuda.empty_cache()
    return distances


def select_device_batch_size(features, anchor_features, cell_count):
    """
    TASK 1) To Figure out the device:
        In step 4) We need to know the features.shape and features size in MB because
        it is used as a buffer in torch model

        In step 5) We need to know the anchor_features.shape and anchor_features size in MB because
        it is used as a buffer in torch model

    TASK 2) To Figure out the batch size:
        Since each batch element corresponds to a single well (all cells within a well to be more precise),
        we need to max_cell_count and feature.shape and feature size in megabytes per well.
    """
    pass


class ROCDatasets(TensorDataset):

    def __init__(self, distance_map, feature_types, channels, transform=None):
        super().__init__()
        # self.feature_types = self.args.heatmap_feature_groups[1:]
        # self.channels = self.args.organelles
        self.distance_map = distance_map
        self.feature_types = feature_types
        self.channels = channels
        self.transform = transform

        self.num_feat_cat = len(self.feature_types)
        self.num_cin = len(self.channels)

        # self.xlabels = [f"{it0}_{it1}" for it0 in self.feature_types for it1 in self.channels]
        self.feat_cols = [[cc for cc in distance_map.columns if it1 in cc and it2 in cc]
                          for ii, it1 in enumerate(self.feature_types)
                          for jj, it2 in enumerate(self.channels)]
        # self.feat_cols = list(itertools.chain.from_iterable(self.feat_cols))
        self.max_len = np.amax([len(it) for it in self.feat_cols])
        self.N = len(self.distance_map)
        self.M = len(self.feat_cols)
        self.M1 = len(self.feature_types)
        self.M2 = len(self.channels)

    def __getitem__(self, idx):
        row_id, col_group_id = idx % self.N, idx // self.N
        cols = self.feat_cols[col_group_id]
        # print(idx, row_id, col_group_id, cols)
        n_cols = len(cols)
        data = np.float32(self.distance_map.loc[row_id, cols])
        data = torch.as_tensor(data)
        if self.transform:
            data, n_cols = self.transform((data, n_cols))
        return data, n_cols

    def __len__(self):
        return self.N*self.M  # usually number of unique (exp-id, well-id) pairs


class ZeroPad2D(object):
    def __init__(self, max_len):
        self.max_len = max_len

    def __call__(self, sample):
        len_ = sample[1]
        out = torch.nn.functional.pad(sample[0], (0, self.max_len-len_), mode='constant', value=0)
        return out, len_


class ROCModel(nn.Module):
    """
    """
    def __init__(self, ):
        super(ROCModel, self).__init__()
        self.thresholds = np.linspace(0, 1, 100)
        self.M = len(self.thresholds)

    def forward(self, x, y):
        # print(x.size(), y.size())
        roc_curves = torch.zeros((x.size(0), self.M), dtype=torch.float32).to(x.device)
        roc_deriv_curves = torch.zeros((x.size(0), self.M), dtype=torch.float32).to(x.device)

        for ii in range(0, self.M-1):
            # print(ii, (self.thresholds[ii] < x).size(), x[self.thresholds[ii] < x].size())
            roc_curves[:, ii] = torch.sum(torch.where(self.thresholds[ii] < x, x, 0), dim=1)
            roc_deriv_curves[:, ii] = \
                torch.sum(torch.where((self.thresholds[ii] < x) & (x <= self.thresholds[ii+1]), x, 0), dim=1)
        roc_curves[:, self.M-1] = torch.sum(torch.where(self.thresholds[self.M-1] < x, x, 0), dim=1)
        roc_deriv_curves[:, self.M-1] = (roc_deriv_curves[:, self.M-2]+roc_deriv_curves[:, self.M-3])/2

        roc_curves = roc_curves/y[:, None]
        roc_deriv_curves = roc_deriv_curves/y[:, None]

        roc_aucs = torch.trapezoid(roc_curves, dim=1)
        roc_deriv_aucs = torch.trapezoid(roc_deriv_curves, dim=1)
        return roc_curves, roc_deriv_curves, roc_aucs, roc_deriv_aucs


def roc_calc_fn(model, data_loader, device):
    # divisor = 1024 * 1024
    # nvmlInit()
    N_rows, N, M1, M2, M = data_loader.dataset.__len__(), data_loader.dataset.N, \
        data_loader.dataset.M1, data_loader.dataset.M2, model.M
    roc_curves = np.zeros((N_rows, M), dtype=np.float32)
    roc_deriv_curves = np.zeros((N_rows, M), dtype=np.float32)

    roc_aucs = np.zeros((N_rows,), dtype=np.float32)
    roc_deriv_aucs = np.zeros((N_rows,), dtype=np.float32)
    bs = data_loader.batch_size

    model.eval()
    with torch.no_grad():
        # for jj, (feats, sids, eids) in tqdm(enumerate(data_loader), total=len(data_loader)):
        for jj, (feats, lens) in enumerate(data_loader):
            # print(feats.size(), lens.size())
            feats, lens = feats.to(device), lens.to(device)
            out = model(feats, lens)
            roc_curves[jj*bs:jj*bs+len(feats)], roc_deriv_curves[jj*bs:jj*bs+len(feats)], \
                roc_aucs[jj*bs:jj*bs+len(feats)], roc_deriv_aucs[jj*bs:jj*bs+len(feats)] = \
                out[0].cpu().numpy(), out[1].cpu().numpy(), out[2].cpu().numpy(), out[3].cpu().numpy()
    torch.cuda.empty_cache()
    roc_curves, roc_deriv_curves = roc_curves.reshape((N, M1, M2, M)), roc_deriv_curves.reshape((N, M1, M2, M))
    roc_aucs, roc_deriv_aucs = roc_aucs.reshape((N, M1, M2)), roc_deriv_aucs.reshape((N, M1, M2))

    roc_curves, roc_deriv_curves = roc_curves.transpose((1, 2, 0, 3)), roc_deriv_curves.transpose((1, 2, 0, 3))
    roc_aucs, roc_deriv_aucs = roc_aucs.transpose((1, 2, 0)), roc_deriv_aucs.transpose((1, 2, 0))
    print(roc_curves.shape, roc_aucs.shape)
    return roc_curves, roc_deriv_curves, roc_aucs, roc_deriv_aucs


# t2d = torch.ones(3, 5)
# max_len = 10
# pad = (0, max_len-5, 0, 0)
# out = torch.nn.functional.pad(t2d, pad, "constant", 0)  # effectively zero padding
# print(t2d.size(), out.size())
# print(t2d, '\n', out)