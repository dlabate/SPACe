import torch
import torch.nn as nn

import numpy as np
import scipy.stats as scstats

import string
from typing import Optional


def alpha_geodesic(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float,
    lmd: float
) -> torch.Tensor:
    r"""
    Taken from the official implementation of "α-Geodesical Skew Divergence".
    https://github.com/nocotan/geodesical_skew_divergence/blob/main/gs_divergence/geodesical_skew_divergence.py
    $\alpha$-geodesic between two probability distributions
    """

    a_ = a + 1e-12
    b_ = b + 1e-12
    if alpha == 1:
        return torch.exp((1 - lmd) * torch.log(a_) + lmd * torch.log(b_))
    elif alpha >= 1e+9:
        return torch.min(a_, b_)
    elif alpha <= -1e+9:
        return torch.max(a_, b_)
    else:
        p = (1 - alpha) / 2
        lhs = a_ ** p
        rhs = b_ ** p
        g = ((1 - lmd) * lhs + lmd * rhs) ** (1/p)

        if alpha > 0 and (g == 0).sum() > 0:
            return torch.min(a_, b_)

        return g


class GSDivLoss(nn.Module):
    r"""
    Taken from the official implementation of "α-Geodesical Skew Divergence":
    https://github.com/nocotan/geodesical_skew_divergence/blob/main/gs_divergence/geodesical_skew_divergence.py


    The alpha-geodesical skew divergence loss measure
    `alpha-geodesical skew divergence`_ is a useful distance measure for continuous
    distributions and is approximation of the 'Kullback-Leibler divergence`.
    This criterion expects a `target` `Tensor` of the same size as the
    `input` `Tensor`.
    """
    def __init__(
        self,
        alpha: float = -1,
        lmd: float = 0.5,
        dim: int = 0,
        reduction: Optional[str] = 'sum') -> None:

        self.alpha = alpha
        self.lmd = lmd
        self.reduction = reduction

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor) -> torch.Tensor:

        return gs_div(input, target, self.alpha, self.lmd, self.dim, self.reduction)


def gs_div(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float = -1,
    lmd: float = 0.5,
    dim: int = 0,
    reduction: Optional[str] = 'sum',
) -> torch.Tensor:
    r"""
    Taken from the official implementation of "α-Geodesical Skew Divergence":
    https://github.com/nocotan/geodesical_skew_divergence/blob/main/gs_divergence/geodesical_skew_divergence.py


    The $\alpha$-geodesical skew divergence.
    Args:
        input: Tensor of arbitrary shape
        target: Tensor of the same shape as input
        alpha: Specifies the coordinate systems which equiped the geodesical skew divergence
        lmd: Specifies the position on the geodesic
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'batchmean'`` | ``'sum'`` | ``'mean'``.
            ``'none'``: no reduction will be applied
            ``'batchmean``': the sum of the output will be divided by the batchsize
            ``'sum'``: the output will be summed
            ``'mean'``: the output will be divided by the number of elements in the output
            Default: ``'sum'``
    """

    assert lmd >= 0 and lmd <= 1

    skew_target = alpha_geodesic(input, target, alpha=alpha, lmd=lmd)
    div = input * torch.log(input / skew_target + 1e-12)
    if reduction == 'batchmean':
        div = torch.sum(div, dim=dim) / input.size()[0]
    elif reduction == 'sum':
        div = torch.sum(div, dim=dim)
    elif reduction == 'mean':
        div = torch.mean(div, dim=dim)

    return div


def apply_along_axis(function, x, axis):
    """https://discuss.pytorch.org/t/apply-a-function-along-an-axis/130440"""
    # print("x in apply_along_...: ", x.device)
    return torch.stack([function(x_i) for x_i in torch.unbind(x, dim=axis)], dim=axis)


def get_densities(xs, bins=25, device="cuda:0"):
    """get the probability density function of xs,
    Let's figure out whether to use samples from probability density or
     probability vectors later.


    """
    # TODO: Fix the device parameter
    # Like torch.histogram, but works with cuda
    min, max = xs.min(), xs.max()
    counts = torch.histc(xs, bins=bins, min=min, max=max)
    # boundaries = torch.linspace(min, max, bins+1).to(device)
    # hist = counts/torch.diff(boundaries)/torch.sum(counts)
    # return counts/torch.diff(boundaries)/torch.sum(counts)
    # hist_np, bin_edges = np.histogram(xs.cpu().numpy(), bins=bins, density=True)
    # print(torch.sum(hist), np.sum(hist_np), np.sum(hist_np*np.diff(bin_edges)))

    # counts = torch.histc(xs, bins=25, min=-1, max=1)
    return counts / torch.sum(counts)


def alpha_geodesical_divergence_symmetric(first, second, alpha=-1, lmd=0.5):
    r"""
    adapted from the official implementation of "α-Geodesical Skew Divergence":
    https://github.com/nocotan/geodesical_skew_divergence/blob/main/gs_divergence/geodesical_skew_divergence.py
    https://github.com/scipy/scipy/blob/v1.9.3/scipy/stats/_stats_py.py#L8675-L8749

    The $\alpha$-geodesical skew divergence.
    Args:
        first: Tensor of arbitrary shape (M, N) where M is the number of features, and N is the number of datapoints
        second: Tensor of the same shape as first
        alpha: Specifies the coordinate systems which equiped the geodesical skew divergence
        lmd: Specifies the position on the geodesic
    """
    assert lmd >= 0 and lmd <= 1
    first_density = apply_along_axis(lambda x: get_densities(x), first, axis=0)
    second_density = apply_along_axis(lambda x: get_densities(x), second, axis=0)

    # first_density = torch.histogramdd(first, bins=20, density=True)[0]
    # second_density = torch.histogramdd(second, bins=20, density=True)[0]
    # print(first_density.shape, second_density.shape)
    skew_first_density = alpha_geodesic(second_density, first_density, alpha=alpha, lmd=lmd)
    skew_second_density = alpha_geodesic(first_density, second_density, alpha=alpha, lmd=lmd)
    div = .5 * torch.abs(first_density * torch.log(first_density / skew_second_density + 1e-12)) + \
          .5 * torch.abs(second_density * torch.log(second_density / skew_first_density + 1e-12))
    div = torch.mean(div, dim=1)
    # div *= torch.log(div + 1e-12)
    return div


def cdf_distance_np(u_values, v_values):
    r"""
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
    u_sorter = np.argsort(u_values)
    v_sorter = np.argsort(v_values)

    all_values = np.concatenate((u_values, v_values))
    all_values.sort(kind='mergesort')

    # Compute the differences between pairs of successive values of u and v.
    deltas = np.diff(all_values)
    # Get the respective positions of the values of u and v among the values of
    # both distributions.
    u_cdf_indices = u_values[u_sorter].searchsorted(all_values[:-1], 'right')
    v_cdf_indices = v_values[v_sorter].searchsorted(all_values[:-1], 'right')

    # Calculate the CDFs of u and v using their weights, if specified.
    u_cdf = u_cdf_indices / u_values.size
    v_cdf = v_cdf_indices / v_values.size

    # Compute the value of the integral based on the CDFs.
    # If p = 1 or p = 2, we avoid using np.power, which introduces an overhead
    # of about 15%.
    return np.sum(np.multiply(np.abs(u_cdf - v_cdf), deltas))


def cdf_distance_torch_1d(u_values, v_values):
    """
    pytorch implementation of the wassertein distance between two distributions.

    u_values: (N1, ) tensor where
    N1 is the number of rows/cells

    v_values: (M2,) anchor tensor where
    N2 is the number of rows/cells in the anchor condition
    """
    # u_sorter = torch.argsort(u_values)
    # v_sorter = torch.argsort(v_values)

    all_values = torch.cat((u_values, v_values))
    all_values, _ = torch.sort(all_values)
    u_sorted, _ = torch.sort(u_values)
    v_sorted, _ = torch.sort(v_values)
    # Compute the differences between pairs of successive values of u and v.
    deltas = torch.diff(all_values)
    # Get the respective positions of the values of u and v among the values of
    # both distributions.
    u_cdf_indices = torch.searchsorted(u_sorted, all_values[:-1], right=True)
    v_cdf_indices = torch.searchsorted(v_sorted, all_values[:-1], right=True)
    # Calculate the CDFs of u and v using their weights, if specified.
    u_cdf = u_cdf_indices / u_values.nelement()
    v_cdf = v_cdf_indices / v_values.nelement()
    # Compute the value of the integral based on the CDFs.
    # If p = 1 or p = 2, we avoid using np.power, which introduces an overhead
    # of about 15%.
    return torch.sum(torch.mul(torch.abs(u_cdf - v_cdf), deltas))


def get_distance_matrix_wellwise_np(features, anchor_features, start_index):
        # get the list of features
        feat_cols = list(features.columns)[start_index:]
        # ensuring unique well-ids are taken from the specific col and not all the features
        anchor_features = anchor_features[feat_cols]
        # group them based on each well
        groups = features[
            ["exp-id", "well-id", ] + feat_cols].groupby(
            ["exp-id", "well-id", ], group_keys=False)
        print("started distance matrix calculations ...")
        print("features shape: ", features.shape)
        print("anchor_features shape: ", anchor_features.shape)

        # get median and mean of each feature
        mean_df = groups.mean(numeric_only=True)
        median_dist = groups.median(numeric_only=True).to_numpy()
        well_ids = mean_df.index
        mean_dist = mean_df.to_numpy()

        # get median and mean of each feature for dmso data frame
        anchor_median = anchor_features.median(axis=0, ).to_numpy()
        anchor_mean = anchor_features.mean(axis=0, ).to_numpy()
        anchor_features = anchor_features.to_numpy()

        # get signed median and mean distances for each feature
        median_dist = np.subtract(median_dist, anchor_median)
        sign_mat = np.sign(median_dist)
        mean_dist = sign_mat * np.subtract(mean_dist, anchor_mean)

        # get signed distribution distances for each feature
        distrib_dist = groups.apply(lambda x: get_wass_dist(x, anchor_features))
        distrib_dist = sign_mat * np.vstack(distrib_dist)
        return well_ids, distrib_dist, median_dist, mean_dist


def get_wass_dist(x, dmso_features):
    """
    x is an (N, num_features+1) pandas dataframe where the first column is the well_id,
    and the rest of the columns are feature columns. x contains info of all the cells belonging
    to a specific well.

    dmso_features is an (M, num_features)
    numpy array where the columns are the DMSO (our reference/zero compound/treatment) features.
    """
    feature_cols = list(x.columns)[1:]
    x = x[feature_cols].to_numpy()
    N = len(feature_cols)
    dists = np.zeros((N,), dtype=np.float32)
    for ii in range(N):
        dists[ii] = scstats.wasserstein_distance(x[:, ii], dmso_features[:, ii])
    return dists


def filter_dark_cells_fn(w0_features, filter_dark_cells, nucleus_mean_intensity_thresh):
    """To take care of illumination issues in some of the experiments. Mainly, the ones with Griener
    plating protocol."""

    if filter_dark_cells:
        return w0_features['Nucleus_Intensities_mean'].to_numpy() > nucleus_mean_intensity_thresh
    return np.ones((w0_features.shape[0],), dtype=bool)


def remove_outer_wells_fn(self, all_features, ):
    """To take care of illumination issues in some of the experiments. Mainly, the ones with Griener
    plating protocol."""
    if self.args.remove_outer_wells:
        rows = list(string.ascii_uppercase[:16])
        cols = [str(ii).zfill(2) for ii in np.arange(1, 25)]
        outer_wells = [row + col for row in ['A', 'P', 'H'] for col in cols] + \
                      [row + col for col in [cols[0], cols[-1]] for row in rows]
        all_features = all_features[~all_features['well-id'].isin(outer_wells)]
        # print(all_features.shape)
    return all_features


def main():
    ########################################################################################
    #############################################################################
    # N1 = 5000
    # N2 = 20000
    # M = 600
    # np.random.seed(10)
    # a = 1 + 3 * np.random.randn(M, N1).astype(np.float32)
    # np.random.seed(20)
    # b = 2 + -1 * np.random.randn(M, N2).astype(np.float32)
    # aa = torch.from_numpy(a).to("cuda")
    # bb = torch.from_numpy(b).to("cuda")
    #
    # ans = np.zeros(M, )
    # ans_1d = torch.zeros(M, device="cuda")
    # st = time.time()
    # ans_2d = cdf_distance_torch_2d(aa, bb)
    # print(f"torch 2d took : {time.time()-st}", )
    #
    # st = time.time()
    # for jj in range(M):
    #     ans_1d[jj] = cdf_distance_torch_1d(aa[jj], bb[jj])
    # print(f"torch 1d took : {time.time()-st}", )
    #
    # st = time.time()
    # for jj in range(M):
    #     ans[jj] = cdf_distance_np(a[jj], b[jj])
    # print(f"numpy took 1d took : {time.time()-st}", )
    #
    # print('\n')
    # for jj in range(M):
    #     print(f"{ans[jj]}  {ans_1d[jj]}   {ans_2d[jj]}")
    #
    # print('\n')
    # #######################################################################################
    # ##########################################################################################
    # # search-sorted 2d version in pytorch. seach-sorted 2d is not supported in numpy.
    # # The two tensors must agree in the first dimension. Otherwise, this function does not work.
    # a = torch.randn(3, 5)
    # b, idx = torch.sort(a, dim=0)
    # print(a, '\n', idx, '\n', b)

    # values = torch.tensor([
    #     [1, 1, 2, 3, 5, 7, 9, 9, 10, 10, ],
    #     [1, 2, 2, 4, 5, 5, 6, 7, 8, 10]])
    # sorted_sequence = torch.tensor([[3, 6, 9], [1, 5, 8]])
    #
    # # sorted_sequence = torch.transpose(sorted_sequence, 0, 1)
    # # values = torch.transpose(values, 0, 1)
    #
    # print(sorted_sequence.size(), values.size())
    # print(sorted_sequence)
    # print(values)
    # print(torch.searchsorted(sorted_sequence, values, right=True))
    # print('\n')
    # ################################################################################################
    # # using index to access a tensor
    # a = torch.randn(6, 2)
    # b = a[[1, 3, 5]]
    # print(a)
    # print(b)

    # a = torch.randint(0, 4, (3, 200))
    # b = torch.randint(0, 4, (3, 5))
    #
    # # wd = cdf_distance_torch_2d(a, b)
    #
    # # ahist, _ = torch.histogram(a.type('torch.FloatTensor'), bins=20, density=True, out=None)
    # # print(a.shape, ahist.shape)
    # ahist = apply_along_axis(
    #     lambda x: torch.histogram(x.type('torch.FloatTensor'), bins=20, density=True, out=None)[0],
    #     a, axis=0)
    # print(a.shape, ahist.shape)

    # print(12/(2*5))
    a = torch.randint(0, 4, (200, 3)).type('torch.FloatTensor')  # .to("cuda:0")
    b = torch.histogramdd(a, bins=10, density=True)
    print(a, "\n")
    print(b, "\n")

    a = np.arange(20)
    hist, bin_edges = np.histogram(a, bins=3, density=True)
    print(hist)
    print(hist.sum())
    print(np.sum(hist * np.diff(bin_edges)))

def get_colors(self, metadata_unix, plot_type):
    # TODO: Choose better colormaps
    N1 = len(metadata_unix)
    if N1 <= 8:
        colors = ["red", "green", "blue", "orange", "purple", "pink", "yellow", "gray"]
    else:
        M1 = int(N1/4)
        c1 = list(plt.cm.rainbow(np.linspace(.1, .5, M1)))
        c2 = list(plt.cm.magma(np.linspace(.5, 1, M1)))
        c3 = list(plt.cm.viridis(np.linspace(0, .4, M1)))
        c4 = list(plt.cm.inferno(np.linspace(.4, 1, N1-3*M1)))
        colors = c1+c2+c3+c4
        # print("colors: ", len(colors))
        # random.shuffle(colors)

    # unix = list(np.unique(list(zip(*metadata_unix))[1]))
    # print(metadata_unix, unix)
    if plot_type == "typeI":
        print(metadata_unix[:, 1])
        unix = list(np.unique(metadata_unix[:, 1]))
        print("get colors: ", unix, unix.index(self.args.anchor_treatment))
        colors[unix.index(self.args.anchor_treatment)] = "black"
    elif plot_type == "typeII":
        unix = list(np.unique(metadata_unix[:, ]))
        colors[unix.index(self.args.anchor_cellline)] = "black"
    else:
        raise ValueError(f"{plot_type} not acceptable!!!")
    # print("# unix", N1)
    # colors = colors[0: len(metadata_unix)]
    return colors

if __name__ == "__main__":
    main()