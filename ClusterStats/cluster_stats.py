import concurrent.futures
import itertools

import cv2
import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import ttest_ind_from_stats


@njit
def calc_mean_var_2groups(a, b):
    """
    Computes the means and standard deviations for two groups
    :param a: a first group ( (Ndata,) or (Ndata, N_columns)
    :param b: a first group ( (Ndata,) or (Ndata, N_columns)
    :return: m1, m2, s1, s2 (the means and standard deviations)
    """
    m1 = np.sum(a, axis=0) / a.shape[0]
    m2 = np.sum(b, axis=0) / b.shape[0]
    a1 = a - m1
    am = np.sum(a1 * a1, axis=0) / (a.shape[0] - 1)
    s1 = np.sqrt(am)
    b1 = b - m2
    bm = np.sum(b1 * b1, axis=0) / (b.shape[0] - 1)
    s2 = np.sqrt(bm)
    return m1, m2, s1, s2


@njit
def calc_mean_var_multi_group(data_here, bins_here, labels_here, unique_labels_here):
    """
    Computes the means and standard deviations for two groups, where the column index (bin) each data belongs to
    is specified in a vector
    :param data_here: the data (as a vector)
    :param bins_here: a vector including the column for each data point
    :param labels_here: a vector indicating the group for each data point
    :param unique_labels_here: the group labels (there must be two of them)
    :return: m1, m2, s1, s2 (the means and standard deviations)
    """
    n_bins = 20
    m1 = np.zeros(n_bins)
    m2 = np.zeros(n_bins)
    s1 = np.zeros(n_bins)
    s2 = np.zeros(n_bins)
    for i in range(n_bins):
        a = data_here[(bins_here == i) & (labels_here == unique_labels_here[0])]
        b = data_here[(bins_here == i) & (labels_here == unique_labels_here[1])]
        m1[i], m2[i], s1[i], s2[i] = calc_mean_var_2groups(a, b)
    return m1, m2, s1, s2


def ttest_with_numba(a, b):
    # print(a)
    # print(b)
    (m1, m2, s1, s2) = calc_mean_var_2groups(a, b)
    return ttest_ind_from_stats(m1, s1, a.shape[0], m2, s2, b.shape[0])


def site_statistics_ttest_ind_multi_group(data_here, labels_here, unique_labels_here, bins_here=None):
    """
    performs an optimized t-test for independent samples where the column index (bin) each data belongs to
    is specified in a vector
    :param data_here: the data (as a vector)
    :param labels_here: a vector indicating the group for each data point
    :param unique_labels_here: the group labels (there must be two of them)
    :param bins_here: a vector including the column for each data point
    :return:
    """
    m1, m2, s1, s2 = calc_mean_var_multi_group(data_here, bins_here, labels_here, unique_labels_here)
    return ttest_ind_from_stats(m1, s1, m1.shape[0], m2, s2, m2.shape[0])


# noinspection PyUnusedLocal
def site_statistics_ttest_ind(data_here, labels_here, unique_labels_here, bins_here=None):
    """
    performs an optimized t-test for independent samples for data in the form (Ndata X Ncolumns)
    :param data_here: the data (as a vector)
    :param labels_here: a vector indicating the group for each data point
    :param unique_labels_here: the group labels (there must be two of them)
    :param bins_here: not used
    :return: t-statistics, pvalue
    """
    groups = [(data_here[labels_here == l, :]).astype(np.float) for l in unique_labels_here]
    tt_result = ttest_with_numba(*groups)
    return tt_result.statistic, tt_result.pvalue


def clust_stats_opencv(stat, pval, threshold=0.05, cyclic=False):
    """
    Finds connected clusters using the algorithm for opencv (works for two dimensional cases) and computes the
    Maris and Oostenveld cluster statistics
    :param stat: the t-statistics per data point
    :param pval: the pvalue per data point
    :param threshold: the site-based threshold
    :param cyclic: if True, the array is considered cyclic (not implemented yet)
    :return: cluster_stats: the cluster statistics, clusters: the clusters
    """
    active_sites = (pval < threshold).astype(np.int8)
    if active_sites.ndim == 1:
        is_1_D = True
        if cyclic:
            raise NotImplementedError("Cyclic not implemented")
    else:
        is_1_D = False
        if cyclic:
            raise ValueError('Cyclic can only be specified for 1D statistics')
    ret, components = cv2.connectedComponents(active_sites)
    if is_1_D:
        components = np.reshape(components, (len(components),))
    clusters = []
    cluster_stats = []
    for l in range(ret):
        lab_l = (components == l)
        if not np.any(active_sites[lab_l] == 0):
            lab_l_int = lab_l.astype(np.int32)
            clusters.append(lab_l_int)
            cluster_stats.append(np.sum(lab_l_int * stat))
    cluster_stats = np.array(cluster_stats)
    ix = np.argsort(cluster_stats)[::-1]
    cluster_stats = cluster_stats[ix]
    clusters = [clusters[i] for i in ix]
    return cluster_stats, clusters


def clust_stats_1d(stat, pval, threshold=0.05, cyclic=False):
    """
    Finds connected clusters for the 1D case and computes the
    Maris and Oostenveld cluster statistics
    :param stat: the t-statistics per data point
    :param pval: the pvalue per data point
    :param threshold: the site-based threshold
    :param cyclic: if True, the array is considered cyclic (not implemented yet)
    :return: cluster_stats: the cluster statistics, clusters: the clusters (as slices)
    """
    active_sites = (pval < threshold).astype(np.int)
    dd = np.diff(active_sites)
    n_start = np.where(dd == 1)[0] + 1

    if active_sites[0] == 1:
        n_start = np.insert(n_start, 0, 0)
    n_stop = np.where(dd == -1)[0] + 1
    if active_sites[-1] == 1:
        n_stop = np.append(n_stop, active_sites.shape[0])
    clusters = [np.s_[n_start[i]:n_stop[i]] for i in range(len(n_start))]

    if cyclic and active_sites[0] == 1 and active_sites[-1] == 1:
        clusters[0] = np.r_[clusters[0], clusters[-1]]
        del clusters[-1]

    cluster_stats = np.array([np.sum(stat[clusters[i]]) for i in range(len(clusters))])
    ix = np.argsort(cluster_stats)[::-1]
    cluster_stats = cluster_stats[ix]
    clusters = [clusters[i] for i in ix]
    return cluster_stats, clusters


def cluster_statistic(data_here: np.numarray, labels_here: np.numarray, unique_labels_here: np.numarray,
                      connectivity='1d', site_alpha=0.05,
                      site_statistics=site_statistics_ttest_ind, bins_here=None):
    """
    computes the Maris and Oostenveld cluster statistics
    :param data_here: the data
    :param labels_here: a vector with labels
    :param unique_labels_here: the unique labels (there must be two of them)
    :param connectivity: can be '1d', '1dcyclic' (for example for polar variables) or '2d' (e.g. spectrograms)
    :param site_alpha: threshold site based, (default 0.05)
    :param site_statistics: the function to be used for the site statistics (default, independent t-test)
    :param bins_here: a vector with the bins data point belong to, in the case that all data are in one vector (e.g.
    for irregular designs
    :return: cluster_stats: the cluster statistics, clusters: the clusters (as slices)
    """
    stat, pval = site_statistics(data_here, labels_here, unique_labels_here, bins_here)

    if connectivity == '1d':
        cluster_stats, clusters = clust_stats_1d(stat, pval, site_alpha)
    elif connectivity == '1dcyclic':
        cluster_stats, clusters = clust_stats_1d(stat, pval, site_alpha, cyclic=True)
    elif connectivity == '2d':
        cluster_stats, clusters = clust_stats_opencv(stat, pval, site_alpha)
    else:
        raise NotImplementedError("connectivity " + connectivity + " not implemented")

    return cluster_stats, clusters


def get_random_seed():
    rf = open('/dev/random', 'rb')
    seed = int.from_bytes(rf.read(4), 'big')
    return seed


data = None
labels = None
unique_labels = None
bins = None


# noinspection PyBroadException
def monte_carlo_iteration(connectivity='1d', site_alpha=0.05,
                          site_statistics=site_statistics_ttest_ind):

    np.random.seed(get_random_seed())
    try:
        labels_here = np.random.permutation(labels.copy())
        cluster_stats, _ = cluster_statistic(data, labels_here, unique_labels, connectivity, site_alpha,
                                             site_statistics=site_statistics, bins_here=bins)
        if len(cluster_stats) == 0:
            return 0
        return cluster_stats[0]
    except BaseException as e:
        raise e


def run_monte_carlo(df, col_groups, col_values, n_repetitions, connectivity='1d', site_alpha=0.05,
                    site_statistics=site_statistics_ttest_ind, parallel=True, col_bins=None):
    """

    :param df: dataframe with the data
    :param col_groups: the column with the group labels
    :param col_values: the column(s) with the data
    :param n_repetitions: how many monte-carlo repetitions
    :param connectivity: can be '1d', '1dcyclic' (for example for polar variables) or '2d' (e.g. spectrograms)
    :param site_alpha: threshold site based, (default 0.05)
    :param site_statistics: the function to be used for the site statistics (default, independent t-test)
    :param parallel: if True, uses concurrent.futures to parallelize monte-carlo execution
    :param col_bins: the column with the bins data point belong to, in the case that all data are in one vector (e.g.
    for irregular designs, default is None for (Ndata X Ncolumns) design
    :return: stats_mc, a sorted series with the statistics value for the monte-carlo trials
    """
    global data
    global labels
    global unique_labels
    global bins

    data = df[col_values].values
    labels = df[col_groups].values
    unique_labels = np.unique(labels).astype(np.int)[::-1]
    if col_bins:
        bins = df[col_bins].values
    else:
        bins = None

    if parallel:
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            stats_mc = executor.map(monte_carlo_iteration,
                                    *[itertools.repeat(d, n_repetitions) for d in (connectivity,
                                                                                   site_alpha, site_statistics)],
                                    chunksize=1250)
    else:
        stats_mc = map(monte_carlo_iteration,
                       *[itertools.repeat(d, n_repetitions) for d in (connectivity,
                                                                      site_alpha, site_statistics)])
    stats_mc = np.array(list(stats_mc))
    stats_mc.sort()
    stats_mc = stats_mc[::-1]
    idx = np.linspace(0, 1, n_repetitions)
    stats_mc = pd.Series(data=stats_mc, index=idx)
    return stats_mc


def cluster_stats_pvalue(df, col_groups, col_values, n_repetitions, connectivity='1d', site_alpha=0.05,
                         site_statistics=site_statistics_ttest_ind, parallel=True, col_bins=None, two_sided=False):
    """

    :param df: dataframe with the data
    :param col_groups: the column with the group labels
    :param col_values: the column(s) with the data
    :param n_repetitions: how many monte-carlo repetitions
    :param connectivity: can be '1d', '1dcyclic' (for example for polar variables) or '2d' (e.g. spectrograms)
    :param site_alpha: threshold site based, (default 0.05)
    :param site_statistics: the function to be used for the site statistics (default, independent t-test)
    :param parallel: if True, uses concurrent.futures to parallelize monte-carlo execution
    :param col_bins: the column with the bins data point belong to, in the case that all data are in one vector (e.g.
    for irregular designs, default is None for (Ndata X Ncolumns) design
    :param two_sided: if True the test is taken to be two sided
    :return: clusters_obs: the observed clusters, cluster_stats_obs: the cluster statistics,
    cluster_pval_obs: the cluster p-value
    """
    data_here = df[col_values].values
    labels_here = df[col_groups].values
    unique_labels_here = np.unique(labels_here)
    unique_labels_here.sort()
    unique_labels_here = unique_labels_here[::-1]
    if col_bins:
        bins_here = df[col_bins].values
    else:
        bins_here = None
    if len(unique_labels_here) != 2:
        raise ValueError("there can be only two groups in the data")
    cluster_stats_obs, clusters_obs = cluster_statistic(data_here, labels_here, unique_labels_here, connectivity,
                                                        site_alpha,
                                                        site_statistics,
                                                        bins_here=bins_here)

    stats_mc = run_monte_carlo(df, col_groups, col_values, n_repetitions, connectivity, site_alpha, site_statistics,
                               parallel, col_bins=col_bins)

    cluster_pval_obs = []

    for i, stats_obs in enumerate(cluster_stats_obs):
        try:
            if two_sided:
                stats_obs = np.abs(stats_obs)
            pval = stats_mc[stats_mc < stats_obs].head(1).index[0]
            if pval == 0:
                pval = 1. / len(stats_mc)
            if two_sided:
                pval *= 2
            if pval > 1:
                pval = 1

            cluster_pval_obs.append(pval)
        except IndexError:
            cluster_pval_obs.append(np.nan)
    return clusters_obs, cluster_stats_obs, cluster_pval_obs


if __name__ == '__main__':
    pass
