import concurrent.futures
import itertools

import cv2
import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import ttest_ind_from_stats


@njit
def calc_mean_var_2groups(a, b):
    m1 = np.sum(a, axis=0) / a.shape[0]
    m2 = np.sum(b, axis=0) / a.shape[0]
    a1 = a - m1
    am = np.sum(a1 * a1, axis=0) / (a.shape[0] - 1)
    s1 = np.sqrt(am)
    b1 = b - m1
    bm = np.sum(b1 * b1, axis=0) / (a.shape[0] - 1)
    s2 = np.sqrt(bm)
    return m1, m2, s1, s2


def ttest_with_numba(a, b):
    # print(a)
    # print(b)
    (m1, m2, s1, s2) = calc_mean_var_2groups(a, b)
    return ttest_ind_from_stats(m1, s1, a.shape[0], m2, s2, b.shape[0])


def site_statistics_ttest_ind(data_test, labels_here, unique_labels_here):
    """

    :param data_test:
    :param labels_here:
    :param unique_labels_here:
    :return:
    """
    groups = [(data_test[labels_here == l, :]).astype(np.float) for l in unique_labels_here]
    tt_result = ttest_with_numba(*groups)
    return tt_result.statistic, tt_result.pvalue


def clust_stats_opencv(stat, pval, threshold=0.05, cyclic=False):
    active_sites = (pval < threshold).astype(np.int8)
    if active_sites.ndim == 1:
        is_1_D = True
        if cyclic:
            raise NotImplementedError("Cyclic not implemented")  # TODO
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
    active_sites = (pval < threshold).astype(np.int)
    dd = np.diff(active_sites)
    n_start = np.where(dd == 1)[0] + 1

    if active_sites[0] == 1:
        np.insert(n_start, 0, 0)
    n_stop = np.where(dd == -1)[0] + 1
    if active_sites[-1] == 1:
        np.append(n_stop, active_sites.shape[0] + 1)
    clusters = [np.s_[n_start[i]:n_stop[i]] for i in range(len(n_start))]

    if cyclic and active_sites[0] == 1 and active_sites[-1] == 1:
        clusters[0] = np.r_[clusters[0], clusters[-1]]
        del clusters[-1]

    cluster_stats = np.array([np.sum(stat[clusters[i]]) for i in range(len(clusters))])
    ix = np.argsort(cluster_stats)[::-1]
    cluster_stats = cluster_stats[ix]
    clusters = [clusters[i] for i in ix]
    return cluster_stats, clusters


def cluster_statistic(data_here, labels_here, unique_labels_here, connectivity='1d', site_alpha=0.05,
                      site_statistics=site_statistics_ttest_ind):
    stat, pval = site_statistics(data_here, labels_here, unique_labels_here)

    if connectivity == '1d':
        cluster_stats, clusters = clust_stats_1d(stat, pval, site_alpha)
    elif connectivity == '1dcyclic':
        cluster_stats, clusters = clust_stats_1d(stat, pval, site_alpha, cyclic=True)
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


# noinspection PyBroadException
def monte_carlo_iteration(connectivity='1d', site_alpha=0.05,
                          site_statistics=site_statistics_ttest_ind):
    np.random.seed(get_random_seed())
    try:
        labels_here = np.random.permutation(labels.copy())
        cluster_stats, _ = cluster_statistic(data, labels_here, unique_labels, connectivity, site_alpha,
                                             site_statistics=site_statistics)
        if len(cluster_stats) == 0:
            return 0
        return cluster_stats[0]
    except BaseException:
        return np.nan


def run_monte_carlo(df, col_groups, col_values, n_repetitions, connectivity='1d', site_alpha=0.05,
                    site_statistics=site_statistics_ttest_ind):
    global data
    global labels
    global unique_labels

    data = df[col_values].values
    labels = df[col_groups].values
    unique_labels = np.unique(labels).astype(np.int)[::-1]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        stats_mc = executor.map(monte_carlo_iteration,
                                *[itertools.repeat(d, n_repetitions) for d in (connectivity,
                                                                               site_alpha, site_statistics)],
                                chunksize=100)
    # concurrent.futures.wait(stats_mc)
    stats_mc = np.array(list(stats_mc))
    stats_mc.sort()
    stats_mc = stats_mc[::-1]
    idx = np.linspace(0, 1, n_repetitions)
    stats_mc = pd.Series(data=stats_mc, index=idx)
    return stats_mc


if __name__ == '__main__':
    pass
