import numpy as np
import pandas as pd

from numba import njit
from scipy.stats import ttest_ind_from_stats
import concurrent.futures
import itertools

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
    (m1, m2, s1, s2) = calc_mean_var_2groups(a, b)
    return ttest_ind_from_stats(m1, s1, a.shape[0], m2, s2, b.shape[0])


def site_statistics_ttest_ind(df, col_groups, col_values):
    """

    :param df:
    :param col_groups:
    :param col_values:
    :return:
    """

    if df[col_groups].n_unique() != 2:
        raise ValueError(col_groups + ' must be a binary column (2 groups)')

    labels = df[col_groups].unique()
    data = df[col_values].values
    groups = [data[df[col_groups] == l] for l in labels]
    tt_result = ttest_with_numba(*groups)
    return tt_result.statistic, tt_result.pvalue


def find_max_clust_stat_1d(stat, pval, threshold=0.05, cyclic=False):
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

    cluster_stats = [np.sum(stat[clusters[i]]) for i in range(len(clusters))]
    ix = np.argsort(cluster_stats)[::-1]
    cluster_stats = cluster_stats[ix]
    clusters = clusters[ix]
    return cluster_stats, clusters


def cluster_statistic(df, col_groups, col_values, connectivity='1d', site_alpha=0.05,
                      site_statistics=site_statistics_ttest_ind):
    stat, pval = site_statistics(df, col_groups, col_values)

    if connectivity == '1d':
        cluster_stats, clusters = find_max_clust_stat_1d(stat, pval, site_alpha)
    elif connectivity == '1dcyclic':
        cluster_stats, clusters = find_max_clust_stat_1d(stat, pval, site_alpha, cyclic=True)
    else:
        raise NotImplementedError("connectivity " + connectivity + " not implemented")

    return cluster_stats, clusters


def monte_carlo_iteration(df, col_groups, col_values, connectivity, site_alpha, site_statistics):
    df[col_groups] = np.random.permutation(df[col_groups])
    cluster_stats, _ = cluster_statistic(df, col_groups, col_values, connectivity, site_alpha,
                      site_statistics=site_statistics)
    return cluster_stats[0]


def run_monte_carlo(df, col_groups, col_values, connectivity, site_alpha, site_statistics, n_repetitions):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        stats_mc = executor.map(lambda x: monte_carlo_iteration(*x),
                                *[itertools.repeat(d, n_repetitions) for d in (df, col_groups, col_values, connectivity,
                                                  site_alpha, site_statistics)],
                                chunksize=100)
    stats_mc = np.array(list(stats_mc))
    stats_mc.sort()
    stats_mc = stats_mc[::-1]
    idx = np.linspace(0, 1, n_repetitions)
    stats_mc = pd.Series(data=stats_mc, idx=idx)
    return stats_mc



