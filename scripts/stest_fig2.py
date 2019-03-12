import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import ClusterStats.cluster_stats as cs
import os


os.chdir(os.path.expanduser('~/Dropbox/EstherProject/data/'))

df_now = pd.read_csv('fig2_stats.csv')

cl, stats, cl_pval = cs.cluster_stats_pvalue(df_now, 'stim_on', 'closestpeaky',
                                             n_repetitions=10000,
                                             site_statistics=cs.site_statistics_ttest_ind_multi_group,
                                             connectivity='1dcyclic',
                                             col_bins='phase_bin', two_sided=True
                                            )