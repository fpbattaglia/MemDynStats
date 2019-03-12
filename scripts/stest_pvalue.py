import ClusterStats.cluster_stats as cs
import pandas as pd
import numpy as np

# or read from disk
df = pd.read_csv('../jupyter/trials.csv')
data_cols = [str(i) for i in np.linspace(-10, 10, 100).tolist()]

cl, stats, pval = cs.cluster_stats_pvalue(df, 'cond', data_cols, n_repetitions=10000, parallel=False)

print(pval)