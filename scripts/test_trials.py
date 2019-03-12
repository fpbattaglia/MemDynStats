import pandas as pd
import numpy as np
import ClusterStats.cluster_stats as cs
from datetime import datetime


df = pd.read_csv('trials.h5')
data_cols = [str(i) for i in np.linspace(-10, 10, 100).tolist()]
startTime = datetime.now()
stats_mc = cs.run_monte_carlo(df, 'cond', data_cols, n_repetitions=10000, parallel=True)
print(datetime.now() - startTime)
stats_mc.to_csv('stats_mc.csv')