import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 
import matplotlib.patches as patch
import ClusterStats.cluster_stats as cs

print("Let's get this party started")

def processStats(df,cl,stats,pval):
    gp = df.groupby(['phase_bin', 'stim_on']).closestpeaky.mean()
    gp = gp.to_frame()
    bins_deg = np.linspace(0, 360, 21)[:-1]
    gp = gp.reset_index().pivot(index='phase_bin', columns='stim_on').set_index(bins_deg)
    signifAmps = gp.iloc[cl[0],1].reset_index()
    signifAmps.columns = ['bin','amp']
    signifAmps['bin'] = signifAmps.bin.astype('int')
    
    signif_all = gp.iloc[cl[0]].reset_index()
    signif_all.columns = ['bin','control','stim']
    signif_all['bin'] = signif_all['bin'].astype('int')
    signif_all['stat'] = stats[0]
    signif_all['pval'] = pval[0]

    return signifAmps, signif_all

print("Reading in data")
df = pd.read_csv('downPeakAmpsAndDur.csv')

eNpHR3_90 = df[(df.virus == 'eNpHR3') & (df.stim_dur == 90)]
n_bins = 20
eNpHR3_90['phase_bin'] = 0
eNpHR3_90['phase_bin'] = (eNpHR3_90.loc[:, 'stim_phase'] // (360/20)).copy().astype(np.int)
eNpHR3_90_nanfree = eNpHR3_90.loc[eNpHR3_90['30_width'].dropna().index,:]

print("Now attempting the big stuff")
cl_eNpHR3_90_dur, stats_eNpHR3_90_dur, cl_pval_eNpHR3_90_dur = cs.cluster_stats_pvalue(eNpHR3_90_nanfree, 'stim_on', '30_width', 
                                             n_repetitions=1000, 
                                             site_statistics=cs.site_statistics_ttest_ind_multi_group, 
                                             connectivity='1dcyclic',
                                             col_bins='phase_bin', two_sided=True
                                            )
print("Processing stats")
signifAmps_90, signif_all_90 = processStats(eNpHR3_90_nanfree,cl_eNpHR3_90_dur,stats_eNpHR3_90_dur,cl_pval_eNpHR3_90_dur)

print("Saving results")
signifAmps_90.to_csv('SignifAmps_eNpHR3_90_down_dur_scriptTest.csv', index=False)
signif_all_90.to_csv('All_SignifAmps_eNpHR3_90_down_stats_pval_dur_scriptTest.csv', index=False)
