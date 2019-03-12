import matplotlib.pyplot as plt
import pandas as pd

stats_mc = pd.read_csv('stats_mc.csv')

stats_mc.plot()
plt.show()