
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st
from scipy.stats import t
import math
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.cm import ScalarMappable
np.random.seed(12345)
df = pd.DataFrame([np.random.normal(32000,200000,3650),
np.random.normal(43000,100000,3650),
np.random.normal(43500,140000,3650),
np.random.normal(48000,70000,3650)],
index=[1992,1993,1994,1995])
df1 = df.transpose()
df1.std()
labels = ['1992', '1993', '1994', '1995']
value = 40000
fig, ax = plt.subplots()
xvals = range(len(df))
yvals = df.mean(axis = 1)
cmap = mpl.colors.LinearSegmentedColormap.from_list('blue_towhiteto_red', (['darkblue','white', 'darkred']))
norm = plt.Normalize(yvals.min(), yvals.max())
colors = cmap(norm(yvals))
y_std = df1.std()/np.sqrt(df1.shape[0])*1.96
ax.bar(xvals, yvals, yerr=y_std, width = 1, capsize=15,
edgecolor = 'black',linewidth = .5, color = colors)
plt.xticks(xvals, labels)
plt.subplots_adjust(bottom=0.25)
ax.axhline(value, color = 'black', linestyle = '-', linewidth = .5, label = value)
plt.legend()
plt.gcf().set_size_inches(9, 8)
cpick = cm.ScalarMappable(cmap = cmap, norm = norm)
cpick.set_array([])
plt.colorbar(cpick, orientation = 'vertical')
plt.ylabel('Values')
plt.xlabel('Date Range')
plt.title('Building a Custom Visualization')
plt.tight_layout()
plt.show()