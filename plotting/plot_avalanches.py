import glob
import pickle

import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
 
from matplotlib import cm
from matplotlib import colors
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde as kde
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Code for creating log-log plot of spike sizes under domain transfer
# Figure 3C in the manuscript.

# First, collect "avalanche" data from domain_transfer.py in --analysis mode. 
# with function "compute_avalanches".

def best_fit(X, Y):
    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)
   
    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2
   
    b = numer / denum
    a = ybar - b * xbar
   
    print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))
   
    return(a,b)

def plot_avalanches(layer, dataset):

    plt.figure(figsize=(6,5))
    colors = {0.10:'#dc2543', 0.25:'#edbc7a', 0.50:'#0384bd'}

    for threshold in [0.10, 0.25, 0.50]:
        histo = np.zeros(2000)
        for seed in range(1,25):

            files = \
                glob.glob("../Avalanche*Layer={0}*Thresh={1}*Dataset={2}*Model={3}.npy".format(
                layer,
                threshold, 
                dataset,
                seed))

            for f in files:
                try:
                    data = np.load(f, allow_pickle=True)
                except:
                    data = np.load(f)
                histo += data[0]

        x = []
        y = []
   
        for h in range(len(histo)):
            if histo[h] != 0:
                x.append(np.log10(data[1][h+1]))
                y.append(np.log10(histo[h]))
        idx = np.array([0,1,2] + list(np.arange(2,len(x), 2)))
        x = np.array(x)
        y = np.array(y)
        plt.scatter(x=x, 
                    y=y, 
                    color=colors[threshold], 
                    alpha=0.9, 
                    s=100, 
                    marker='*', 
                    edgecolor='black',
                    zorder=2, 
                    linewidth=0.15)

        a,b = best_fit(x,y)
        end = np.where(np.log10(data[1][1:]) == x[-1])[0][0]
        yfit = [a+b*xi for xi in np.log10(data[1][1:end])]
        plt.plot(np.log10(data[1][1:end]), 
                 yfit, 
                 zorder=3, 
                 color=colors[threshold], 
                 linewidth=2, 
                 linestyle='solid')

    plt.ylim(-0.5,8.5)

plot_avalanches(layer='C3', dataset='imagenet')
plt.tight_layout()
plt.show()


