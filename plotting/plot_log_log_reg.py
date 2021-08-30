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

# Code for creating log-log plot of regulation under domain transfer
# Figure 3D,E in the manuscript.

# First, collect count data from domain_transfer.py in --analysis mode

def plot_log_log_reg(layer, dataset, richORscarce='Rich'):

    plt.figure(figsize=(6,5))
    c = 0
    for bias in [0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12]:
        files = glob.glob("TSAR_Rich_Bias={0}_Seed=*_Dataset=imagenet.npy".format(float(bias)))
        champion_score = 0.0
        champion_run = 999
        for f in files:
            performance = np.mean(np.load(f, allow_pickle=True))
            print(performance)
            if performance > champion_score:
                champion_score = performance
                champion_run = f.split('Seed=')[1].split('_')[0]
            else:
                next
   
        data = np.load("{0}*Bias={1}*SignalCounts*Layer={2}_{3}_Model={4}.npy".format(
                richORscarce,
                float(bias), 
                layer, 
                dataset,
                champion_run), 
                allow_pickle=True)

        bins = data[1][1:]
        data = data[0]
        x = []
        y = []

        for d in range(len(data)):
            if data[d] != 0:
                y.append(data[d])
                x.append(bins[d])

        x = np.log10(x)
        y = np.log10(y)
        if bias in [0,-1,-2,-3,-4,-5]:
            s_color = mapcolors[-1]
            l_color = mapcolors[-2]
            zorder= 1
            s = 20
            marker = "o"
        elif bias in [-12, -11]:
            s_color = 'teal'
            l_color = 'teal'#mapcolors[4]
            zorder = 2
            s = 20
            marker='o'
        else:
            s_color = mapcolors[2]
            l_color = mapcolors[0]
            zorder = 2
            s = 20
            marker='o'
    
        s_alpha = 0.15
        l_alpha = 0.8
        lwidth=1
        style='dashed'
        idx = np.array(list(np.arange(0,10,1)) + list(np.arange(0,1000,4)))

        if bias in [0,-1,-2,-3,-4,-5,-11,-12]:
            plt.scatter(x=x, 
                        y=y, 
                        color=s_color, 
                        s=s, 
                        marker=marker, 
                        zorder=zorder, 
                        alpha=s_alpha, 
                        facecolor='none')
        else:
            plt.scatter(x=x, 
                        y=y, 
                        color=s_color, 
                        s=s, 
                        marker=marker, 
                        zorder=zorder, 
                        alpha=s_alpha, 
                        facecolor='none')
   
        plt.scatter(x=x[1], 
                    y=y[1], 
                    color=s_color, 
                    s=s, 
                    marker=marker, 
                    zorder=zorder, 
                    alpha=s_alpha, 
                    facecolor='none')
        
        plt.scatter(x=x[0], 
                    y=y[0], 
                    color=s_color, 
                    s=s, 
                    marker=marker, 
                    zorder=2, 
                    alpha=0.8)
        
        plt.scatter(x=x[-2:], 
                    y=y[-2:], 
                    color=s_color, 
                    s=s, 
                    marker=marker, 
                    zorder=2, 
                    alpha=0.8)

        a,b = best_fit(x,y)
        yfit = [a+b*xi for xi in np.log10(bins)]
        plt.plot(np.log10(bins), 
                 yfit, 
                 zorder=zorder, 
                 color=l_color, 
                 linewidth=lwidth, 
                 alpha=l_alpha, 
                 linestyle=style)

        c += 1 
    ax = plt.gca()
    plt.ylim(-0.6, 10.6)
    plt.yticks([0,2,4,6,8,10])
    ax.grid(linestyle='-', color='#999999', linewidth=2, alpha=0.3)


plot_rank_rank_reg(layer='C3', dataset='ImageNet')
plt.tight_layout()
plt.show()

