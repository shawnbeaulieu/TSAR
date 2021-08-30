import glob
import pickle

import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from matplotlib import cm
from matplotlib import colors
from scipy.stats import rankdata
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde as kde
from mpl_toolkits.axes_grid1 import make_axes_locatable

sns.set_style("darkgrid")
plt.rcParams.update({'font.size': 32})
plt.rcParams.update({'xtick.labelsize': 32})
plt.rc('axes', labelsize=30)
plt.figure(figsize=(10,10))
ax = plt.gca()

def plot_rank_rank_histo(countMatrix, c1=False):

    masked = np.ma.masked_where(np.transpose(countMatrix) == 0, np.transpose(countMatrix))
    im = plt.imshow(masked, cmap=plt.get_cmap('viridis'),  norm=mcolors.LogNorm())

    plt.ylim(-1, 102)
    plt.xlim(-1,102)

    major_ticks = [-1.6,0, 20, 40, 60, 80, 100]
    minor_yticks = np.arange(-1.6,104,1)
    minor_xticks = np.arange(-1.55,104,1)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_xticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_yticks, minor=True)
    ax.grid(which='minor', linestyle='-', color='whitesmoke', linewidth=0.5)
    ax.grid(which='major', color='whitesmoke', linewidth=0.001)

    if c1:
        plt.xticks([100,80,60,40,20,0], [3.5,2.8,2.1,1.4,0.7,0])
        plt.yticks([100,80,60,40,20,0], [3.5,2.8,2.1,1.4,0.7,0])
    else:
        plt.xticks([100,80,60,40,20,0], [5,4,3,2,1,0])
        plt.yticks([100,80,60,40,20,0], [5,4,3,2,1,0])
    
    plt.plot([-5,120],[-5,120], color='white', linewidth=1.0, zorder=2)

    ax.invert_xaxis()
    ax.invert_yaxis()

    plt.ylabel("Ranked Task-Specific Activity")
    plt.xlabel("Ranked Task-Agnostic Activity")
    plt.gca().set_aspect('equal', adjustable='box')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)


# Use the "compute_rank_rank_histo" function in domain_transfer.py to obtain
# rank-rank data before trying to execute this code.

files = glob.glob("../TSAR_rankrank_Layer=C3_Model=*_Seed=99_Dataset=cifar.npy")
countrMatrix = files[0]

sns.set_style("dark")
plot_rank_rank_histo(countMatrix)
plt.tight_layout()
plt.show()

