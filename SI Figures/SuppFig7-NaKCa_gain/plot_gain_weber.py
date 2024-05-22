# -*- coding: utf-8 -*-
"""
Script reproducing figures in the manuscript titled

Bifurcation enhances temporal information encoding in the olfactory periphery

This script plots the weber law scaling

Copyright 2024 Nirag Kadakia, Kiri Choi
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import os, sys

sys.path.append(os.path.abspath('../../models'))

from models import *
from paper_utils import save_fig

SAVEFIG = False

#%%

file = './data/NaKCa_gains_m1_sR_Km0.1.pkl'
with open(file, 'rb') as f:
	data_dir = pickle.load(f)

mu_scaled = data_dir['mu_scaled']
sigma_scaled = data_dir['sigma_scaled']

mus = np.linspace(4, 15, 10)
sigmas = np.logspace(-2, 0.5, 10)

#%% Supp Fig 7

import matplotlib

custom_cmap = matplotlib.colormaps.get_cmap("viridis").resampled(len(mu_scaled))
norm = matplotlib.colors.LogNorm(vmin=0.01, vmax=0.5)

fig, ax = plt.subplots(figsize=(4,3))
for i in range(len(mu_scaled)):
    plt.plot(mus, mu_scaled[:,i], color=custom_cmap(i), lw=3)
plt.xlim(3, 19)
plt.ylim(1, 20)
plt.xscale('log')
plt.yscale('log')
plt.xticks([10], ['$10^{1}$'], fontsize=15)
plt.yticks([1, 10], ['$10^{0}$', '$10^{1}$'], fontsize=15)
ax.tick_params(which='major', length=7)
ax.tick_params(which='minor', length=4)
plt.gca().tick_params(which='minor', labelleft=False, labelbottom=False)
plt.ylabel('Gain', fontsize=15)
plt.xlabel('$\mu$', fontsize=15)
cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=custom_cmap), 
                    ax=ax, aspect=15)
cbar.ax.tick_params(labelsize=15)
plt.tight_layout()
if SAVEFIG:
    save_fig('NaP_K_weber_1')
plt.show()


