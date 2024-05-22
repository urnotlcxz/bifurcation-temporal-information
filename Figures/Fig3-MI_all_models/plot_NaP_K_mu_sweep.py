# -*- coding: utf-8 -*-
"""
Script reproducing figures in the manuscript titled

Bifurcation enhances temporal information encoding in the olfactory periphery

This script plots MI- heatmap between sigma and mu using Na+K model

Copyright 2024 Nirag Kadakia, Kiri Choi
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import os, sys

sys.path.append(os.path.abspath('../../utils'))

from paper_utils import save_fig

SAVEFIG = False

#%%

file = './data/Na_K_mu_sweep_rsigma2_bn200.pkl'
with open(file, 'rb') as f:
	data_dir = pickle.load(f)
mus = data_dir['params']['mus']
sigmas = data_dir['params']['sigmas']
MI_neg = data_dir['MI_neg']
MI_pos = data_dir['MI_pos']
MI_tot = data_dir['MI_tot']

#%% Figure 3B

fig, ax = plt.subplots(figsize=(3.5,3))
X, Y = np.meshgrid(sigmas, mus)
mappable = ax.pcolormesh(X, Y, MI_neg, cmap='afmhot', linewidth=0,
                         rasterized=True, vmin=0)
mappable.set_edgecolor('face')
ax.set_xscale('log')
ax.set_xlim((0.0075, 2.45))
ax.set_yticks([4, 4.5, 5])
ax.tick_params(axis='x', labelsize=17)
ax.tick_params(axis='y', labelsize=17)
cbar = plt.colorbar(mappable)
cbar.ax.tick_params(labelsize=17)
fig.tight_layout()
if SAVEFIG:
    save_fig('Na_K_mu_sweep_2')
plt.show()


