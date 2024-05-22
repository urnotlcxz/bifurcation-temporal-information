# -*- coding: utf-8 -*-
"""
Script reproducing figures in the manuscript titled

Bifurcation enhances temporal information encoding in the olfactory periphery

This script plots MI- heatmap between tau_stim and tau_rate using Na+K model

Copyright 2024 Nirag Kadakia, Kiri Choi
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os, sys

sys.path.append(os.path.abspath('../../models'))
sys.path.append(os.path.abspath('../../utils'))

from models import *
from paper_utils import save_fig

SAVEFIG = False

#%% Fig 3D

file = './data/two_taus_sweep.pkl'
with open(file, 'rb') as f:
	data_dir = pickle.load(f)
MI_neg = data_dir['neg_MI']
MI_neg = np.array(MI_neg)
MI_pos = data_dir['pos_MI']
MI_pos = np.array(MI_pos)
MI_tot = data_dir['total_MI']
MI_tot = np.array(MI_tot)
tau_s = data_dir['tau_s']
tau_r = data_dir['tau_r']

fig, ax = plt.subplots(figsize=(3.5,3))
X, Y = np.meshgrid(tau_s, tau_r)
mappable = ax.pcolormesh(X, Y, MI_neg, cmap='afmhot', linewidth=0,
                         rasterized=True, vmin=0, vmax=1.25)
mappable.set_edgecolor('face')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xticks([1e1, 1e2, 1e3, 1e4])	
ax.tick_params(axis='x', labelsize=17)
ax.tick_params(axis='y', labelsize=17)
cbar = plt.colorbar(mappable)
cbar.ax.tick_params(labelsize=17)
fig.tight_layout()
if SAVEFIG:
    save_fig('Na_K_two_taus_neg_MI_heatmap_cbar_2')
plt.show()
