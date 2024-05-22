# -*- coding: utf-8 -*-
"""
Script reproducing figures in the manuscript titled

Bifurcation enhances temporal information encoding in the olfactory periphery

This script plots MI- heatmap between sigma and tau_rate using Na+K model

Copyright 2024 Nirag Kadakia, Kiri Choi
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os, sys

sys.path.append(os.path.abspath('../../models'))
sys.path.append(os.path.abspath('../../utils'))

from models import *
from paper_utils import save_fig, gen_plot

SAVEFIG = False

#%% Fig 3A

file = './data/Na_K_tau_sweep.pkl'
with open(file, 'rb') as f:
	data_dir = pickle.load(f)
MI_neg = data_dir['MI_neg']
MI_neg = np.array(MI_neg)
MI_tot = data_dir['MI_tot']
MI_tot = np.array(MI_tot)
tau_rates = data_dir['params']['tau_rates']
sigmas = data_dir['params']['sigmas']

ir = 14
print (tau_rates[ir])
fig = gen_plot(1.2, 1.2)
plt.plot(sigmas, MI_neg[ir], color='green')
plt.plot(sigmas, MI_tot[ir], color='k')
plt.xscale('log')
plt.xticks(fontsize=7)
plt.xlim(1e-2, 1)
plt.yticks(np.arange(4), fontsize=7)
plt.ylim(0, 2.5)
if SAVEFIG:
    save_fig('Na_K_tau=%1.2f' % tau_rates[ir])
plt.show()

#%% Fig 3C

fig, ax = plt.subplots(figsize=(3.5,3))
X, Y = np.meshgrid(sigmas, tau_rates)
mappable = ax.pcolormesh(X, Y, MI_neg, cmap='afmhot', linewidth=0,
                         rasterized=True, vmin=0, vmax=1.25)
mappable.set_edgecolor('face')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_yticks([1e0, 1e1, 1e2])	
ax.set_xticks([1e-2, 1e-1, 1e0])	
plt.ylim(tau_rates[0], tau_rates[-5])
plt.xlim(1e-2, 1)
ax.tick_params(axis='x', labelsize=17)
ax.tick_params(axis='y', labelsize=17)
cbar = plt.colorbar(mappable)
cbar.ax.tick_params(labelsize=17)
fig.tight_layout()
if SAVEFIG:
    save_fig('Na_K_all_taus_neg_MI_heatmap_2')
plt.show()
