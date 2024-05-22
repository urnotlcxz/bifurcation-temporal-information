# -*- coding: utf-8 -*-
"""
Script reproducing figures in the manuscript titled

Bifurcation enhances temporal information encoding in the olfactory periphery

This script plots MI- plot sliced from heatmap between tau_stim and tau_rate

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


#%% Supp Fig 3

file = './data/Na_K_tau_sweep.pkl'
with open(file, 'rb') as f:
	data_dir = pickle.load(f)
MI_neg = data_dir['MI_neg']
MI_neg = np.array(MI_neg)
MI_pos = data_dir['MI_pos']
MI_pos = np.array(MI_pos)
MI_tot = data_dir['MI_tot']
MI_tot = np.array(MI_tot)
tau_rates = data_dir['params']['tau_rates']
sigmas = data_dir['params']['sigmas']

sigma_idx_to_plot = 6
print (sigmas[sigma_idx_to_plot])
fig = gen_plot(1.2, 1.2)
plt.plot(tau_rates[:-5], MI_neg[:-5, sigma_idx_to_plot], color='green')
plt.xscale('log')
plt.xticks([1, 10, 100, 1000], fontsize=7)	
plt.xlim(tau_rates[0], tau_rates[-5])
plt.yticks([0, 0.5, 1], fontsize=7)	
if SAVEFIG:
    save_fig('Na_K_neg_MI_vs_tau_sigma=%1.2f' % sigmas[sigma_idx_to_plot])
plt.show()
