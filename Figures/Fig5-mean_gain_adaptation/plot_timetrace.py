# -*- coding: utf-8 -*-
"""
Script reproducing figures in the manuscript titled

Bifurcation enhances temporal information encoding in the olfactory periphery

This script plots adaptation schematics using Na+K+Ca model

Copyright 2024 Nirag Kadakia, Kiri Choi
"""

import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append(os.path.abspath('../../models'))
sys.path.append(os.path.abspath('../../utils'))

from models import *
from paper_utils import save_fig, gen_plot

SAVEFIG = False

#%% Fig 5B

data = np.loadtxt('./data/NaP_K_adapting_timetrace_same_sigma.npy',)

colors2 = ['k', 'k']
fig = gen_plot(1.5, 1)
plt.plot(data[:data.shape[0]//2 + 2, 0]/1000, 
		 data[:data.shape[0]//2 + 2, 1], color=colors2[0], lw=0.7)
plt.plot(data[data.shape[0]//2:, 0]/1000, 
		 data[data.shape[0]//2:, 1], color=colors2[-1], lw=0.7)
plt.xticks(np.arange(0, 50, 10), fontsize=7)
plt.xlim(0, data[-1, 0]/1000)
plt.yticks(fontsize=7)
plt.tight_layout()
if SAVEFIG:
    save_fig('timetrace_stim_same_sigma')
plt.show()

fig = gen_plot(1.5, 1)
plt.plot(data[:data.shape[0]//2 + 2, 0]/1000, 
		 data[:data.shape[0]//2 + 2, 2], color=colors2[0], lw=0.7)
plt.plot(data[data.shape[0]//2:, 0]/1000, 
		 data[data.shape[0]//2:, 2], color=colors2[-1], lw=0.7)
plt.xticks(np.arange(0, 50, 10), fontsize=7)
plt.xlim(0, data[-1, 0]/1000)
plt.yticks([0, 250], fontsize=7)
plt.tight_layout()
if SAVEFIG:
    save_fig('timetrace_resp_same_sigma')
plt.show()

