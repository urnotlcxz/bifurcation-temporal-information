# -*- coding: utf-8 -*-
"""
Script reproducing figures in the manuscript titled

Bifurcation enhances temporal information encoding in the olfactory periphery

This script plots dose responses curves (unscaled, scaled)

Copyright 2024 Nirag Kadakia, Kiri Choi
"""

import sys, os
import numpy as np
import pickle
import matplotlib.pyplot as plt



sys.path.append(os.path.abspath('../../models'))
sys.path.append(os.path.abspath('../../utils'))

from models import *
from paper_utils import gen_plot, save_fig

SAVEFIG = True

#%% SNIC - Fig 2B,C

file = './data/SNIC.pkl'
with open(file, 'rb') as f:
	data_dir = pickle.load(f)
avgs = data_dir['avgs']
bins = data_dir['bins']
sigmas = data_dir['sigmas']

colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(sigmas)))
fig = gen_plot(1.3, 1.1)
for iS, sigma in enumerate(sigmas):
	if iS < 2: 
		continue
	shift = 4.54
	plt.plot((bins[iS] - shift), avgs[iS], 
			 color=colors[iS], label='%.2f' % sigma, lw=0.7)
plt.xticks(np.arange(-2, 2, 1), fontsize=7)
plt.yticks(np.arange(0, 150, 20), fontsize=7)
plt.xlim(-1.5, 0.6)
plt.ylim(0, 70)
plt.axvline(0, color='k', lw=1, ls='--')
plt.tight_layout()
if SAVEFIG:
    save_fig('SNIC_unscaled_lo_full')
plt.show()

colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(sigmas)))
fig = gen_plot(1.3, 1.1)
for iS, sigma in enumerate(sigmas):
	if iS < 2: 
		continue
	shift = 4.54
	plt.plot((bins[iS] - shift)/sigma, avgs[iS], 
			 color=colors[iS], label='%.2f' % sigma, lw=0.7)
plt.xticks(np.arange(-2, 2, 1), fontsize=7)
plt.yticks(np.arange(0, 50, 10), fontsize=7)
plt.xlim(-2, 0.2)
plt.ylim(0, 38)
plt.axvline(0, color='k', lw=1, ls='--')
plt.tight_layout()
if SAVEFIG:
    save_fig('SNIC_scaled_lo')
plt.show()

#%% Hopf subcritical - Fig 2D,E

file = './data/Hopf_sub.pkl'
with open(file, 'rb') as f:
	data_dir = pickle.load(f)
avgs = data_dir['avgs']
bins = data_dir['bins']
sigmas = data_dir['sigmas']

colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(sigmas)))
fig = gen_plot(1.3, 1.1)
for iS, sigma in enumerate(sigmas):
	if iS < 4: 
		continue
	shift = 103.5
	plt.plot((bins[iS] - shift), avgs[iS], 
			 color=colors[iS], label='%.2f' % sigma, lw=0.7)
plt.xticks(np.arange(-100, 100, 50), fontsize=7)
plt.yticks(np.arange(0, 50, 5), fontsize=7)
plt.xlim(-100.0, 70)
plt.ylim(0, 17)
plt.axvline(0, color='k', lw=1, ls='--')
plt.tight_layout()
if SAVEFIG:
    save_fig('Hopf_sub_unscaled_lo_full')
plt.show()

colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(sigmas)))
fig = gen_plot(1.3, 1.1)
for iS, sigma in enumerate(sigmas):
	if iS < 4: 
		continue
	shift = 103.5
	plt.plot((bins[iS] - shift)/sigma, avgs[iS], 
			 color=colors[iS], label='%.2f' % sigma, lw=0.7)
plt.xticks(np.arange(-3, 2, 1), fontsize=7)
plt.yticks(np.arange(0, 50, 5), fontsize=7)
plt.xlim(-3, 0.5)
plt.ylim(0, 12)
plt.axvline(0, color='k', lw=1, ls='--')
plt.tight_layout()
if SAVEFIG:
    save_fig('Hopf_sub_scaled_lo')
plt.show()
