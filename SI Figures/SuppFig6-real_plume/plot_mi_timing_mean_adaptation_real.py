# -*- coding: utf-8 -*-
"""
Script reproducing figures in the manuscript titled

Bifurcation enhances temporal information encoding in the olfactory periphery

This script plots temporal information heatmap using Na+K+Ca model when 
the input signal from the experimental data

Copyright 2024 Nirag Kadakia, Kiri Choi
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import os, sys

sys.path.append(os.path.abspath('../../models'))
sys.path.append(os.path.abspath('../../utils'))

from paper_utils import save_fig

SAVEFIG = False

#%%

file = './data/Ca_MI_timing_real_1.pkl'
with open(file, 'rb') as f:
	data_dir = pickle.load(f)

MI_tr_on_Ca_c = data_dir['MI_tr_on']
MI_tr_off_Ca_c = data_dir['MI_tr_off']
MI_tr_dur_Ca_c = data_dir['MI_tr_dur']
MI_tr_bdur_Ca_c = data_dir['MI_tr_bdur']

mus = np.linspace(4, 15, 10)
sigmas = np.logspace(-2, 0.5, 10)

#%% Supp Fig 6C

fig, ax = plt.subplots(figsize=(3.5,3))
X, Y = np.meshgrid(sigmas, mus)
mappable = ax.pcolormesh(X, Y, MI_tr_on_Ca_c, cmap='afmhot', linewidth=0,
                         rasterized=True, vmin=0, vmax=1.25)
mappable.set_edgecolor('face')
ax.set_xscale('log')
ax.set_xlim((0.0075, 2.45))
ax.set_yticks([5, 10, 15])
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
cbar = plt.colorbar(mappable)
cbar.ax.set_yticks([0, 0.5, 1])
cbar.ax.tick_params(labelsize=15)
fig.tight_layout()
if SAVEFIG:
    save_fig('MI_Ca_tr_thres30_T_ON_3')
plt.show()

fig, ax = plt.subplots(figsize=(3.5,3))
X, Y = np.meshgrid(sigmas, mus)
mappable = ax.pcolormesh(X, Y, MI_tr_off_Ca_c, cmap='afmhot', linewidth=0,
                         rasterized=True, vmin=0, vmax=1.25)
mappable.set_edgecolor('face')
ax.set_xscale('log')
ax.set_xlim((0.0075, 2.45))
ax.set_yticks([5, 10, 15])
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
cbar = plt.colorbar(mappable)
cbar.ax.set_yticks([0, 0.5, 1])
cbar.ax.tick_params(labelsize=15)
fig.tight_layout()
if SAVEFIG:
    save_fig('MI_Ca_tr_thres30_T_OFF_3')
plt.show()

fig, ax = plt.subplots(figsize=(3.5,3))
X, Y = np.meshgrid(sigmas, mus)
mappable = ax.pcolormesh(X, Y, MI_tr_dur_Ca_c, cmap='afmhot', linewidth=0,
                         rasterized=True, vmin=0, vmax=1.5)
mappable.set_edgecolor('face')
ax.set_xscale('log')
ax.set_xlim((0.0075, 2.45))
ax.set_yticks([5, 10, 15])
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
cbar = plt.colorbar(mappable)
cbar.ax.set_yticks([0, 0.5, 1, 1.5])
cbar.ax.tick_params(labelsize=15)
fig.tight_layout()
if SAVEFIG:
    save_fig('MI_Ca_tr_thres30_dur_3')
plt.show()


