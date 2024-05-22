# -*- coding: utf-8 -*-
"""
Script reproducing figures in the manuscript titled

Bifurcation enhances temporal information encoding in the olfactory periphery

This script plots temporal information heatmap using Na+K model when 
the input signal is different from O-U signal

Copyright 2024 Nirag Kadakia, Kiri Choi
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import os, sys

sys.path.append(os.path.abspath('../../models'))
sys.path.append(os.path.abspath('../../utils'))

import utils
from paper_utils import save_fig

SAVEFIG = False

#%%

file = './data/NaK_MI_timing_new_dOUbin_1.pkl'
with open(file, 'rb') as f:
	data_dir = pickle.load(f)

MI_r_on_NaK_c = data_dir['MI_r_on']
MI_r_off_NaK_c = data_dir['MI_r_off']
MI_r_dur_NaK_c = data_dir['MI_r_dur']
MI_r_bdur_NaK_c = data_dir['MI_r_bdur']

mus = np.linspace(4., 5, 10)
sigmas = np.logspace(-2, 0.5, 10)


#%% Supp Fig 5B

fig, ax = plt.subplots(figsize=(3.5,3))
X, Y = np.meshgrid(sigmas, mus)
mappable = ax.pcolormesh(X, Y, MI_r_on_NaK_c, cmap='afmhot', linewidth=0,
                         rasterized=True, vmin=0, vmax=1.25)
mappable.set_edgecolor('face')
ax.set_xscale('log')
ax.set_xlim((0.0075, 2.45))
ax.set_yticks([4, 4.5, 5])
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
cbar = plt.colorbar(mappable)
cbar.ax.set_yticks([0, 0.5, 1])
cbar.ax.tick_params(labelsize=15)
fig.tight_layout()
if SAVEFIG:
    save_fig('MI_timing_dist_delta_T_ON_OUbin_1')
plt.show()

fig, ax = plt.subplots(figsize=(3.5,3))
X, Y = np.meshgrid(sigmas, mus)
mappable = ax.pcolormesh(X, Y, MI_r_off_NaK_c, cmap='afmhot', linewidth=0,
                         rasterized=True, vmin=0, vmax=1.25)
mappable.set_edgecolor('face')
ax.set_xscale('log')
ax.set_xlim((0.0075, 2.45))
ax.set_yticks([4, 4.5, 5])
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
cbar = plt.colorbar(mappable)
cbar.ax.set_yticks([0, 0.5, 1])
cbar.ax.tick_params(labelsize=15)
fig.tight_layout()
if SAVEFIG:
    save_fig('MI_timing_dist_delta_T_OFF_OUbin_1')
plt.show()

fig, ax = plt.subplots(figsize=(3.5,3))
X, Y = np.meshgrid(sigmas, mus)
mappable = ax.pcolormesh(X, Y, MI_r_dur_NaK_c, cmap='afmhot', linewidth=0,
                         rasterized=True, vmin=0, vmax=1.5)
mappable.set_edgecolor('face')
ax.set_xscale('log')
ax.set_xlim((0.0075, 2.45))
ax.set_yticks([4, 4.5, 5])
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
cbar = plt.colorbar(mappable)
cbar.ax.set_yticks([0, 0.5, 1, 1.5])
cbar.ax.tick_params(labelsize=15)
fig.tight_layout()
if SAVEFIG:
    save_fig('MI_timing_dist_dur_OUbin_1')
plt.show()

fig, ax = plt.subplots(figsize=(3.5,3))
X, Y = np.meshgrid(sigmas, mus)
mappable = ax.pcolormesh(X, Y, MI_r_bdur_NaK_c, cmap='afmhot', linewidth=0,
                         rasterized=True, vmin=0, vmax=3)
mappable.set_edgecolor('face')
ax.set_xscale('log')
ax.set_xlim((0.0075, 2.45))
ax.set_yticks([4, 4.5, 5])
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
cbar = plt.colorbar(mappable)
cbar.ax.set_yticks([0, 1, 2, 3])
cbar.ax.tick_params(labelsize=15)
fig.tight_layout()
if SAVEFIG:
    save_fig('MI_timing_dist_bdur_OUbin_1')
plt.show()

#%%

file = './data/NaK_MI_timing_new_dOU_bin_partial_1.pkl'
with open(file, 'rb') as f:
	data_dir = pickle.load(f)

MI_r_on_NaK_c = data_dir['MI_r_on']
MI_r_off_NaK_c = data_dir['MI_r_off']
MI_r_dur_NaK_c = data_dir['MI_r_dur']
MI_r_bdur_NaK_c = data_dir['MI_r_bdur']

mus = np.linspace(4., 5, 10)
sigmas = np.logspace(-2, 0.5, 10)


#%% Supp Fig 5C

fig, ax = plt.subplots(figsize=(3.5,3))
X, Y = np.meshgrid(sigmas, mus)
mappable = ax.pcolormesh(X, Y, MI_r_on_NaK_c, cmap='afmhot', linewidth=0,
                         rasterized=True, vmin=0, vmax=1.25)
mappable.set_edgecolor('face')
ax.set_xscale('log')
ax.set_xlim((0.0075, 2.45))
ax.set_yticks([4, 4.5, 5])
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
cbar = plt.colorbar(mappable)
cbar.ax.set_yticks([0, 0.5, 1])
cbar.ax.tick_params(labelsize=15)
fig.tight_layout()
if SAVEFIG:
    save_fig('MI_timing_dist_delta_T_ON_OUbin_partial_1')
plt.show()

fig, ax = plt.subplots(figsize=(3.5,3))
X, Y = np.meshgrid(sigmas, mus)
mappable = ax.pcolormesh(X, Y, MI_r_off_NaK_c, cmap='afmhot', linewidth=0,
                         rasterized=True, vmin=0, vmax=1.25)
mappable.set_edgecolor('face')
ax.set_xscale('log')
ax.set_xlim((0.0075, 2.45))
ax.set_yticks([4, 4.5, 5])
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
cbar = plt.colorbar(mappable)
cbar.ax.set_yticks([0, 0.5, 1])
cbar.ax.tick_params(labelsize=15)
fig.tight_layout()
if SAVEFIG:
    save_fig('MI_timing_dist_delta_T_OFF_OUbin_partial_1')
plt.show()

fig, ax = plt.subplots(figsize=(3.5,3))
X, Y = np.meshgrid(sigmas, mus)
mappable = ax.pcolormesh(X, Y, MI_r_dur_NaK_c, cmap='afmhot', linewidth=0,
                         rasterized=True, vmin=0, vmax=1.5)
mappable.set_edgecolor('face')
ax.set_xscale('log')
ax.set_xlim((0.0075, 2.45))
ax.set_yticks([4, 4.5, 5])
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
cbar = plt.colorbar(mappable)
cbar.ax.set_yticks([0, 0.5, 1, 1.5])
cbar.ax.tick_params(labelsize=15)
fig.tight_layout()
if SAVEFIG:
    save_fig('MI_timing_dist_dur_OUbin_partial_1')
plt.show()

fig, ax = plt.subplots(figsize=(3.5,3))
X, Y = np.meshgrid(sigmas, mus)
mappable = ax.pcolormesh(X, Y, MI_r_bdur_NaK_c, cmap='afmhot', linewidth=0,
                         rasterized=True, vmin=0, vmax=3)
mappable.set_edgecolor('face')
ax.set_xscale('log')
ax.set_xlim((0.0075, 2.45))
ax.set_yticks([4, 4.5, 5])
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
cbar = plt.colorbar(mappable)
cbar.ax.set_yticks([0, 1, 2, 3])
cbar.ax.tick_params(labelsize=15)
fig.tight_layout()
if SAVEFIG:
    save_fig('MI_timing_dist_bdur_OUbin_partial_1')
plt.show()


#%%

file = './data/NaK_MI_timing_new_dGauss_tau200_2.pkl'
with open(file, 'rb') as f:
	data_dir = pickle.load(f)

MI_r_on_NaK_c = data_dir['MI_r_on']
MI_r_off_NaK_c = data_dir['MI_r_off']
MI_r_dur_NaK_c = data_dir['MI_r_dur']
MI_r_bdur_NaK_c = data_dir['MI_r_bdur']

mus = np.linspace(4., 5, 10)
sigmas = np.logspace(-2, 0.5, 10)


#%% Supp Fig 5D

fig, ax = plt.subplots(figsize=(3.5,3))
X, Y = np.meshgrid(sigmas, mus)
mappable = ax.pcolormesh(X, Y, MI_r_on_NaK_c, cmap='afmhot', linewidth=0,
                         rasterized=True, vmin=0, vmax=1.25)
mappable.set_edgecolor('face')
ax.set_xscale('log')
ax.set_xlim((0.0075, 2.45))
ax.set_yticks([4, 4.5, 5])
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
cbar = plt.colorbar(mappable)
cbar.ax.set_yticks([0, 0.5, 1])
cbar.ax.tick_params(labelsize=15)
fig.tight_layout()
if SAVEFIG:
    save_fig('MI_timing_dist_delta_T_ON_Gauss_1')
plt.show()

fig, ax = plt.subplots(figsize=(3.5,3))
X, Y = np.meshgrid(sigmas, mus)
mappable = ax.pcolormesh(X, Y, MI_r_off_NaK_c, cmap='afmhot', linewidth=0,
                         rasterized=True, vmin=0, vmax=1.25)
mappable.set_edgecolor('face')
ax.set_xscale('log')
ax.set_xlim((0.0075, 2.45))
ax.set_yticks([4, 4.5, 5])
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
cbar = plt.colorbar(mappable)
cbar.ax.set_yticks([0, 0.5, 1])
cbar.ax.tick_params(labelsize=15)
fig.tight_layout()
if SAVEFIG:
    save_fig('MI_timing_dist_delta_T_OFF_Gauss_1')
plt.show()

fig, ax = plt.subplots(figsize=(3.5,3))
X, Y = np.meshgrid(sigmas, mus)
mappable = ax.pcolormesh(X, Y, MI_r_dur_NaK_c, cmap='afmhot', linewidth=0,
                         rasterized=True, vmin=0, vmax=1.5)
mappable.set_edgecolor('face')
ax.set_xscale('log')
ax.set_xlim((0.0075, 2.45))
ax.set_yticks([4, 4.5, 5])
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
cbar = plt.colorbar(mappable)
cbar.ax.set_yticks([0, 0.5, 1, 1.5])
cbar.ax.tick_params(labelsize=15)
fig.tight_layout()
if SAVEFIG:
    save_fig('MI_timing_dist_dur_Gauss_1')
plt.show()

fig, ax = plt.subplots(figsize=(3.5,3))
X, Y = np.meshgrid(sigmas, mus)
mappable = ax.pcolormesh(X, Y, MI_r_bdur_NaK_c, cmap='afmhot', linewidth=0,
                         rasterized=True, vmin=0, vmax=3)
mappable.set_edgecolor('face')
ax.set_xscale('log')
ax.set_xlim((0.0075, 2.45))
ax.set_yticks([4, 4.5, 5])
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
cbar = plt.colorbar(mappable)
cbar.ax.set_yticks([0, 1, 2, 3])
cbar.ax.tick_params(labelsize=15)
fig.tight_layout()
if SAVEFIG:
    save_fig('MI_timing_dist_bdur_Gauss_1')
plt.show()

