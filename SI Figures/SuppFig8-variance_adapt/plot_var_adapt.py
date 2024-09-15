# -*- coding: utf-8 -*-
"""
Script reproducing figures in the manuscript titled

Bifurcation enhances temporal information encoding in the olfactory periphery

This script tests the response of biphysical model to change in signal variance

Copyright 2024 Kiri Choi
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import os, sys

sys.path.append(os.path.abspath('../../models'))
sys.path.append(os.path.abspath('../../utils'))

from models import *
from paper_utils import save_fig, gen_plot

SAVE = False
SAVEFIG = False

#%%

dt = 0.1
odor_tau = 500
rate_sigma = 0
smooth_T = 20
num_ticks = 1_000_000
num_rate_bins = 100
num_stim_bins = 100
tau_rate = 50
mu = 4.54

seed = 12345

low_sigma = ORN_model_Ca(dt=dt, N=num_ticks, odor_tau=odor_tau, odor_mu=mu, 
                odor_sigma=0.08, rate_sigma=rate_sigma, 
                k_rate=1./tau_rate)
low_sigma.gen_sig_trace_OU(seed=seed, smooth_T=smooth_T)
low_sigma.integrate()
low_sigma.calc_rate()

high_sigma = ORN_model_Ca(dt=dt, N=num_ticks, odor_tau=odor_tau, odor_mu=mu, 
                odor_sigma=0.24, rate_sigma=rate_sigma, 
                k_rate=1./tau_rate)
high_sigma.gen_sig_trace_OU(seed=seed, smooth_T=smooth_T)
high_sigma.integrate()
high_sigma.calc_rate()

ls_hist, ls_be = np.histogram(low_sigma.rate[int(num_ticks/2):], bins=100, density=True, range=(0, 80))
hs_hist, hs_be = np.histogram(high_sigma.rate[int(num_ticks/2):], bins=100, density=True, range=(0, 80))

ls_stim_hist, ls_stim_be = np.histogram(low_sigma.stim[int(num_ticks/2):], bins=100, density=True, range=(3.75, 5.25))
hs_stim_hist, hs_stim_be = np.histogram(high_sigma.stim[int(num_ticks/2):], bins=100, density=True, range=(3.75, 5.25))

#%%

from scipy.ndimage import gaussian_filter1d

ls_hist_smooth = gaussian_filter1d(ls_hist, 2)
hs_hist_smooth = gaussian_filter1d(hs_hist, 3)

ls_stim_hist_smooth = gaussian_filter1d(ls_stim_hist, 2)
hs_stim_hist_smooth = gaussian_filter1d(hs_stim_hist, 2)

#%%

fig, ax = plt.subplots(figsize=(8,2))
ax.plot(np.linspace(0, int(num_ticks*dt)/1000, num_ticks), high_sigma.stim, color='tab:red', lw=1)
ax.plot(np.linspace(int(num_ticks*dt)/1000, 2*int(num_ticks*dt)/1000, num_ticks), low_sigma.stim, color='tab:blue', lw=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xticks(fontsize=15)
plt.yticks([4, 4.5, 5], fontsize=15)
plt.ylim(3.6, 5.4)
plt.xlabel("Time (s)", fontsize=15)
plt.ylabel("Stimulus (AU)", fontsize=15)
if SAVEFIG:
    save_fig('stim_var_timetrace')
plt.show()

ls_stim_bc = (ls_stim_be[:-1] + ls_stim_be[1:]) / 2
hs_stim_bc = (hs_stim_be[:-1] + hs_stim_be[1:]) / 2

fig, ax = plt.subplots(figsize=(4,3))
ax.plot(ls_stim_bc, ls_stim_hist_smooth*np.diff(ls_stim_be)[0], color='tab:blue', lw=3)
ax.plot(hs_stim_bc, hs_stim_hist_smooth*np.diff(hs_stim_be)[0], color='tab:red', lw=3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xticks([4, 4.5, 5], fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Stimulus (AU)", fontsize=15)
plt.ylabel("Probability", fontsize=15)
if SAVEFIG:
    save_fig('stim_var_dist')
plt.show()

ls_bc = (ls_be[:-1] + ls_be[1:]) / 2
hs_bc = (hs_be[:-1] + hs_be[1:]) / 2

fig, ax = plt.subplots(figsize=(4,3))
ax.plot(ls_bc, ls_hist_smooth*np.diff(ls_be)[0], color='tab:blue', lw=3)
ax.plot(hs_bc, hs_hist_smooth*np.diff(hs_be)[0], color='tab:red', lw=3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xticks(fontsize=15)
plt.yticks([0, 0.05], fontsize=15)
plt.xlabel("Firing Rate (Hz)", fontsize=15)
plt.ylabel("Probability", fontsize=15)
if SAVEFIG:
    save_fig('stim_var_rate_dist')
plt.show()


