# -*- coding: utf-8 -*-
"""
Script reproducing figures in the manuscript titled

Bifurcation enhances temporal information encoding in the olfactory periphery

This script plots schematics explaining signal timing statistics

Copyright 2024 Nirag Kadakia, Kiri Choi
"""

import numpy as np
import matplotlib.pyplot as plt
import os, sys
import copy

sys.path.append(os.path.abspath('../../models'))
sys.path.append(os.path.abspath('../../utils'))

from models import *
from paper_utils import save_fig

SAVE = False
SAVEFIG = False

#%%
dt = 0.05
odor_tau = 500
rate_sigma = 0
smooth_T = 20
num_ticks = 1_000_000
num_rate_bins = 100
num_stim_bins = 100
tau_rate = 50
cutout = 5000

seed = 12345

mu = 4.54
sigma = 0.4

MI_r_on_NaK = []
MI_r_off_NaK = []
MI_r_dur_NaK = []
MI_tr_on_NaK = []
MI_tr_off_NaK = []
MI_tr_dur_NaK = []

c = NaP_K_model(dt=dt, N=num_ticks, odor_tau=odor_tau, odor_mu=mu, 
                odor_sigma=sigma, rate_sigma=rate_sigma, 
                k_rate=1./tau_rate)
c.gen_sig_trace_OU(seed=seed, smooth_T=smooth_T)


stims_NaK = c.stim[int(cutout/dt):]

bin_stim_NaK = np.zeros(len(stims_NaK))

bin_stim_NaK[np.where(stims_NaK > 4.54)[0]] = 1

t_stim_NaK = np.diff(bin_stim_NaK)

on_stim_NaK_event = copy.deepcopy(t_stim_NaK)
off_stim_NaK_event = copy.deepcopy(t_stim_NaK)

on_stim_NaK_event[on_stim_NaK_event < 0] = 0
off_stim_NaK_event[off_stim_NaK_event > 0] = 0
off_stim_NaK_event[off_stim_NaK_event < 0] = 1


#%% Fig 4A

fig = plt.figure(figsize=(3,1.5))
plt.plot(stims_NaK[200000:350000], lw=2, color='k')
plt.hlines(4.54, 0, 150000, color='tab:blue', ls='dashed', lw=2)
plt.axis('off')
if SAVEFIG:
    save_fig('MI_timing_s_raw')
plt.show()

fig = plt.figure(figsize=(3,.5))
plt.plot(on_stim_NaK_event[200000:350000], lw=2, color='tab:blue')
plt.axis('off')
if SAVEFIG:
    save_fig('MI_timing_s_ton')
plt.show()

fig = plt.figure(figsize=(3,.5))
plt.plot(off_stim_NaK_event[200000:350000], lw=2, color='tab:orange')
plt.axis('off')
if SAVEFIG:
    save_fig('MI_timing_s_toff')
plt.show()

fig = plt.figure(figsize=(3,.5))
plt.plot(bin_stim_NaK[200000:350000], lw=2, color='tab:red')
plt.axis('off')
if SAVEFIG:
    save_fig('MI_timing_s_tdur')
plt.show()