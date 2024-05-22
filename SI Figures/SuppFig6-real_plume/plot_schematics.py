# -*- coding: utf-8 -*-
"""
Script reproducing figures in the manuscript titled

Bifurcation enhances temporal information encoding in the olfactory periphery

This script plots distributions comparing the input signal from 
O-U signal and the experimental data

Copyright 2024 Nirag Kadakia, Kiri Choi
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import os, sys

sys.path.append(os.path.abspath('../../models'))
sys.path.append(os.path.abspath('../../utils'))

from models import *
from paper_utils import save_fig, gen_plot

# Load real signal
signal = np.load('./data/traj21_signal_1.npy')

SAVEFIG = False

thres = 2.94128358

#%% Supp Fig 6A

fig, ax = gen_plot(1.2, 0.8)
plt.plot(signal[:500], lw=0.7, color='k')
plt.xticks(range(0, 500, 50))
plt.yticks(np.arange(0, 10, 1))
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
plt.axhline(thres, color='b', ls='--')
if SAVEFIG:
    save_fig('real_odor')
plt.show()


dt = 0.05
odor_tau = 500
rate_sigma = 0
smooth_T = 20
num_ticks = 10_000_000
num_rate_bins = 100
num_stim_bins = 100
tau_rate = 50
cutout = 100

c = NaP_K_model(dt=dt, N=num_ticks, odor_tau=odor_tau, odor_mu=4.54, 
                odor_sigma=1, rate_sigma=rate_sigma, 
                k_rate=1./tau_rate)

c.gen_sig_trace_OU(seed=1111, smooth_T=smooth_T)

h1 = np.histogram(c.stim, bins=50, density=True)
h1g = scipy.ndimage.gaussian_filter(h1[0], 1)

fig, ax = gen_plot(1.2, 0.8)
plt.plot(h1g)
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
if SAVEFIG:
    save_fig('ex_odor_hist')
plt.show()

h2 = np.histogram(signal, bins=50, density=True)
h2g = scipy.ndimage.gaussian_filter(h2[0], 1)

fig, ax = gen_plot(1.2, 0.8)
plt.plot(h2g)
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
if SAVEFIG:
    save_fig('real_odor_hist')
plt.show()