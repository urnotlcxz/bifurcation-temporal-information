# -*- coding: utf-8 -*-
"""
Script reproducing figures in the manuscript titled

Bifurcation enhances temporal information encoding in the olfactory periphery

This script plots schematics explainin input signals different from O-U signal

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


#%% Supp Fig 5A

mu = 4.54
N = 200000
dt = 0.05
a = NaP_K_model(dt=dt, N=N, odor_tau=500, odor_mu=mu, odor_sigma=0.1, 
				rate_sigma=0.01	, k_rate=1/50)
a.gen_sig_trace_OU_binary(seed=2, smooth_T=20)
a.integrate()
a.calc_rate()

fig, ax = gen_plot(1.2, 0.8)
plt.plot(a.Tt/1000, a.stim, lw=0.7, color='k')
plt.xticks(range(0, 20, 2))
plt.yticks(np.arange(0, 10, 0.25))
plt.xlim(0, N*dt/1000)
plt.ylim(min(a.stim)*0.99, max(a.stim)*1.01)
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
plt.axhline(4.54, color='b', ls='--')
if SAVEFIG:
    save_fig('ex_odor_OU_bin')
plt.show()

mu = 4.54
N = 200000
dt = 0.05
a = NaP_K_model(dt=dt, N=N, odor_tau=500, odor_mu=mu, odor_sigma=0.1, 
				rate_sigma=0.01	, k_rate=1/50)
a.gen_sig_trace_OU_binary(seed=2, smooth_T=20, partial='up')
a.integrate()
a.calc_rate()

fig, ax = gen_plot(1.2, 0.8)
plt.plot(a.Tt/1000, a.stim, lw=0.7, color='k')
plt.xticks(range(0, 20, 2))
plt.yticks(np.arange(0, 10, 0.25))
plt.xlim(0, N*dt/1000)
plt.ylim(min(a.stim)*0.99, max(a.stim)*1.01)
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
plt.axhline(4.54, color='b', ls='--')
if SAVEFIG:
    save_fig('ex_odor_OU_bin_p')
plt.show()

mu = 4.54
N = 200000
dt = 0.05
a = NaP_K_model(dt=dt, N=N, odor_tau=500, odor_mu=mu, odor_sigma=0.1, 
				rate_sigma=0.01	, k_rate=1/50)
a.gen_sig_trace_Gaussian(seed=2)
a.integrate()
a.calc_rate()

fig, ax = gen_plot(1.2, 0.8)
plt.plot(a.Tt/1000, a.stim, lw=0.7, color='k')
plt.xticks(range(0, 20, 2))
plt.yticks(np.arange(0, 10, 0.25))
plt.xlim(0, N*dt/1000)
plt.ylim(min(a.stim)*0.99, max(a.stim)*1.01)
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
plt.axhline(4.54, color='b', ls='--')
if SAVEFIG:
    save_fig('ex_odor_gauss')
plt.show()


