# -*- coding: utf-8 -*-
"""
Script reproducing figures in the manuscript titled

Bifurcation enhances temporal information encoding in the olfactory periphery

This script plots generic signal, voltage trace, and firing rate

Copyright 2024 Nirag Kadakia, Kiri Choi
"""

import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append(os.path.abspath('../../models'))
sys.path.append(os.path.abspath('../../utils'))

from models import *
from paper_utils import save_fig, gen_plot

SAVEFIG =True

#%% Fig 2A

# mu = 4.54
mu = 16
N = 200000
dt = 0.05
a = SimplifiedLIFModel(dt=dt, N=N, odor_tau=500, odor_mu=mu, odor_sigma=0.1, 
				rate_sigma=0.01	, k_rate=1/20)
# a = NaP_K_model(dt=dt, N=N, odor_tau=500, odor_mu=mu, odor_sigma=0.1, 
# 				rate_sigma=0.01	, k_rate=1/50)
a.gen_sig_trace_OU(seed=2, smooth_T=20)
a.integrate()
a.calc_rate()  #filt_type='exp'

print("rate max:", np.max(a.rate), "min:", np.min(a.rate))
# print("spikes sum:", np.sum(a.spikes))
print("mean V:", np.mean(a.V))
print("max V:", np.max(a.V), "min V:", np.min(a.V))

fig, ax = gen_plot(1.2, 0.8)
plt.plot(a.Tt/1000, a.stim, lw=0.7, color='k')
plt.xticks(range(0, 20, 2))
plt.yticks(np.arange(0, 10, 0.25))
plt.xlim(0, N*dt/1000)
plt.ylim(min(a.stim)*0.99, max(a.stim)*1.01)
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
plt.axhline(10, color='b', ls='--')
if SAVEFIG:
    save_fig('ex_odor')
plt.show()

fig, ax = gen_plot(1.2, 0.8)
# plt.plot(a.Tt/1000, a.x[:, 0], lw=0.3, color='k')
plt.scatter(a.Tt/1000, a.spikes, lw=0.3, color='k')

plt.xticks(range(0, 20, 2))
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
plt.xlim(0, N*dt/1000)
if SAVEFIG:
    save_fig('ex_spikes')
plt.show()


fig, ax = gen_plot(1.2, 0.8)
plt.plot(a.Tt/1000, a.rate, lw=0.7, color='k')
plt.xticks(range(0, 20, 2))
plt.yticks(np.arange(0, 100, 20))
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
plt.xlim(0, N*dt/1000)
if SAVEFIG:
    save_fig('ex_rate')
plt.show()


fig, ax = gen_plot(1.2, 0.8)
plt.scatter(a.Tt/1000, a.V, lw=0.3, color='k')
plt.xticks(range(0, 20, 2))
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
plt.xlim(0, N*dt/1000)
if SAVEFIG:
    save_fig('ex_V')
plt.show()