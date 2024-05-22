# -*- coding: utf-8 -*-
"""
Script reproducing figures in the manuscript titled

Bifurcation enhances temporal information encoding in the olfactory periphery

This script plots schematics explaining lag

Copyright 2024 Nirag Kadakia, Kiri Choi
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import os, sys

sys.path.append(os.path.abspath('../../models'))
sys.path.append(os.path.abspath('../../utils'))

from models import *
from paper_utils import save_fig

SAVEFIG = False

#%% Fig 6A

dt1 = 0.1
odor_tau = 500
rate_sigma = 0
smooth_T = 20
num_ticks = 1_000_000
num_rate_bins = 100
num_stim_bins = 100
tau_rate = 50
cutout = 20000
cutout_end = 0

mus = [5, 7]
sigma = 0.4
lags_ORN_Ca = []
acc_ORN_Ca = []

seed = np.random.randint(1000, size=len(mus))

for im, mu in enumerate(mus):
    print(mu, sigma)
    b = ORN_model_Ca(dt=dt1, N=num_ticks, odor_tau=odor_tau, odor_mu=mu, 
                    odor_sigma=sigma, rate_sigma=rate_sigma,
                    k_rate=1./tau_rate)
    b.gen_sig_trace_OU(seed=seed[im], smooth_T=smooth_T)
    b.integrate()
    b.calc_rate()
    
    stims = b.stim[int(cutout/dt1):]
    rates = b.rate[int(cutout/dt1):]
    
    acc = scipy.signal.correlate(stims-np.mean(stims), rates-np.mean(rates))
    acc_norm = acc/(np.std(stims)*np.std(rates)*len(stims))
    acc_ORN_Ca.append(acc_norm)
    acc_lag = scipy.signal.correlation_lags(len(stims), len(rates))
    lags_ORN_Ca.append(acc_lag[np.argmax(acc_norm)])


didx = num_ticks-int(cutout/dt1)

fig = plt.figure(figsize=(2.5,2.5))
plt.plot(acc_ORN_Ca[0][didx-2000:didx+18000], lw=3, color='k')

plt.plot(acc_ORN_Ca[1][didx-7000:didx+13000], lw=3, color='tab:orange')
plt.xticks([0, 10000, 20000], ['0', '10', '20',], fontsize=13)
plt.yticks([0, 1], ['0', '1'], fontsize=13)
plt.ylabel("Cross corr. (norm)", fontsize=15)
plt.xlabel("Lag ($ms$)", fontsize=15)
plt.ylim(0, 1.25)
plt.tight_layout()
if SAVEFIG:
    save_fig('crosscorrelation', tight_layout=False)
plt.show()



