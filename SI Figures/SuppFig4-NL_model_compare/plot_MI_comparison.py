# -*- coding: utf-8 -*-
"""
Script reproducing figures in the manuscript titled

Bifurcation enhances temporal information encoding in the olfactory periphery

This script analyzes LN and NL models using SNIC nonlinearity

Copyright 2024 Nirag Kadakia, Kiri Choi
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

sys.path.append(os.path.abspath('../../models'))
sys.path.append(os.path.abspath('../../utils'))

from models import *

#%%

odor_tau = 300
odor_mu = 4.54
rate_sigma = 0
smooth_T = 20
num_ticks = 1_000_000
num_rate_bins = 100
num_stim_bins = 100
tau_rate = 10

sigmas = np.logspace(-1, 0, 12)

stims = []

rate_LN_relu = []
MI_LN_relu = []
MIneg_LN_relu = []
MIpos_LN_relu = []

rate_LN_snic = []
MI_LN_snic = []
MIneg_LN_snic = []
MIpos_LN_snic = []

rate_NL_relu = []
MI_NL_relu = []
MIneg_NL_relu = []
MIpos_NL_relu = []

rate_NL_snic = []
MI_NL_snic = []
MIneg_NL_snic = []
MIpos_NL_snic = []

rate_raw = []
MI_raw = []
MIneg_raw = []
MIpos_raw = []


for sigma in sigmas:
    print("Sigma: " + str(sigma))
    
    a = LN_model(N=num_ticks, odor_tau=odor_tau, odor_mu=odor_mu, 
                 odor_sigma=sigma, rate_sigma=rate_sigma, k_rate=1./tau_rate, NF_type='relu')
    a.gen_sig_trace_OU(seed=1111)
    stims.append(a.stim)
    a.gen_firing_rate()
    a.get_opt_corr_lag(no_lag=True)
    rate_LN_relu.append(a.rate)
    a.calc_MI(num_stim_bins=num_stim_bins, num_rate_bins=num_rate_bins, rate_lims=[0, 1])
    MI_LN_relu.append(a.MI)
    a.calc_MI(split_stim='neg', num_stim_bins=num_stim_bins, num_rate_bins=num_rate_bins, rate_lims=[0, 1])
    MIneg_LN_relu.append(a.MI)
    a.calc_MI(split_stim='pos', num_stim_bins=num_stim_bins, num_rate_bins=num_rate_bins, rate_lims=[0, 1])
    MIpos_LN_relu.append(a.MI)
    

    b = LN_model(N=num_ticks, odor_tau=odor_tau, odor_mu=odor_mu, 
                 odor_sigma=sigma, rate_sigma=rate_sigma, k_rate=1./tau_rate,
                 SNIC_response='../../models/SNIC_filter.npy', NF_type='snic')
    b.gen_sig_trace_OU(seed=1111)
    b.gen_firing_rate()
    b.get_opt_corr_lag(no_lag=True)
    rate_LN_snic.append(b.rate)
    b.calc_MI(num_stim_bins=num_stim_bins, num_rate_bins=num_rate_bins, rate_lims=[0, 1])
    MI_LN_snic.append(b.MI)
    b.calc_MI(split_stim='neg', num_stim_bins=num_stim_bins, num_rate_bins=num_rate_bins, rate_lims=[0, 1])
    MIneg_LN_snic.append(b.MI)
    b.calc_MI(split_stim='pos', num_stim_bins=num_stim_bins, num_rate_bins=num_rate_bins, rate_lims=[0, 1])
    MIpos_LN_snic.append(b.MI)
    
    
    c = NL_model(N=num_ticks, odor_tau=odor_tau, odor_mu=odor_mu, 
                 odor_sigma=sigma, rate_sigma=rate_sigma, k_rate=1./tau_rate, NF_type='relu')
    c.gen_sig_trace_OU(seed=1111)
    c.gen_firing_rate()
    c.get_opt_corr_lag(no_lag=True)
    rate_NL_relu.append(c.rate)
    c.calc_MI(num_stim_bins=num_stim_bins, num_rate_bins=num_rate_bins, rate_lims=[0, 1])
    MI_NL_relu.append(c.MI)
    c.calc_MI(split_stim='neg', num_stim_bins=num_stim_bins, num_rate_bins=num_rate_bins, rate_lims=[0, 1])
    MIneg_NL_relu.append(c.MI)
    c.calc_MI(split_stim='pos', num_stim_bins=num_stim_bins, num_rate_bins=num_rate_bins, rate_lims=[0, 1])
    MIpos_NL_relu.append(c.MI)
    

    d = NL_model(N=num_ticks, odor_tau=odor_tau, odor_mu=odor_mu, 
                 odor_sigma=sigma, rate_sigma=rate_sigma, k_rate=1./tau_rate,
                 SNIC_response='../../models/SNIC_filter.npy', NF_type='snic')
    d.gen_sig_trace_OU(seed=1111)
    d.gen_firing_rate()
    d.get_opt_corr_lag(no_lag=True)
    rate_NL_snic.append(d.rate)
    d.calc_MI(num_stim_bins=num_stim_bins, num_rate_bins=num_rate_bins, rate_lims=[0, 1])
    MI_NL_snic.append(d.MI)
    d.calc_MI(split_stim='neg', num_stim_bins=num_stim_bins, num_rate_bins=num_rate_bins, rate_lims=[0, 1])
    MIneg_NL_snic.append(d.MI)
    d.calc_MI(split_stim='pos', num_stim_bins=num_stim_bins, num_rate_bins=num_rate_bins, rate_lims=[0, 1])
    MIpos_NL_snic.append(d.MI)

#%% Supp Fig 4
    
fig, ax = plt.subplots(figsize=(3,2))
plt.plot(sigmas, np.std(rate_LN_relu, axis=1)/np.std(stims, axis=1), lw=3, ls='dotted', color='tab:red')
plt.plot(sigmas, np.std(rate_LN_snic, axis=1)/np.std(stims, axis=1), lw=3, ls='dotted', color='tab:orange')
plt.xticks(fontsize=15)
plt.xscale('log')
plt.yscale('log')
ax.yaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_yticks([0.5, 1, 2])
ax.set_yticklabels([0.5, 1, 2], fontsize=15)
plt.show()

fig, ax = plt.subplots(figsize=(3,2))
plt.plot(sigmas, np.std(rate_NL_relu, axis=1)/np.std(stims, axis=1), lw=3, ls='dotted', color='tab:green')
plt.plot(sigmas, np.std(rate_NL_snic, axis=1)/np.std(stims, axis=1), lw=3, ls='dotted', color='tab:blue')
plt.xticks(fontsize=15)
plt.xscale('log')
plt.yscale('log')
ax.yaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_yticks([0.5, 1, 2])
ax.set_yticklabels([0.5, 1, 2], fontsize=15)
plt.show()


fig = plt.figure(figsize=(3,2))
plt.plot(sigmas, MI_LN_relu, color='tab:red', lw=3)
plt.plot(sigmas, MI_LN_snic, color='tab:orange', lw=3)
plt.plot(sigmas, MI_NL_relu, color='tab:green', lw=3)
plt.plot(sigmas, MI_NL_snic, color='tab:blue', lw=3)
plt.xscale('log')
plt.xticks(fontsize=15)
plt.yticks([1.5, 2], fontsize=15)
plt.show()


