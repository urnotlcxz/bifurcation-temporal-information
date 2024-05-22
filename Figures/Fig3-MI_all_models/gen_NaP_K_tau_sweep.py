# -*- coding: utf-8 -*-
"""
Script reproducing figures in the manuscript titled

Bifurcation enhances temporal information encoding in the olfactory periphery

This script generates MI- heatmap between sigma and tau_rate using Na+K model

Copyright 2024 Nirag Kadakia, Kiri Choi
"""

import numpy as np
import pickle
import os, sys

sys.path.append(os.path.abspath('../../models'))

from models import *

SAVE = False
#%%

dt = 0.05
odor_tau = 500
odor_mu = 4.54
rate_sigma = 0
smooth_T = 20
num_ticks = 1_000_000
num_rate_bins = 100
num_stim_bins = 100
cutout = 5000
tau_rates = np.logspace(0, 3, 25)
sigmas = np.logspace(-2.5, 0.5, 12)

MI_neg = [[] for i in range(len(tau_rates))]
MI_pos = [[] for i in range(len(tau_rates))]
MI_tot = [[] for i in range(len(tau_rates))]

for iR, tau_rate in enumerate(tau_rates):
    for sigma in sigmas:
        
        print (tau_rate, sigma)
        
        a = NaP_K_model(dt=dt, N=num_ticks, odor_tau=odor_tau, odor_mu=odor_mu,
                        odor_sigma=sigma, rate_sigma=rate_sigma, 
                        k_rate=1./tau_rate)
        a.gen_sig_trace_OU(seed=np.random.randint(100), smooth_T=smooth_T)
        a.integrate()
        a.calc_rate()
        
        a.calc_MI(cutout=cutout, num_stim_bins=num_stim_bins, 
                  num_rate_bins=num_rate_bins)
        MI_tot[iR].append(a.MI)
        a.calc_MI(cutout=cutout, split_stim='neg', 
                  num_stim_bins=num_stim_bins, num_rate_bins=num_rate_bins)
        MI_neg[iR].append(a.MI)
        a.calc_MI(cutout=cutout, split_stim='pos', 
                  num_stim_bins=num_stim_bins, num_rate_bins=num_rate_bins)
        MI_pos[iR].append(a.MI)
        
data_dir = dict()

params = dict()
params_to_save = ['dt', 'odor_tau', 'sigmas', 'tau_rates', 'smooth_T', 
                  'num_stim_bins', 'num_rate_bins', 'cutout', 'num_ticks']
for key in params_to_save:
    exec("params['%s'] = %s" % (key, key))
print (params)
data_dir['params'] = params

data_dir['MI_neg'] = MI_neg
data_dir['MI_pos'] = MI_pos
data_dir['MI_tot'] = MI_tot

if SAVE:
    file = './data/Na_K_tau_sweep.pkl'
    with open(file, 'wb') as f:
        pickle.dump(data_dir, f)
