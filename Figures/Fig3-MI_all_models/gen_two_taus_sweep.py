# -*- coding: utf-8 -*-
"""
Script reproducing figures in the manuscript titled

Bifurcation enhances temporal information encoding in the olfactory periphery

This script generates MI- heatmap between tau_stim and tau_rate using Na+K model

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
odor_mu = 4.54
sigma = 0.1 #nice round number
rate_sigma = 2
smooth_T = 20
num_ticks = 10_000_000
num_rate_bins = 100
num_stim_bins = 100
cutout = 5000
tau_rates = np.logspace(0, 3, 12)
odor_taus = np.logspace(1, 4, 12)

MI_neg = [[] for i in range(len(tau_rates))]
MI_pos = [[] for i in range(len(tau_rates))]
MI_tot = [[] for i in range(len(tau_rates))]

for iR, tau_rate in enumerate(tau_rates):
    
    for odor_tau in odor_taus:
        
        print (tau_rate, odor_tau)
        
        a = NaP_K_model(dt=dt, N=num_ticks, I_tau=odor_tau, I_mu=odor_mu,
                        I_sigma=sigma, rate_sigma=rate_sigma, 
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
        
    
dictionary = {
    'tau_s': odor_taus,
    'tau_r': tau_rates,
    'total_MI':MI_tot,
    'neg_MI':MI_neg,
    'pos_MI':MI_pos
}

if SAVE:
    file = './data/two_taus_sweep.pkl'
    with open(file, 'wb') as f:
        pickle.dump(dictionary, f)
    