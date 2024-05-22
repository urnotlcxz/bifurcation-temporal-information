# -*- coding: utf-8 -*-
"""
Script reproducing figures in the manuscript titled

Bifurcation enhances temporal information encoding in the olfactory periphery

This script generates MI- plot over mu using Na+K model when the input is Gaussian noise

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
rate_sigma = 2
smooth_T = 20
num_ticks = 1_000_000
num_rate_bins = 100
num_stim_bins = 100
tau_rate = 50
cutout = 5000

seed = 12345

mus = np.linspace(4., 5, 10)
sigmas = np.logspace(-2, 0.5, 10)
MI_neg = np.empty((len(mus), len(sigmas)))
MI_pos = np.empty((len(mus), len(sigmas)))
MI_tot = np.empty((len(mus), len(sigmas)))
num_spikes = np.empty((len(mus), len(sigmas)), dtype=int)

for m_i, mu in enumerate(mus):
    for s_i, sigma in enumerate(sigmas):
        print(sigma)
        stim_lims = [mu-3*sigma, mu+3*sigma]
        rate_lims = [0, 200]
        
        a = NaP_K_model(dt=dt, N=num_ticks, odor_tau=odor_tau, odor_mu=mu, 
                        odor_sigma=sigma, rate_sigma=rate_sigma, 
                        k_rate=1./tau_rate)
        a.gen_sig_trace_Gaussian(seed=seed)
        a.integrate()
        a.calc_rate()
        
        num_spikes[m_i,s_i] = int(np.sum(a.bin_arr[int(cutout/dt):]))
        
        a.calc_MI(cutout=cutout, num_stim_bins=num_stim_bins, 
                  num_rate_bins=num_rate_bins, stim_lims=stim_lims, rate_lims=rate_lims)
        MI_tot[m_i,s_i] = a.MI
        a.calc_MI(cutout=cutout, split_stim='neg', 
                  num_stim_bins=num_stim_bins, num_rate_bins=num_rate_bins,
                  stim_lims=stim_lims, rate_lims=rate_lims)
        MI_neg[m_i,s_i] = a.MI
        a.calc_MI(cutout=cutout, split_stim='pos', 
                  num_stim_bins=num_stim_bins, num_rate_bins=num_rate_bins,
                  stim_lims=stim_lims, rate_lims=rate_lims)
        MI_pos[m_i,s_i] = a.MI

data_dir = dict()
params = dict()
params_to_save = ['dt', 'odor_tau', 'sigmas', 'tau_rate', 'mus', 'smooth_T', 
                  'num_stim_bins', 'num_rate_bins', 'cutout', 'num_ticks']
for key in params_to_save:
    exec("params['%s'] = %s" % (key, key))
print(params)
data_dir['params'] = params

data_dir['MI_neg'] = MI_neg
data_dir['MI_pos'] = MI_pos
data_dir['MI_tot'] = MI_tot
data_dir['num_spikes'] = num_spikes

if SAVE:
    file = './data/Na_K_mu_sweep_rsigma2_dGauss_1.pkl'
    with open(file, 'wb') as f:
        pickle.dump(data_dir, f)
                 

