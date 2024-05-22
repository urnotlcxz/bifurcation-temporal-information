# -*- coding: utf-8 -*-
"""
Script reproducing figures in the manuscript titled

Bifurcation enhances temporal information encoding in the olfactory periphery

This script generates MI- plot over mu using Na+K+Ca model

Copyright 2024 Nirag Kadakia, Kiri Choi
"""

import numpy as np
import pickle
import os, sys

sys.path.append(os.path.abspath('../../models'))

from models import *

SAVE = False

#%%

dt = 0.1
odor_tau = 500
rate_sigma = 1
smooth_T = 20
num_ticks = 10_000_000
num_rate_bins = 100
num_stim_bins = 100
tau_rate = 50
cutout = 20000
cutout_end = 0

mus = [4, 6, 10, 15]
sigmas = np.logspace(-2, 0.5, 12)

seed = 12345

MI_neg = np.empty((len(mus), len(sigmas)))
MI_pos = np.empty((len(mus), len(sigmas)))
MI_tot = np.empty((len(mus), len(sigmas)))

for im, mu in enumerate(mus):
    for sigma_i, sigma in enumerate(sigmas):
        print(mu, sigma)
        stim_lims = [mu-3*sigma, mu+3*sigma]
        rate_lims = [0, 200]
        
        a = ORN_model_Ca(dt=dt, N=num_ticks, odor_tau=odor_tau, odor_mu=mu, 
                        odor_sigma=sigma, rate_sigma=rate_sigma, 
                        k_rate=1./tau_rate)
        a.gen_sig_trace_OU(seed=np.random.randint(100), smooth_T=smooth_T)
        a.integrate()
        a.calc_rate()
        
        a.calc_MI(cutout=cutout, num_stim_bins=num_stim_bins, stim_lims=stim_lims, rate_lims=rate_lims,
                  num_rate_bins=num_rate_bins)
        MI_tot[im][sigma_i] = a.MI
        a.calc_MI(cutout=cutout, split_stim='neg', stim_lims=stim_lims, rate_lims=rate_lims,
                  num_stim_bins=num_stim_bins, num_rate_bins=num_rate_bins)
        MI_neg[im][sigma_i] = a.MI
        a.calc_MI(cutout=cutout, split_stim='pos', stim_lims=stim_lims, rate_lims=rate_lims,
                  num_stim_bins=num_stim_bins, num_rate_bins=num_rate_bins)
        MI_pos[im][sigma_i] = a.MI
        
data_dir = dict()
params = dict()
params_to_save = ['dt', 'odor_tau', 'sigmas', 'tau_rate', 'smooth_T', 'mus',
                  'rate_sigma', 'num_stim_bins', 'num_rate_bins', 
                  'cutout', 'num_ticks', 'rate_sigma', 'seed']

for key in params_to_save:
    exec("params['%s'] = %s" % (key, key))
print (params)
data_dir['params'] = params

data_dir['MI_neg'] = MI_neg
data_dir['MI_pos'] = MI_pos
data_dir['MI_tot'] = MI_tot

if SAVE:
    file = './data/NaKCa_MI_mu_sweep_m1_sR_Km0.1.pkl'
    with open(file, 'wb') as f:
        pickle.dump(data_dir, f)


