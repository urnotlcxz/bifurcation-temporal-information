# -*- coding: utf-8 -*-
"""
Script reproducing figures in the manuscript titled

Bifurcation enhances temporal information encoding in the olfactory periphery

This script generates data testing different binsizes 

Copyright 2024 Nirag Kadakia, Kiri Choi
"""

import numpy as np
import pickle
import os, sys

sys.path.append(os.path.abspath('../../models'))

from models import *

SAVE = False
                 
#%% Binsize test

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

mu = 4.54
sigmas = np.logspace(-2, 0.5, 10)
MI_tot = np.empty((7, 7, len(sigmas)))

for ri, rbin in enumerate([50, 75, 100, 150, 200, 500, 1000]):
    print(rbin)
    for si, sbin in enumerate([50, 75, 100, 150, 200, 500, 1000]):
        print(sbin)
        for s_i, sigma in enumerate(sigmas):
            print(sigma)
            stim_lims = [mu-3*sigma, mu+3*sigma]
            rate_lims = [0, 200]
            
            a = NaP_K_model(dt=dt, N=num_ticks, odor_tau=odor_tau, odor_mu=mu, 
                            odor_sigma=sigma, rate_sigma=rate_sigma, 
                            k_rate=1./tau_rate)
            a.gen_sig_trace_OU(seed=seed, smooth_T=smooth_T)
            a.integrate()
            a.calc_rate()
            
            a.calc_MI(cutout=cutout, num_stim_bins=int(sbin), num_rate_bins=int(rbin), 
                      stim_lims=stim_lims, rate_lims=rate_lims)
            MI_tot[ri,si,s_i] = a.MI

data_dir = dict()
params = dict()
params_to_save = ['dt', 'odor_tau', 'sigmas', 'tau_rate', 'mu', 'smooth_T', 
                  'num_stim_bins', 'num_rate_bins', 'cutout', 'num_ticks']
for key in params_to_save:
    exec("params['%s'] = %s" % (key, key))
print(params)
data_dir['params'] = params

data_dir['MI_tot'] = MI_tot

if SAVE:
    file = './data/Na_K_mu_bintest_4.pkl'
    with open(file, 'wb') as f:
        pickle.dump(data_dir, f)
        
