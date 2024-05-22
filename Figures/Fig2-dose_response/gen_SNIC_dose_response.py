# -*- coding: utf-8 -*-
"""
Script reproducing figures in the manuscript titled

Bifurcation enhances temporal information encoding in the olfactory periphery

This script generates dose responses for a SNIC birucating model

Copyright 2024 Nirag Kadakia, Kiri Choi
"""

import sys, os
import numpy as np
import pickle

sys.path.append(os.path.abspath('../../models'))

from models import *

SAVE = False

#%%

num_sigmas = 11
num_ticks = 10_000_000

bins = []
avgs = []
sigmas = np.logspace(-1.4, 0.2, num_sigmas)
for sigma in sigmas:
    print (sigma)
    a = NaP_K_model(dt=0.05, N=num_ticks, odor_tau=500, odor_mu=4.54, odor_sigma=sigma, 
                        rate_sigma=2, k_rate=1/100)
    a.gen_sig_trace_OU(seed=np.random.randint(1000), smooth_T=20)
    a.integrate()
    a.calc_rate()
    a.calc_avg_dose_response(num_stim_bins=100)
    bins.append(a.dose_response_bins)
    avgs.append(a.dose_response_avg)
	
data_dir = dict()
params = dict()
params['dt'] = 0.05
params['I_tau'] = 500
params['I_mu'] = 4.54
params['k_rate'] = 1./100
params['rate_sigma'] = 2
params['num_ticks'] = num_ticks
params['smooth_T'] = 20
params['num_bins'] = 100

data_dir['sigmas'] = sigmas
data_dir['params'] = params
data_dir['avgs'] = np.array(avgs)
data_dir['bins'] = np.array(bins)

if SAVE:
    file = './data/SNIC.pkl'
    with open(file, 'wb') as f:
        pickle.dump(data_dir, f)