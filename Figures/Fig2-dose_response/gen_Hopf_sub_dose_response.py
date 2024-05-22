# -*- coding: utf-8 -*-
"""
Script reproducing figures in the manuscript titled

Bifurcation enhances temporal information encoding in the olfactory periphery

This script generates dose responses for a Hopf subcritical bifurcation model

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
sigmas = np.logspace(0, 1.7, 11)
for sigma in sigmas:
    print (sigma)
    a = Hopf_Subcritical_model(dt=0.05, N=num_ticks, I_tau=200, I_mu=97, 
							   I_sigma=sigma, k_rate=1./50, rate_sigma=0.5)
    a.gen_sig_trace_OU(seed=np.random.randint(1000), smooth_T=10)
    a.integrate()
    a.calc_rate()
    a.calc_avg_dose_response(num_stim_bins=100)
    bins.append(a.dose_response_bins)
    avgs.append(a.dose_response_avg)
	
data_dir = dict()
params = dict()
params['dt'] = 0.05
params['I_tau'] = 200
params['I_mu'] = 97
params['k_rate'] = 1./50
params['rate_sigma'] = 0.5
params['num_ticks'] = num_ticks
params['smooth_T'] = 10
params['num_bins'] = 100

data_dir['sigmas'] = sigmas
data_dir['params'] = params
data_dir['avgs'] = np.array(avgs)
data_dir['bins'] = np.array(bins)

if SAVE:
    file = './data/Hopf_sub.pkl'
    with open(file, 'wb') as f:
        pickle.dump(data_dir, f)