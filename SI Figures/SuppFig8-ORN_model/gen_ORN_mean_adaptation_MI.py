# -*- coding: utf-8 -*-
"""
Script reproducing figures in the manuscript titled

Bifurcation enhances temporal information encoding in the olfactory periphery

This script generates MI- plot over mu using ORN model

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
tau_eps = 300
tau_rate = 50
cutout = 20000

mus = [4, 6, 10, 15]
sigmas = np.logspace(-2, 0.5, 12)
MI_neg = [[] for i in range(len(mus))]
MI_pos = [[] for i in range(len(mus))]
MI_tot = [[] for i in range(len(mus))]

for im, mu in enumerate(mus):
    for sigma in sigmas:
        print(mu, sigma)
        stim_lims = [mu-3*sigma, mu+3*sigma]
        rate_lims = [0, 200]
        
        a = ORN_model(dt=dt, N=num_ticks, odor_tau=odor_tau, odor_mu=mu, 
                        odor_sigma=sigma, rate_sigma=rate_sigma,
                        k_rate=1./tau_rate)
        a.tau_eps = tau_eps
        a.gen_sig_trace_OU(seed=np.random.randint(100), smooth_T=smooth_T)
        a.integrate()
        a.calc_rate()
        a.calc_MI(cutout=cutout, num_stim_bins=num_stim_bins, stim_lims=stim_lims, rate_lims=rate_lims,
                  num_rate_bins=num_rate_bins)
        MI_tot[im].append(a.MI)
        a.calc_MI(cutout=cutout, split_stim='neg', stim_lims=stim_lims, rate_lims=rate_lims,
                  num_stim_bins=num_stim_bins, num_rate_bins=num_rate_bins)
        MI_neg[im].append(a.MI)
        a.calc_MI(cutout=cutout, split_stim='pos', stim_lims=stim_lims, rate_lims=rate_lims,
                  num_stim_bins=num_stim_bins, num_rate_bins=num_rate_bins)
        MI_pos[im].append(a.MI)

data_dir = dict()

params = dict()
params_to_save = ['dt', 'odor_tau', 'sigmas', 'tau_rate', 'smooth_T', 'mus',
                  'tau_eps', 'rate_sigma', 'num_stim_bins', 'num_rate_bins', 
                  'cutout', 'num_ticks']

for key in params_to_save:
    exec("params['%s'] = %s" % (key, key))
print (params)
data_dir['params'] = params

data_dir['MI_neg'] = MI_neg
data_dir['MI_pos'] = MI_pos
data_dir['MI_tot'] = MI_tot

if SAVE:
    file = './data/NaP_K_adapting_MI_mu_sweep.pkl'
    with open(file, 'wb') as f:
        pickle.dump(data_dir, f)


