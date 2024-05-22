# -*- coding: utf-8 -*-
"""
Script reproducing figures in the manuscript titled

Bifurcation enhances temporal information encoding in the olfactory periphery

This script generates lag traces from cross-correlation

Copyright 2024 Nirag Kadakia, Kiri Choi
"""

import numpy as np
import scipy
import os, sys

sys.path.append(os.path.abspath('../../models'))
sys.path.append(os.path.abspath('../../utils'))

from models import *

SAVE = False

#%% ORN model

N = 10

dt1 = 0.1
odor_tau = 500
rate_sigma = 0
smooth_T = 20
num_ticks = 1_000_000
num_rate_bins = 100
num_stim_bins = 100
tau_eps = 300
tau_rate = 50
cutout = 20000

mus = np.logspace(0.65, 1.2, 10)
sigma = 0.4
lags_ORN = []
acc_ORN = []

seed = np.random.randint(1000, size=N)

for im, mu in enumerate(mus):
    print(mu, sigma)
    lags_ORN_t = []
    acc_ORN_t = []
    for i in range(N):
        a = ORN_model(dt=dt1, N=num_ticks, odor_tau=odor_tau, odor_mu=mu, 
                        odor_sigma=sigma, rate_sigma=rate_sigma,
                        k_rate=1./tau_rate)
        a.tau_eps = tau_eps
        a.gen_sig_trace_OU(seed=seed[i], smooth_T=smooth_T)
        a.integrate()
        a.calc_rate()
        
        stims = a.stim[int(cutout/dt1):]
        rates = a.rate[int(cutout/dt1):]
        
        acc = scipy.signal.correlate(stims-np.mean(stims), rates-np.mean(rates))
        acc_norm = acc/(np.std(stims)*np.std(rates)*len(stims))
        acc_ORN_t.append(acc_norm[int(cutout/dt1):])
        acc_lag = scipy.signal.correlation_lags(len(stims), len(rates))
        lags_ORN_t.append(acc_lag[np.argmax(acc)])
    
    acc_ORN.append(acc_ORN_t)
    lags_ORN.append(lags_ORN_t)
    
if SAVE:
    file = './data/ORN_lag_3.npy'
    np.save(file, lags_ORN)

#%% Na+K+Ca model

N = 10

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

mus = np.logspace(0.65, 1.2, 10)
sigma = 0.4
lags_ORN_Ca = []
acc_ORN_Ca = []

seed = np.random.randint(1000, size=N)

for im, mu in enumerate(mus):
    print(mu, sigma)
    lags_ORN_Ca_t = []
    acc_ORN_Ca_t = []
    for i in range(N):
        b = ORN_model_Ca(dt=dt1, N=num_ticks, odor_tau=odor_tau, odor_mu=mu, 
                        odor_sigma=sigma, rate_sigma=rate_sigma,
                        k_rate=1./tau_rate)
        b.gen_sig_trace_OU(seed=seed[i], smooth_T=smooth_T)
        b.integrate()
        b.calc_rate()
        
        stims = b.stim[int(cutout/dt1):]
        rates = b.rate[int(cutout/dt1):]
        
        acc = scipy.signal.correlate(stims-np.mean(stims), rates-np.mean(rates))
        acc_norm = acc/(np.std(stims)*np.std(rates)*len(stims))
        acc_ORN_Ca_t.append(acc_norm[num_ticks-int(cutout/dt1):])
        acc_lag = scipy.signal.correlation_lags(len(stims), len(rates))
        lags_ORN_Ca_t.append(acc_lag[np.argmax(acc_norm)])
    
    lags_ORN_Ca.append(lags_ORN_Ca_t)
    acc_ORN_Ca.append(acc_ORN_Ca_t)
    
if SAVE:
    file = './data/NaKCa_lag_2.npy'
    np.save(file, lags_ORN_Ca)
        
#%% Na+K model

dt2 = 0.05
odor_tau = 500
rate_sigma = 0
smooth_T = 20
num_ticks = 1_000_000
num_rate_bins = 100
num_stim_bins = 100
tau_rate = 50
cutout = 5000

mus = np.logspace(0.65, 1.2, 10)
sigma = 0.4
lags_NaK = []
acc_NaK = []

seed = np.random.randint(10000, size=10)

for im, mu in enumerate(mus):
    print(mu, sigma)
    lags_NaK_t = []
    acc_NaK_t = []
    for i in range(10):
        c = NaP_K_model(dt=dt2, N=num_ticks, odor_tau=odor_tau, odor_mu=mu, 
                        odor_sigma=sigma, rate_sigma=rate_sigma,
                        k_rate=1./tau_rate)
        c.gen_sig_trace_OU(seed=seed[i], smooth_T=smooth_T)
        c.integrate()
        c.calc_rate()
        
        stims = c.stim[int(cutout/dt2):]
        rates = c.rate[int(cutout/dt2):]
        
        acc = scipy.signal.correlate(stims-np.mean(stims), rates-np.mean(rates))
        acc_norm = acc/(np.std(stims)*np.std(rates)*len(stims))
        acc_NaK_t.append(acc_norm[int(cutout/dt2):])
        acc_lag = scipy.signal.correlation_lags(len(stims), len(rates))
        lags_NaK_t.append(acc_lag[np.argmax(acc)])
    
    acc_NaK.append(acc_NaK_t)
    lags_NaK.append(lags_NaK_t)

if SAVE:
    file = './data/NaK_lag_3.npy'
    np.save(file, lags_NaK)


