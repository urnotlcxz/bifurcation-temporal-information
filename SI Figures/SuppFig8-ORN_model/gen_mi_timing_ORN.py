# -*- coding: utf-8 -*-
"""
Script reproducing figures in the manuscript titled

Bifurcation enhances temporal information encoding in the olfactory periphery

This script generates temporal information heatmap using ORN model

Copyright 2024 Nirag Kadakia, Kiri Choi
"""

import numpy as np
import pickle
import os, sys
import copy

sys.path.append(os.path.abspath('../../models'))
sys.path.append(os.path.abspath('../../utils'))

from models import *
import utils

SAVE = False

#%% ORN model

dt = 0.1
odor_tau = 500
rate_sigma = 0
smooth_T = 20
num_ticks = 1_000_000
num_rate_bins = 100
num_stim_bins = 100
tau_eps = 300
tau_rate = 50
cutout = 20000

seed = 12345

mus = np.linspace(4, 15, 10)
sigmas = np.logspace(-2, 0.5, 10)

MI_tr_on_ORN = []
MI_tr_off_ORN = []
MI_tr_dur_ORN = []
MI_tr_bdur_ORN = []
bin_arr = []

for mu in mus:
    print('mu = ' + str(mu))
    MI_tr_on_ORN_t = []
    MI_tr_off_ORN_t = []
    MI_tr_dur_ORN_t = []
    MI_tr_bdur_ORN_t = []
    for sigma in sigmas:
        b = ORN_model(dt=dt, N=num_ticks, odor_tau=odor_tau, odor_mu=mu, 
                        odor_sigma=sigma, rate_sigma=rate_sigma, 
                        k_rate=1./tau_rate)
        b.tau_eps = tau_eps
        b.gen_sig_trace_OU(seed=seed, smooth_T=smooth_T)
       
        stims_ORN = b.stim
        
        bin_stim_ORN = np.zeros(len(stims_ORN))
        
        bin_stim_ORN[np.where(stims_ORN > 4.54)[0]] = 1
        
        t_stim_ORN = np.diff(bin_stim_ORN)
        
        on_stim_ORN = np.where(t_stim_ORN == 1)[0]
        off_stim_ORN = np.where(t_stim_ORN == -1)[0]
        
        b.integrate()
        b.calc_rate()
        
        rates_ORN = b.rate
        
        bin_rate_ORN = np.zeros(len(rates_ORN))
        
        bin_rate_ORN[np.where(rates_ORN > 30)[0]] = 1
        
        t_rate_ORN = np.diff(bin_rate_ORN)
        
        on_rate_ORN = np.where(t_rate_ORN == 1)[0]
        off_rate_ORN = np.where(t_rate_ORN == -1)[0]
            
        if all(bin_rate_ORN == 0) or all(bin_rate_ORN == 1) or len(on_rate_ORN) < 5 or len(off_rate_ORN) < 5:
            MI_tr_on_ORN_t.append(0)
            MI_tr_off_ORN_t.append(0)
            MI_tr_dur_ORN_t.append(0)
            MI_tr_bdur_ORN_t.append(0)
        else:
            print("sigma = " + str(sigma))
            dt_on_rate_ORN = np.diff(on_rate_ORN)*dt
            dt_off_rate_ORN = np.diff(off_rate_ORN)*dt
            if len(on_rate_ORN) > len(off_rate_ORN):
                dur_rate_ORN = off_rate_ORN-on_rate_ORN[:-1]
            elif len(on_rate_ORN) < len(off_rate_ORN):
                dur_rate_ORN = off_rate_ORN[1:]-on_rate_ORN
            else:
                if off_rate_ORN[0] < on_rate_ORN[0]:
                    dur_rate_ORN = off_rate_ORN[1:]-on_rate_ORN[:-1]
                else:
                    dur_rate_ORN = off_rate_ORN-on_rate_ORN
            if len(on_rate_ORN) > len(off_rate_ORN):
                bdur_rate_ORN = on_rate_ORN[1:]-off_rate_ORN
            elif len(on_rate_ORN) < len(off_rate_ORN):
                bdur_rate_ORN = on_rate_ORN-off_rate_ORN[:-1]
            else:
                if off_rate_ORN[0] > on_rate_ORN[0]:
                    bdur_rate_ORN = on_rate_ORN[1:]-off_rate_ORN[:-1]
                else:
                    bdur_rate_ORN = on_rate_ORN-off_rate_ORN        
            
            stims_ORN_cut = stims_ORN[int(cutout/dt):]
            rates_ORN_cut = rates_ORN[int(cutout/dt):]
            
            dt_on_rate_mesh_ORN = np.zeros(len(rates_ORN_cut))
            
            for i in range(len(on_rate_ORN)-1):
                dt_on_rate_mesh_ORN[on_rate_ORN[i]:on_rate_ORN[i]+int(dt_on_rate_ORN[i]/dt)+1] = dt_on_rate_ORN[i]
            
            dt_off_rate_mesh_ORN = np.zeros(len(rates_ORN_cut))
            
            for i in range(len(off_rate_ORN)-1):
                dt_off_rate_mesh_ORN[off_rate_ORN[i]:off_rate_ORN[i]+int(dt_off_rate_ORN[i]/dt)+1] = dt_off_rate_ORN[i]
            
            dt_dur_rate_mesh_ORN = np.zeros(len(rates_ORN_cut))
            
            for i in range(len(on_rate_ORN)-1):
                dt_dur_rate_mesh_ORN[on_rate_ORN[i]:on_rate_ORN[i]+int(dur_rate_ORN[i])+1] = dur_rate_ORN[i]
                
            dt_bdur_rate_mesh_ORN = np.zeros(len(rates_ORN_cut))
            
            for i in range(len(on_rate_ORN)-1):
                dt_bdur_rate_mesh_ORN[off_rate_ORN[i]:off_rate_ORN[i]+int(bdur_rate_ORN[i])+1] = bdur_rate_ORN[i]
                
            dt_on_rate_ORN_mean = np.mean(np.unique(dt_on_rate_mesh_ORN))
            dt_on_rate_ORN_std = np.std(np.unique(dt_on_rate_mesh_ORN))
            dt_off_rate_ORN_mean = np.mean(np.unique(dt_off_rate_mesh_ORN))
            dt_off_rate_ORN_std = np.std(np.unique(dt_off_rate_mesh_ORN))
            dt_dur_rate_ORN_mean = np.mean(np.unique(dt_dur_rate_mesh_ORN))
            dt_dur_rate_ORN_std = np.std(np.unique(dt_dur_rate_mesh_ORN))
            dt_bdur_rate_ORN_mean = np.mean(np.unique(dt_bdur_rate_mesh_ORN))
            dt_bdur_rate_ORN_std = np.std(np.unique(dt_bdur_rate_mesh_ORN))
            
            MI_tr_on_ORN_t.append(utils.calc_MI_xy(stims_ORN_cut, np.linspace(mu-3*sigma, mu+3*sigma, num_stim_bins),
                                                  dt_on_rate_mesh_ORN, np.linspace(1, dt_on_rate_ORN_mean+3*dt_on_rate_ORN_std, num_rate_bins)))
            MI_tr_off_ORN_t.append(utils.calc_MI_xy(stims_ORN_cut, np.linspace(mu-3*sigma, mu+3*sigma, num_stim_bins),
                                                   dt_off_rate_mesh_ORN, np.linspace(1, dt_off_rate_ORN_mean+3*dt_off_rate_ORN_std, num_rate_bins)))
            MI_tr_dur_ORN_t.append(utils.calc_MI_xy(stims_ORN_cut, np.linspace(mu-3*sigma, mu+3*sigma, num_stim_bins),
                                                   dt_dur_rate_mesh_ORN, np.linspace(1, dt_dur_rate_ORN_mean+3*dt_dur_rate_ORN_std, num_rate_bins)))
            MI_tr_bdur_ORN_t.append(utils.calc_MI_xy(stims_ORN_cut, np.linspace(mu-3*sigma, mu+3*sigma, num_stim_bins),
                                                    dt_bdur_rate_mesh_ORN, np.linspace(1, dt_bdur_rate_ORN_mean+3*dt_bdur_rate_ORN_std, num_rate_bins)))
                
    MI_tr_on_ORN.append(MI_tr_on_ORN_t)
    MI_tr_off_ORN.append(MI_tr_off_ORN_t)
    MI_tr_dur_ORN.append(MI_tr_dur_ORN_t)
    MI_tr_bdur_ORN.append(MI_tr_bdur_ORN_t)

MI_tr_on_ORN_c = np.array(copy.deepcopy(MI_tr_on_ORN))
MI_tr_off_ORN_c = np.array(copy.deepcopy(MI_tr_off_ORN))
MI_tr_dur_ORN_c = np.array(copy.deepcopy(MI_tr_dur_ORN))
MI_tr_bdur_ORN_c = np.array(copy.deepcopy(MI_tr_bdur_ORN))

MI_tr_on_ORN_c[MI_tr_on_ORN_c<0] = 0
MI_tr_off_ORN_c[MI_tr_off_ORN_c<0] = 0
MI_tr_dur_ORN_c[MI_tr_dur_ORN_c<0] = 0
MI_tr_bdur_ORN_c[MI_tr_bdur_ORN_c<0] = 0

data_dir = dict()
params = dict()
params_to_save = ['dt', 'odor_tau', 'sigmas', 'tau_rate', 'smooth_T', 'mus',
                  'rate_sigma', 'num_stim_bins', 'num_rate_bins', 
                  'cutout', 'num_ticks']

for key in params_to_save:
    exec("params['%s'] = %s" % (key, key))
print (params)
data_dir['params'] = params

data_dir['MI_tr_on'] = MI_tr_on_ORN_c
data_dir['MI_tr_off'] = MI_tr_off_ORN_c
data_dir['MI_tr_dur'] = MI_tr_dur_ORN_c
data_dir['MI_tr_bdur'] = MI_tr_bdur_ORN_c

if SAVE:
    file = './data/ORN_MI_tr_thres30_new_1.pkl'
    with open(file, 'wb') as f:
        pickle.dump(data_dir, f)

