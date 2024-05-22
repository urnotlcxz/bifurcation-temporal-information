# -*- coding: utf-8 -*-
"""
Script reproducing figures in the manuscript titled

Bifurcation enhances temporal information encoding in the olfactory periphery

This script generates temporal information heatmap using Na+K+Ca model

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

#%% Na+K+Ca model

dt = 0.1
odor_tau = 500
rate_sigma = 0
smooth_T = 20
num_ticks = 1_000_000
num_rate_bins = 100
num_stim_bins = 100
tau_rate = 50
cutout = 20000

seed = 12345

mus = np.linspace(4, 15, 10)
sigmas = np.logspace(-2, 0.5, 10)

MI_tr_on_Ca = []
MI_tr_off_Ca = []
MI_tr_dur_Ca = []
MI_tr_bdur_Ca = []
bin_arr = []

for mu in mus:
    print('mu = ' + str(mu))
    MI_tr_on_Ca_t = []
    MI_tr_off_Ca_t = []
    MI_tr_dur_Ca_t = []
    MI_tr_bdur_Ca_t = []
    for sigma in sigmas:
        b = ORN_model_Ca(dt=dt, N=num_ticks, odor_tau=odor_tau, odor_mu=mu, 
                        odor_sigma=sigma, rate_sigma=rate_sigma, 
                        k_rate=1./tau_rate)
        b.gen_sig_trace_OU(seed=seed, smooth_T=smooth_T)
       
        stims_Ca = b.stim
        
        bin_stim_Ca = np.zeros(len(stims_Ca))
        
        bin_stim_Ca[np.where(stims_Ca > 4.54)[0]] = 1
        
        t_stim_Ca = np.diff(bin_stim_Ca)
        
        on_stim_Ca = np.where(t_stim_Ca == 1)[0]
        off_stim_Ca = np.where(t_stim_Ca == -1)[0]
        
        b.integrate()
        b.calc_rate()
        
        rates_Ca = b.rate
        
        bin_rate_Ca = np.zeros(len(rates_Ca))
        
        bin_rate_Ca[np.where(rates_Ca > 30)[0]] = 1
        
        t_rate_Ca = np.diff(bin_rate_Ca)
        
        on_rate_Ca = np.where(t_rate_Ca == 1)[0]
        off_rate_Ca = np.where(t_rate_Ca == -1)[0]
            
        if all(bin_rate_Ca == 0) or all(bin_rate_Ca == 1) or len(on_rate_Ca) < 5 or len(off_rate_Ca) < 5:
            MI_tr_on_Ca_t.append(0)
            MI_tr_off_Ca_t.append(0)
            MI_tr_dur_Ca_t.append(0)
            MI_tr_bdur_Ca_t.append(0)
        else:
            print("sigma = " + str(sigma))
            dt_on_rate_Ca = np.diff(on_rate_Ca)*dt
            dt_off_rate_Ca = np.diff(off_rate_Ca)*dt
            if len(on_rate_Ca) > len(off_rate_Ca):
                dur_rate_Ca = off_rate_Ca-on_rate_Ca[:-1]
            elif len(on_rate_Ca) < len(off_rate_Ca):
                dur_rate_Ca = off_rate_Ca[1:]-on_rate_Ca
            else:
                if off_rate_Ca[0] < on_rate_Ca[0]:
                    dur_rate_Ca = off_rate_Ca[1:]-on_rate_Ca[:-1]
                else:
                    dur_rate_Ca = off_rate_Ca-on_rate_Ca
            if len(on_rate_Ca) > len(off_rate_Ca):
                bdur_rate_Ca = on_rate_Ca[1:]-off_rate_Ca
            elif len(on_rate_Ca) < len(off_rate_Ca):
                bdur_rate_Ca = on_rate_Ca-off_rate_Ca[:-1]
            else:
                if off_rate_Ca[0] > on_rate_Ca[0]:
                    bdur_rate_Ca = on_rate_Ca[1:]-off_rate_Ca[:-1]
                else:
                    bdur_rate_Ca = on_rate_Ca-off_rate_Ca        
            
            stims_Ca_cut = stims_Ca[int(cutout/dt):]
            rates_Ca_cut = rates_Ca[int(cutout/dt):]
            
            dt_on_rate_mesh_Ca = np.zeros(len(rates_Ca_cut))
            
            for i in range(len(on_rate_Ca)-1):
                dt_on_rate_mesh_Ca[on_rate_Ca[i]:on_rate_Ca[i]+int(dt_on_rate_Ca[i]/dt)+1] = dt_on_rate_Ca[i]
            
            dt_off_rate_mesh_Ca = np.zeros(len(rates_Ca_cut))
            
            for i in range(len(off_rate_Ca)-1):
                dt_off_rate_mesh_Ca[off_rate_Ca[i]:off_rate_Ca[i]+int(dt_off_rate_Ca[i]/dt)+1] = dt_off_rate_Ca[i]
            
            dt_dur_rate_mesh_Ca = np.zeros(len(rates_Ca_cut))
            
            for i in range(len(on_rate_Ca)-1):
                dt_dur_rate_mesh_Ca[on_rate_Ca[i]:on_rate_Ca[i]+int(dur_rate_Ca[i])+1] = dur_rate_Ca[i]
                
            dt_bdur_rate_mesh_Ca = np.zeros(len(rates_Ca_cut))
            
            for i in range(len(on_rate_Ca)-1):
                dt_bdur_rate_mesh_Ca[off_rate_Ca[i]:off_rate_Ca[i]+int(bdur_rate_Ca[i])+1] = bdur_rate_Ca[i]
                
            dt_on_rate_Ca_mean = np.mean(np.unique(dt_on_rate_mesh_Ca))
            dt_on_rate_Ca_std = np.std(np.unique(dt_on_rate_mesh_Ca))
            dt_off_rate_Ca_mean = np.mean(np.unique(dt_off_rate_mesh_Ca))
            dt_off_rate_Ca_std = np.std(np.unique(dt_off_rate_mesh_Ca))
            dt_dur_rate_Ca_mean = np.mean(np.unique(dt_dur_rate_mesh_Ca))
            dt_dur_rate_Ca_std = np.std(np.unique(dt_dur_rate_mesh_Ca))
            dt_bdur_rate_Ca_mean = np.mean(np.unique(dt_bdur_rate_mesh_Ca))
            dt_bdur_rate_Ca_std = np.std(np.unique(dt_bdur_rate_mesh_Ca))
            
            MI_tr_on_Ca_t.append(utils.calc_MI_xy(stims_Ca_cut, np.linspace(mu-3*sigma, mu+3*sigma, num_stim_bins),
                                                  dt_on_rate_mesh_Ca, np.linspace(1, dt_on_rate_Ca_mean+3*dt_on_rate_Ca_std, num_rate_bins)))
            MI_tr_off_Ca_t.append(utils.calc_MI_xy(stims_Ca_cut, np.linspace(mu-3*sigma, mu+3*sigma, num_stim_bins),
                                                   dt_off_rate_mesh_Ca, np.linspace(1, dt_off_rate_Ca_mean+3*dt_off_rate_Ca_std, num_rate_bins)))
            MI_tr_dur_Ca_t.append(utils.calc_MI_xy(stims_Ca_cut, np.linspace(mu-3*sigma, mu+3*sigma, num_stim_bins),
                                                   dt_dur_rate_mesh_Ca, np.linspace(1, dt_dur_rate_Ca_mean+3*dt_dur_rate_Ca_std, num_rate_bins)))
            MI_tr_bdur_Ca_t.append(utils.calc_MI_xy(stims_Ca_cut, np.linspace(mu-3*sigma, mu+3*sigma, num_stim_bins),
                                                    dt_bdur_rate_mesh_Ca, np.linspace(1, dt_bdur_rate_Ca_mean+3*dt_bdur_rate_Ca_std, num_rate_bins)))
                
    MI_tr_on_Ca.append(MI_tr_on_Ca_t)
    MI_tr_off_Ca.append(MI_tr_off_Ca_t)
    MI_tr_dur_Ca.append(MI_tr_dur_Ca_t)
    MI_tr_bdur_Ca.append(MI_tr_bdur_Ca_t)

MI_tr_on_Ca_c = np.array(copy.deepcopy(MI_tr_on_Ca))
MI_tr_off_Ca_c = np.array(copy.deepcopy(MI_tr_off_Ca))
MI_tr_dur_Ca_c = np.array(copy.deepcopy(MI_tr_dur_Ca))
MI_tr_bdur_Ca_c = np.array(copy.deepcopy(MI_tr_bdur_Ca))

MI_tr_on_Ca_c[MI_tr_on_Ca_c<0] = 0
MI_tr_off_Ca_c[MI_tr_off_Ca_c<0] = 0
MI_tr_dur_Ca_c[MI_tr_dur_Ca_c<0] = 0
MI_tr_bdur_Ca_c[MI_tr_bdur_Ca_c<0] = 0

data_dir = dict()
params = dict()
params_to_save = ['dt', 'odor_tau', 'sigmas', 'tau_rate', 'smooth_T', 'mus',
                  'rate_sigma', 'num_stim_bins', 'num_rate_bins', 
                  'cutout', 'num_ticks']

for key in params_to_save:
    exec("params['%s'] = %s" % (key, key))
print (params)
data_dir['params'] = params

data_dir['MI_tr_on'] = MI_tr_on_Ca_c
data_dir['MI_tr_off'] = MI_tr_off_Ca_c
data_dir['MI_tr_dur'] = MI_tr_dur_Ca_c
data_dir['MI_tr_bdur'] = MI_tr_bdur_Ca_c

if SAVE:
    file = './data/Ca_MI_tr_thres30_new_1.pkl'
    with open(file, 'wb') as f:
        pickle.dump(data_dir, f)
