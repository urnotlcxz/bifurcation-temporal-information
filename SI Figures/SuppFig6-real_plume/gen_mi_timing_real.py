# -*- coding: utf-8 -*-
"""
Script reproducing figures in the manuscript titled

Bifurcation enhances temporal information encoding in the olfactory periphery

This script generates temporal information heatmap using Na+K model when 
the input signal from the experimental data

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

# Load real signal
signal = np.load('./data/traj21_signal_1.npy')

SAVE = False

#%% 

dt = 0.05
odor_tau = 500
rate_sigma = 0
smooth_T = 20
num_ticks = 1_000_000
num_rate_bins = 100
num_stim_bins = 100
tau_rate = 50
cutout = 100

seed = 12345

mus = np.linspace(4., 5, 10)
sigmas = np.logspace(-2, 0.5, 10)

MI_r_on_NaK = []
MI_r_off_NaK = []
MI_r_dur_NaK = []
MI_r_bdur_NaK = []
MI_r_dur_NaK1 = []
MI_r_bdur_NaK1 = []

for mu in mus:
    print('mu = ' + str(mu))
    MI_r_on_NaK_t = []
    MI_r_off_NaK_t = []
    MI_r_dur_NaK_t = []
    MI_r_bdur_NaK_t = []
    for sigma in sigmas:
        c = NaP_K_model(dt=dt, N=num_ticks, odor_tau=odor_tau, odor_mu=mu, 
                        odor_sigma=sigma, rate_sigma=rate_sigma, 
                        k_rate=1./tau_rate)
        signal_intp = np.interp(np.linspace(0, len(signal), num_ticks), np.arange(len(signal)), signal)
        
        testsignal = signal_intp - np.mean(signal_intp)
        testsignal = testsignal/np.std(testsignal)*sigma
        testsignal = testsignal + mu
        
        c.stim = testsignal
        
        bin_stim_NaK = np.zeros(len(testsignal))
        
        bin_stim_NaK[np.where(testsignal > 4.54)[0]] = 1
        
        t_stim_NaK = np.diff(bin_stim_NaK)
        
        on_stim_NaK = np.where(t_stim_NaK == 1)[0]
        off_stim_NaK = np.where(t_stim_NaK == -1)[0]
        
        if all(bin_stim_NaK == 0) or all(bin_stim_NaK == 1) or len(on_stim_NaK) < 10 or len(off_stim_NaK) < 10:
            MI_r_on_NaK_t.append(0)
            MI_r_off_NaK_t.append(0)
            MI_r_dur_NaK_t.append(0)
            MI_r_bdur_NaK_t.append(0)
        else:
            print("sigma = " + str(sigma))
            c.integrate()
            c.calc_rate()
            
            rates_NaK = c.rate
            
            bin_rate_NaK = np.zeros(len(rates_NaK))
            
            bin_rate_NaK[np.where(rates_NaK > 0 )[0]] = 1
            
            t_rate_NaK = np.diff(bin_rate_NaK)
            
            on_rate_NaK = np.where(t_rate_NaK == 1)[0]
            off_rate_NaK = np.where(t_rate_NaK == -1)[0]
            
            dt_on_stim_NaK = np.diff(on_stim_NaK)*dt
            dt_off_stim_NaK = np.diff(off_stim_NaK)*dt
            if len(on_stim_NaK) > len(off_stim_NaK):
                dur_stim_NaK = off_stim_NaK-on_stim_NaK[:-1]
            elif len(on_stim_NaK) < len(off_stim_NaK):
                dur_stim_NaK = off_stim_NaK[1:]-on_stim_NaK
            else:
                if off_stim_NaK[0] < on_stim_NaK[0]:
                    dur_stim_NaK = off_stim_NaK[1:]-on_stim_NaK[:-1]
                else:
                    dur_stim_NaK = off_stim_NaK-on_stim_NaK
            if len(on_stim_NaK) > len(off_stim_NaK):
                bdur_stim_NaK = on_stim_NaK[1:]-off_stim_NaK
            elif len(on_stim_NaK) < len(off_stim_NaK):
                bdur_stim_NaK = on_stim_NaK-off_stim_NaK[:-1]
            else:
                if off_stim_NaK[0] > on_stim_NaK[0]:
                    bdur_stim_NaK = on_stim_NaK[1:]-off_stim_NaK[:-1]
                else:
                    bdur_stim_NaK = on_stim_NaK-off_stim_NaK
            
            dt_on_stim_mesh_NaK = np.zeros(len(testsignal))
            
            for i in range(len(on_stim_NaK)-1):
                dt_on_stim_mesh_NaK[on_stim_NaK[i]:on_stim_NaK[i]+int(dt_on_stim_NaK[i]/dt)+1] = dt_on_stim_NaK[i]
            
            dt_off_stim_mesh_NaK = np.zeros(len(testsignal))
            
            for i in range(len(off_stim_NaK)-1):
                dt_off_stim_mesh_NaK[off_stim_NaK[i]:off_stim_NaK[i]+int(dt_off_stim_NaK[i]/dt)+1] = dt_off_stim_NaK[i]
            
            dt_dur_stim_mesh_NaK = np.zeros(len(testsignal))
            
            for i in range(len(on_stim_NaK)-1):
                dt_dur_stim_mesh_NaK[on_stim_NaK[i]:on_stim_NaK[i]+int(dur_stim_NaK[i])+1] = dur_stim_NaK[i]
            
            dt_bdur_stim_mesh_NaK = np.zeros(len(testsignal))
            
            for i in range(len(on_stim_NaK)-1):
                dt_bdur_stim_mesh_NaK[off_stim_NaK[i]:off_stim_NaK[i]+int(bdur_stim_NaK[i])+1] = bdur_stim_NaK[i]
            
            stims_NaK_cut = testsignal[int(cutout/dt):]
            rates_NaK_cut = rates_NaK[int(cutout/dt):]
            dt_on_stim_mesh_NaK_cut = dt_on_stim_mesh_NaK[int(cutout/dt):]
            dt_off_stim_mesh_NaK_cut = dt_off_stim_mesh_NaK[int(cutout/dt):]
            dt_dur_stim_mesh_NaK_cut = dt_dur_stim_mesh_NaK[int(cutout/dt):]
            dt_bdur_stim_mesh_NaK_cut = dt_bdur_stim_mesh_NaK[int(cutout/dt):]
            
            dt_on_stim_NaK_mean = np.mean(np.unique(dt_on_stim_mesh_NaK_cut))
            dt_on_stim_NaK_std = np.std(np.unique(dt_on_stim_mesh_NaK_cut))
            dt_off_stim_NaK_mean = np.mean(np.unique(dt_off_stim_mesh_NaK_cut))
            dt_off_stim_NaK_std = np.std(np.unique(dt_off_stim_mesh_NaK_cut))
            dt_dur_stim_NaK_mean = np.mean(np.unique(dt_dur_stim_mesh_NaK_cut))
            dt_dur_stim_NaK_std = np.std(np.unique(dt_dur_stim_mesh_NaK_cut))
            dt_bdur_stim_NaK_mean = np.mean(np.unique(dt_bdur_stim_mesh_NaK_cut))
            dt_bdur_stim_NaK_std = np.std(np.unique(dt_bdur_stim_mesh_NaK_cut))
            
            rates_NaK_cut_mean = np.mean(rates_NaK_cut)
            rates_NaK_cut_std = np.std(rates_NaK_cut)
            
            # y rate
            MI_r_on_NaK_t.append(utils.calc_MI_xy(dt_on_stim_mesh_NaK_cut, np.linspace(1, dt_on_stim_NaK_mean+3*dt_on_stim_NaK_std, num_stim_bins),
                                                  rates_NaK_cut, np.linspace(0, 200, num_rate_bins)))
            MI_r_off_NaK_t.append(utils.calc_MI_xy(dt_off_stim_mesh_NaK_cut, np.linspace(1, dt_off_stim_NaK_mean+3*dt_off_stim_NaK_std, num_stim_bins),
                                                    rates_NaK_cut, np.linspace(0, 200, num_rate_bins)))
            MI_r_dur_NaK_t.append(utils.calc_MI_xy(dt_dur_stim_mesh_NaK_cut, np.linspace(1, (dt_dur_stim_NaK_mean+3*dt_dur_stim_NaK_std), num_stim_bins),
                                                    rates_NaK_cut, np.linspace(0, 200, num_rate_bins)))
            MI_r_bdur_NaK_t.append(utils.calc_MI_xy(dt_bdur_stim_mesh_NaK_cut, np.linspace(1, dt_bdur_stim_NaK_mean+3*dt_bdur_stim_NaK_std, num_stim_bins),
                                                    rates_NaK_cut, np.linspace(0, 200, num_rate_bins)))
            
    MI_r_on_NaK.append(MI_r_on_NaK_t)
    MI_r_off_NaK.append(MI_r_off_NaK_t)
    MI_r_dur_NaK.append(MI_r_dur_NaK_t)
    MI_r_bdur_NaK.append(MI_r_bdur_NaK_t)

MI_r_on_NaK_c = np.array(copy.deepcopy(MI_r_on_NaK))
MI_r_off_NaK_c = np.array(copy.deepcopy(MI_r_off_NaK))
MI_r_dur_NaK_c = np.array(copy.deepcopy(MI_r_dur_NaK))
MI_r_bdur_NaK_c = np.array(copy.deepcopy(MI_r_bdur_NaK))

MI_r_on_NaK_c[MI_r_on_NaK_c<0] = 0
MI_r_off_NaK_c[MI_r_off_NaK_c<0] = 0
MI_r_dur_NaK_c[MI_r_dur_NaK_c<0] = 0
MI_r_bdur_NaK_c[MI_r_bdur_NaK_c<0] = 0

data_dir = dict()
params = dict()
params_to_save = ['dt', 'odor_tau', 'sigmas', 'tau_rate', 'smooth_T', 'mus',
                  'rate_sigma', 'num_stim_bins', 'num_rate_bins', 
                  'cutout', 'num_ticks']

for key in params_to_save:
    exec("params['%s'] = %s" % (key, key))
print (params)
data_dir['params'] = params

data_dir['MI_r_on'] = MI_r_on_NaK_c
data_dir['MI_r_off'] = MI_r_off_NaK_c
data_dir['MI_r_dur'] = MI_r_dur_NaK_c
data_dir['MI_r_bdur'] = MI_r_bdur_NaK_c

if SAVE:
    file = './data/NaK_MI_timing_real_2.pkl'
    with open(file, 'wb') as f:
        pickle.dump(data_dir, f)
        
