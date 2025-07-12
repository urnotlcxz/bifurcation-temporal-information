import numpy as np
import pickle
import os, sys
import matplotlib.pyplot as plt

#%% Figure 3B
sys.path.append(os.path.abspath('../../utils'))
sys.path.append(os.path.abspath('../../models'))
from paper_utils import save_fig
from models import *

SAVE = True
SAVEFIG = True
#%%

dt = 0.05
odor_tau = 500
# odor_mu = 4.54
odor_mu = 16

rate_sigma = 2
smooth_T = 20
num_ticks = 1_000_000
num_rate_bins = 100
num_stim_bins = 100
tau_rate = 50
cutout = 5000

seed = 12345

mus = np.linspace(9.5, 12.5, 10)
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
        
        a = SimplifiedLIFModel(dt=dt, N=num_ticks, odor_tau=odor_tau, odor_mu=mu, 
                        odor_sigma=sigma, rate_sigma=rate_sigma, 
                        k_rate=1./tau_rate)
        a.gen_sig_trace_OU(seed=seed, smooth_T=smooth_T)
        a.integrate()
        a.calc_rate()
        
        # num_spikes[m_i,s_i] = int(np.sum(a.bin_arr[int(cutout/dt):]))
        num_spikes[m_i, s_i] = int(np.sum(a.spikes[int(cutout/dt):]))

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
    file = './data/LIF_mu_sweep_rsigma2_bn200.pkl'
    with open(file, 'wb') as f:
        pickle.dump(data_dir, f)
# with open(file, 'rb') as f:
# 	data_dir = pickle.load(f)
 
mus = data_dir['params']['mus']
sigmas = data_dir['params']['sigmas']
MI_neg = data_dir['MI_neg']
MI_pos = data_dir['MI_pos']
MI_tot = data_dir['MI_tot']

#%% Figure 3B

fig, ax = plt.subplots(figsize=(3.5,3))
X, Y = np.meshgrid(sigmas, mus)
mappable = ax.pcolormesh(X, Y, MI_neg, cmap='afmhot', linewidth=0,
                         rasterized=True, vmin=0)
mappable.set_edgecolor('face')
ax.set_xscale('log')
# ax.set_xlim((0.0075, 2.45))
# ax.set_yticks([4, 4.5, 5])
ax.tick_params(axis='x', labelsize=17)
ax.tick_params(axis='y', labelsize=17)
cbar = plt.colorbar(mappable)
cbar.ax.tick_params(labelsize=17)
fig.tight_layout()
if SAVEFIG:
    save_fig('LIF_mu_sweep_2')
plt.show()


