
import numpy as np
import pickle
import os, sys


#%% Fig 3D
sys.path.append(os.path.abspath('../../models'))
sys.path.append(os.path.abspath('../../utils'))
from paper_utils import save_fig, gen_plot
from models import *
import matplotlib.pyplot as plt

SAVE = True
SAVEFIG = True
#%%

dt = 0.05
odor_mu = 4.54
sigma = 0.1 #nice round number
rate_sigma = 2
smooth_T = 20
num_ticks = 10_000_000
num_rate_bins = 100
num_stim_bins = 100
cutout = 5000
tau_rates = np.logspace(0, 3, 12)
odor_taus = np.logspace(1, 4, 12)

MI_neg = [[] for i in range(len(tau_rates))]
MI_pos = [[] for i in range(len(tau_rates))]
MI_tot = [[] for i in range(len(tau_rates))]

for iR, tau_rate in enumerate(tau_rates):
    
    for odor_tau in odor_taus:
        
        print (tau_rate, odor_tau)
        
        a = SimplifiedLIFModel(dt=dt, N=num_ticks, I_tau=odor_tau, I_mu=odor_mu,
                        I_sigma=sigma, rate_sigma=rate_sigma, 
                        k_rate=1./tau_rate)
        a.gen_sig_trace_OU(seed=np.random.randint(100), smooth_T=smooth_T)
        a.integrate()
        a.calc_rate()
        
        a.calc_MI(cutout=cutout, num_stim_bins=num_stim_bins, 
                  num_rate_bins=num_rate_bins)
        MI_tot[iR].append(a.MI)
        a.calc_MI(cutout=cutout, split_stim='neg', 
                  num_stim_bins=num_stim_bins, num_rate_bins=num_rate_bins)
        MI_neg[iR].append(a.MI)
        a.calc_MI(cutout=cutout, split_stim='pos', 
                  num_stim_bins=num_stim_bins, num_rate_bins=num_rate_bins)
        MI_pos[iR].append(a.MI)
        
    
dictionary = {
    'tau_s': odor_taus,
    'tau_r': tau_rates,
    'total_MI':MI_tot,
    'neg_MI':MI_neg,
    'pos_MI':MI_pos
}

if SAVE:
    file = './data/LIF_two_taus_sweep.pkl'
    with open(file, 'wb') as f:
        pickle.dump(dictionary, f)

with open(file, 'rb') as f:
	data_dir = pickle.load(f)  
MI_neg = data_dir['neg_MI']
MI_neg = np.array(MI_neg)
MI_pos = data_dir['pos_MI']
MI_pos = np.array(MI_pos)
MI_tot = data_dir['total_MI']
MI_tot = np.array(MI_tot)
tau_s = data_dir['tau_s']
tau_r = data_dir['tau_r']

fig, ax = plt.subplots(figsize=(3.5,3))
X, Y = np.meshgrid(tau_s, tau_r)
mappable = ax.pcolormesh(X, Y, MI_neg, cmap='afmhot', linewidth=0,
                         rasterized=True, vmin=0, vmax=1.25)
mappable.set_edgecolor('face')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xticks([1e1, 1e2, 1e3, 1e4])	
ax.tick_params(axis='x', labelsize=17)
ax.tick_params(axis='y', labelsize=17)
cbar = plt.colorbar(mappable)
cbar.ax.tick_params(labelsize=17)
fig.tight_layout()
if SAVEFIG:
    save_fig('LIF_two_taus_neg_MI_heatmap_cbar_2')
plt.show()
