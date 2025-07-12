import numpy as np
import pickle
import os, sys
import matplotlib.pyplot as plt

# Fig 3C
sys.path.append(os.path.abspath('../../models'))
sys.path.append(os.path.abspath('../../utils'))
from models import *
from paper_utils import save_fig, gen_plot

SAVE = True
SAVEFIG = True
#%%

dt = 0.05
odor_tau = 500
odor_mu = 4.54
rate_sigma = 0
smooth_T = 20
num_ticks = 1_000_000
num_rate_bins = 100
num_stim_bins = 100
cutout = 5000
tau_rates = np.logspace(0, 3, 25)
sigmas = np.logspace(-2.5, 0.5, 12)

MI_neg = [[] for i in range(len(tau_rates))]
MI_pos = [[] for i in range(len(tau_rates))]
MI_tot = [[] for i in range(len(tau_rates))]

for iR, tau_rate in enumerate(tau_rates):
    for sigma in sigmas:
        
        print (tau_rate, sigma)
        
        a = SimplifiedLIFModel(dt=dt, N=num_ticks, odor_tau=odor_tau, odor_mu=odor_mu,
                        odor_sigma=sigma, rate_sigma=rate_sigma, 
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
        
data_dir = dict()

params = dict()
params_to_save = ['dt', 'odor_tau', 'sigmas', 'tau_rates', 'smooth_T', 
                  'num_stim_bins', 'num_rate_bins', 'cutout', 'num_ticks']
for key in params_to_save:
    exec("params['%s'] = %s" % (key, key))
print (params)
data_dir['params'] = params

data_dir['MI_neg'] = MI_neg
data_dir['MI_pos'] = MI_pos
data_dir['MI_tot'] = MI_tot

if SAVE:
    file = './data/LIF_tau_sweep.pkl'
    with open(file, 'wb') as f:
        pickle.dump(data_dir, f)
        
MI_neg = data_dir['MI_neg']
MI_neg = np.array(MI_neg)
MI_tot = data_dir['MI_tot']
MI_tot = np.array(MI_tot)
tau_rates = data_dir['params']['tau_rates']
sigmas = data_dir['params']['sigmas']

ir = 14
print (tau_rates[ir])
fig = gen_plot(1.2, 1.2)
plt.plot(sigmas, MI_neg[ir], color='green')
plt.plot(sigmas, MI_tot[ir], color='k')
plt.xscale('log')
plt.xticks(fontsize=7)
plt.xlim(1e-2, 1)
plt.yticks(np.arange(4), fontsize=7)
plt.ylim(0, 2.5)
if SAVEFIG:
    save_fig('LIF_tau=%1.2f' % tau_rates[ir])
plt.show()

#%% Fig 3C

fig, ax = plt.subplots(figsize=(3.5,3))
X, Y = np.meshgrid(sigmas, tau_rates)
mappable = ax.pcolormesh(X, Y, MI_neg, cmap='afmhot', linewidth=0,
                         rasterized=True, vmin=0, vmax=1.25)
mappable.set_edgecolor('face')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_yticks([1e0, 1e1, 1e2])	
ax.set_xticks([1e-2, 1e-1, 1e0])	
plt.ylim(tau_rates[0], tau_rates[-5])
plt.xlim(1e-2, 1)
ax.tick_params(axis='x', labelsize=17)
ax.tick_params(axis='y', labelsize=17)
cbar = plt.colorbar(mappable)
cbar.ax.tick_params(labelsize=17)
fig.tight_layout()
if SAVEFIG:
    save_fig('LIF_all_taus_neg_MI_heatmap_2')
plt.show()
