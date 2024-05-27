"""
Helper functions

Copyright 2024 Nirag Kadakia

This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import numpy as np

def calc_MI_xy(x, x_bins, y, y_bins):

    num_x_bins = len(x_bins)
    dx = x_bins[1] - x_bins[0]
    dy = y_bins[1] - y_bins[0]

    # Bin the rates (Y) for a given stimulus (X = x) to get P(Y|X = x)
    # H(Y|X) = sum_X P(X) sum_Y H(Y|X = x)
    #          = sum_X P(X) sum_Y p(Y|X = x) log p(Y|X = x)
    H_Y_X = 0
    H_Y = 0
    p_x, _ = np.histogram(x, x_bins, density=True)
    p_x += 1e-8

    # For each time in the stimulus, digitizes the stimulus into stim_bins
    bin_vals = np.digitize(x, x_bins, right=True) - 1
    
    for iS in range(num_x_bins - 1):

        # Get all times where the stimulus is in the iS'th bin
        #idxs_in_bin = np.where(bin_vals == iS)[0]
        idxs_in_bin = (bin_vals == iS)
        
        # Get rates for all times at which stim is in iS'th bin, histogram
        if len(idxs_in_bin) > 0:
            p_Y_x, _ = np.histogram(y[idxs_in_bin], y_bins, density=True)
            H_y_x = -np.nansum(p_Y_x*np.log(p_Y_x)/np.log(2))*dy
            if np.isfinite(H_y_x):
                H_Y_X += p_x[iS]*H_y_x*dx
    
    # H(Y)
    p_Y, _ = np.histogram(y, y_bins, density=True)
    H_Y = -np.nansum(p_Y*np.log(p_Y)/np.log(2))*dy
    MI = H_Y - H_Y_X
    print(MI)

    return MI
