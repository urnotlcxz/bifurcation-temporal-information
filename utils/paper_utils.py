"""
Functions for generating plot formats for figures for the paper.

Copyright 2024 Nirag Kadakia

This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, 
visit http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""


import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"


def gen_plot(width, height):
	"""
	Generic plot for all figures to get right linewidth, font sizes, 
	and bounding boxes
	"""
	
	fig = plt.figure(figsize=(width, height))
	ax = plt.subplot(111)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.tick_params(direction='in', length=3, width=0.5)
	for axis in ['top','bottom','left','right']:
		ax.spines[axis].set_linewidth(0.5)
	
	return fig, ax

def get_arrow(angle):
	"""
	Isosceles triangle for plot markers
	"""

	angle = -angle + 90
	a = np.deg2rad(angle)
	ar = np.array([[-.25, 0], [.25, 0], [0, .75], [-.25, 0]]).T
	rot = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
	return np.dot(rot, ar).T

def save_fig(fig_name, subdir=None, clear_plot=True, tight_layout=True,	 
			 save_svg=True):
	"""
	Save figures in subfigures folder
	"""
	
	if subdir is None:
		out_dir = '../subfigures'
	else:
		out_dir = '../subfigures/%s' % subdir
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	if tight_layout == True:
		plt.tight_layout()
	filename = '%s/%s' % (out_dir, fig_name)
	plt.savefig('%s.png' % filename, bbox_inches='tight', dpi=300)
	if save_svg == True:
		plt.savefig('%s.svg' % filename, bbox_inches='tight', dpi=300)