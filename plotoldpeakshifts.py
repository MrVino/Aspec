import numpy as np
import glob
import matplotlib.pyplot as plt
#import matplotlib.ticker as mtick
from scipy.misc import imread, imsave
from scipy import ndimage
import pypylon
import sys
from scipy.signal import savgol_filter#argrelextrema, 
import argparse
import peakutils
import time
import astropy.stats as aps
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from itertools import groupby
import pickle
from Sensor import ASpec
prop_cycle = plt.rcParams['axes.prop_cycle']
plt.rcParams.update({'errorbar.capsize': 2})
colors = prop_cycle.by_key()['color']

'''
image_data0 = imread('Fiberguide_stacked_HoDi_absorption_7fibers_0degrees.tiff').astype(np.float32)


all_pixelpositions_dict = {}

temperatures = [0, 20,50]

for t in temperatures:

    print(t)

    image_data = imread('Fiberguide_stacked_HoDi_absorption_7fibers_'+str(t)+'degrees.tiff').astype(np.float32)
    analyze = ASpec(temp=t)
    analyze.initialize(image=image_data0)
    
    pixelpos_dict = analyze.pixelpositions_of_peaks(image_data, threshold=2, bitsize=8)

    print(analyze.used_fibers)
    all_pixelpositions_dict[t] = pixelpos_dict


#pickle.dump(all_pixelpositions_dict, open( 'old_peak_positions_hodi.p', "wb" ) )
'''
all_pixelpositions_dict = pickle.load( open( "old_peak_positions_hodi.p", "rb" ) )

f, axarr = plt.subplots(6, sharex=True, figsize=(10,12))
f2, axarr2 = plt.subplots(2, sharex=True, figsize=(10,4))


#f.suptitle(r"T = "+str(temp)+"$^\circ$C")


laser_peak_positions = all_pixelpositions_dict[0][2]['wavelengths']

for ip, pp in enumerate(laser_peak_positions):

    axarr[ip].set_title(str(pp)+'nm')
#axarr[1].set_title('637 nm')
#axarr[2].set_title('856 nm')



for t in sorted(all_pixelpositions_dict.keys()):

        for j, k in enumerate(sorted(all_pixelpositions_dict[t].keys())):
            print(k)
            
            axarr2[j].set_title('Fiber'+str(k))
            
            
            #print(pixelpos_dict[k])
            #print(np.mean(pixelpos_dict[k], axis=0))
            
            
            #mean, median, sigma = aps.sigma_clipped_stats(all_pixelpositions_dict[t][k]['positions'], axis=0)
            
            #diff = (mean50-mean20)*analyze20.pxl
            
            
            for i, pos in enumerate(all_pixelpositions_dict[t][k]['positions']):
                print(pos)
                axarr[i].scatter(t, pos-all_pixelpositions_dict[0][k]['positions'][i], label='Fiber'+str(k), color=colors[k-1])
                axarr2[j].scatter(t, pos-all_pixelpositions_dict[0][k]['positions'][i], label=str(laser_peak_positions[i])+'nm', color=colors[i])




axarr[0].legend(loc='best', fancybox=True, framealpha=0.5, fontsize=8)
axarr2[0].legend(loc='best', fancybox=True, framealpha=0.5, fontsize=8)
                    
                    

#axarr[3].set_ylabel(r'$\mathrm{Position\, [um]}$')
axarr[3].set_ylabel(r'$\mathrm{Position\, [pixels]}$')

axarr[5].set_xlabel(r'$\mathrm{Temperature\, [}^{\circ}\mathrm{C]}$')

#axarr2[0].set_ylabel(r'$\mathrm{Position\, [um]}$')
axarr2[0].set_ylabel(r'$\mathrm{Position\, [pixels]}$')

axarr2[1].set_xlabel(r'$\mathrm{Temperature\, [}^{\circ}\mathrm{C]}$')


plt.show()