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
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']



#This script generates plots that illustrate the method that we use to determine the optimal focus position of the motorized translation stage on which the camera is mounted.At a given temperature, per laser wavelength, we plot the width of the laserpeak as a function of translation stage position. Then we fit a polynomial to the curve of each fiber, and determine the minimum of that curve. Finally we average the minima over the fibers, to determine the optimal focus position.


#peakwidths = pickle.load( open( "laser_peak_widths_T0_final.p", "rb" ) )
peakwidths = pickle.load( open( "laser_peak_widths_T0_16bit_0-40.p", "rb" ) )
#peakwidths = pickle.load( open( "laser_peak_widths_T20_translation_tilt7.5.p", "rb" ) )
#peakwidths2 = pickle.load( open( "laser_peak_widths_T20_translation_tilt7.5_part2-900--1500.p", "rb" ) )
#peakwidths3 = pickle.load( open( "laser_peak_widths_T20_translation_tilt7.5_part3_300-1000.p", "rb" ) )


temp = 0
used_fibers = [1,2,4,5,6,9,10]


#Here we remove the translation stage position larger than 3500 um, and corresponding peakwidths, because they were shown to give strange peakwidths, which complicates the curve fitting.
for t in [temp]:

    indices = np.where(np.asarray(peakwidths[t]['positions'])>3500)[0]
    
    for fibernr in sorted(list(set(peakwidths[t].keys()).difference(set(['positions'])))):
        
        for wl in sorted(peakwidths[t][fibernr].keys()):
      
            peakwidths[t][fibernr][wl] = np.delete(np.array(peakwidths[t][fibernr][wl]), indices)
            
    peakwidths[t]['positions'] = np.delete(np.array(peakwidths[t]['positions']), indices)
    print(t, peakwidths[t]['positions'])



f, axarr = plt.subplots(3, sharex=True, figsize=(10,6))
f2, axarr2 = plt.subplots(len(used_fibers), sharex=True, figsize=(10,len(used_fibers)*3))

f.suptitle(r"T = "+str(temp)+"$^\circ$C")
f2.suptitle(r"T = "+str(temp)+"$^\circ$C")

axarr[0].set_title('420 nm')
axarr[1].set_title('637 nm')
axarr[2].set_title('856 nm')


optima402 = []
optima637 = []
optima856 = []


#Sort the positions and corresponding peakwidths
pos_sorting_indices = np.argsort(peakwidths[temp]['positions'])
sorted_positions = np.array(peakwidths[temp]['positions'])[pos_sorting_indices]
#Create an array for the curve fitting routine
x = np.linspace(sorted_positions[0],sorted_positions[-1],10000)

        
for fiber_index, fibernr in enumerate(sorted(used_fibers)):

    axarr2[fiber_index].set_title('Fiber '+str(fibernr))
    
    for wl in sorted(peakwidths[temp][fibernr].keys()):

        
        sorted_wavelengths = np.array(peakwidths[temp][fibernr][wl])[pos_sorting_indices]
           
        #Fit a polynomial to the curve. A polynomial order of 6 gave the best results
        polcoefs = np.polyfit(sorted_positions, sorted_wavelengths, 6)
        
        #Determine the minima of the polynomial by finding the real points where the derivative equals 0.
        derivative = np.polyder(polcoefs)
        roots = np.roots(derivative)
        realroots = roots[np.isreal(roots)]
        
        #There can be more than 1 real root of the derivative, so we choose the one(s) that are in the domain for which the measured the peakwidths 
        likelyroots = realroots[np.where(np.logical_and(realroots>=sorted_positions[0], realroots<=sorted_positions[-1]))]
        
        #In case there's still more than 1 possible root, we choose the root for which the polynomial returns the smallest value
        ind_min = np.argmin(np.polyval(polcoefs, likelyroots))


        #Add the found minimum for each fiber to the array of the corresponding wavelength, such that we can take the mean later.
        if wl == 402:
            pi = 0
            
            optima402.append(likelyroots[np.argmin(np.polyval(polcoefs, likelyroots))])
            
        elif wl == 637:
            pi = 1
            
            optima637.append(likelyroots[np.argmin(np.polyval(polcoefs, likelyroots))])           
            
        elif wl == 856:

            optima856.append(likelyroots[np.argmin(np.polyval(polcoefs, likelyroots))])

            pi = 2

        axarr[pi].plot(sorted_positions, sorted_wavelengths, label = "Fiber "+str(fibernr), color=colors[fibernr-1])
        axarr[pi].plot(x, np.polyval(polcoefs, x), linestyle='dashed', color='black')
        
        
        axarr[pi].vlines(likelyroots[np.argmin(np.polyval(polcoefs, likelyroots))], 0, 10, linestyle='dotted', color='black')

        
        
        axarr2[fiber_index].plot(sorted_positions, sorted_wavelengths, label = str(wl)+' nm', color=colors[pi**2+2*pi])


'''
#This is another method that we tried, but we no longer used. For each wavelength we averaged the data over all fibers, and then fitted a polynomial to that average curve (contrary to fitting a polynomial to the curve of each fiber seperately). We decided not to use this method, because the method above gave better results when visually comparing the determined minima to the curves.
averages_402 = []
averages_637 = []
averages_856 = []

for j in np.arange(len(peakwidths[temp]['positions'])):

    averages_402.append(np.mean([peakwidths[temp][i][402][j] for i in used_fibers]))
    averages_637.append(np.mean([peakwidths[temp][i][637][j] for i in used_fibers]))   
    averages_856.append(np.mean([peakwidths[temp][i][856][j] for i in used_fibers]))

sorted_averages_402 = np.array(averages_402)[pos_sorting_indices]
sorted_averages_637 = np.array(averages_637)[pos_sorting_indices]
sorted_averages_856 = np.array(averages_856)[pos_sorting_indices]



polcoefs_402 = np.polyfit(sorted_positions, sorted_averages_402, 6)

derivative_402 = np.polyder(polcoefs_402)
roots_402 = np.roots(derivative_402)
realroots_402 = roots_402[np.isreal(roots_402)]

likelyroot_402 = realroots_402[np.where(np.logical_and(realroots_402>=sorted_positions[0], realroots_402<=sorted_positions[-1]))]


polcoefs_637 = np.polyfit(sorted_positions, sorted_averages_637, 6)

derivative_637 = np.polyder(polcoefs_637)
roots_637 = np.roots(derivative_637)
realroots_637 = roots_637[np.isreal(roots_637)]

likelyroot_637 = realroots_637[np.where(np.logical_and(realroots_637>=sorted_positions[0], realroots_637<=sorted_positions[-1]))]


polcoefs_856 = np.polyfit(sorted_positions, sorted_averages_856, 6)

derivative_856 = np.polyder(polcoefs_856)
roots_856 = np.roots(derivative_856)
realroots_856 = roots_856[np.isreal(roots_856)]

likelyroot_856 = realroots_856[np.where(np.logical_and(realroots_856>=sorted_positions[0], realroots_856<=sorted_positions[-1]))]

print(likelyroot_402, np.mean(optima402))
print(likelyroot_637, np.mean(optima637))
print(likelyroot_856, np.mean(optima856))


#axarr[0].plot(x, np.polyval(polcoefs_402, x), linestyle='dashed', color='yellow')
#axarr[1].plot(x, np.polyval(polcoefs_637, x), linestyle='dashed', color='yellow')
#axarr[2].plot(x, np.polyval(polcoefs_856, x), linestyle='dashed', color='yellow')


#axarr[0].plot(sorted_positions, sorted_averages_402, label = "Averaged", color='blue', linestyle='dashed')
#axarr[1].plot(sorted_positions, sorted_averages_637, label = "Averaged", color='blue', linestyle='dashed')
#axarr[2].plot(sorted_positions, sorted_averages_856, label = "Averaged", color='blue', linestyle='dashed')


#axarr[0].vlines(likelyroot_402, 0, 10, linestyle='dotted', color='blue')
#axarr[1].vlines(likelyroot_637, 0, 10, linestyle='dotted', color='blue')
#axarr[2].vlines(likelyroot_856, 0, 10, linestyle='dotted', color='blue')
'''

axarr[0].vlines(np.mean(optima402), 0, 10, linestyle='dotted', color='red')
axarr[1].vlines(np.mean(optima637), 0, 10, linestyle='dotted', color='red')
axarr[2].vlines(np.mean(optima856), 0, 10, linestyle='dotted', color='red')

axarr[0].legend(loc='best', fancybox=True, framealpha=0.5, fontsize=8)
axarr2[0].legend(loc='best', fancybox=True, framealpha=0.5, fontsize=8)

axarr[1].set_ylabel(r'$\mathrm{peakwidth\, [nm]}$')
axarr2[int(len(used_fibers)/2)].set_ylabel(r'$\mathrm{peakwidth\, [nm]}$')
    
axarr[2].set_xlabel(r'$\mathrm{Actuator\,position\, [}\mu \mathrm{m]}$')


plt.show()