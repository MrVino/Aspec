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


peakwidths = pickle.load( open( "laser_peak_widths_T0_final.p", "rb" ) )

#peakwidths = pickle.load( open( "laser_peak_widths_T20_translation_tilt7.5.p", "rb" ) )
#peakwidths2 = pickle.load( open( "laser_peak_widths_T20_translation_tilt7.5_part2-900--1500.p", "rb" ) )
#peakwidths3 = pickle.load( open( "laser_peak_widths_T20_translation_tilt7.5_part3_300-1000.p", "rb" ) )


temp = 0
used_fibers = [1,2,4,5,6,9,10]




#print(peakwidths[temp]['positions'])
'''
for f in peakwidths[temp].keys():

    if f == 'positions':

        peakwidths[temp][f].extend(peakwidths2[temp][f])
        peakwidths[temp][f].extend(peakwidths3[temp][f])

    else:
    
        for w in peakwidths[temp][f].keys():
            peakwidths[temp][f][w].extend(peakwidths2[temp][f][w])
            peakwidths[temp][f][w].extend(peakwidths3[temp][f][w])
    
  
'''    

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

pos_sorting_indices = np.argsort(peakwidths[temp]['positions'])
sorted_positions = np.array(peakwidths[temp]['positions'])[pos_sorting_indices]
x = np.linspace(sorted_positions[0],sorted_positions[-1],10000)


        
for fiber_index, fibernr in enumerate(sorted(used_fibers)):

    axarr2[fiber_index].set_title('Fiber '+str(fibernr))
    
    for wl in sorted(peakwidths[temp][fibernr].keys()):

        
        sorted_wavelengths = np.array(peakwidths[temp][fibernr][wl])[pos_sorting_indices]
           
        
        polcoefs = np.polyfit(sorted_positions, sorted_wavelengths, 6)
        
        derivative = np.polyder(polcoefs)
        roots = np.roots(derivative)
        realroots = roots[np.isreal(roots)]
        
        likelyroots = realroots[np.where(np.logical_and(realroots>=sorted_positions[0], realroots<=sorted_positions[-1]))]
        
        
        ind_min = np.argmin(np.polyval(polcoefs, likelyroots))
        print(likelyroots)
        #print(likelyroots[ind_min])
        print(likelyroots[np.argmin(np.polyval(polcoefs, likelyroots))])

        #print('------')
        #print(likelyroots)
        #print('------')
        #polcoefs4 = np.polyfit(sorted_positions, sorted_wavelengths, 4)
        #polcoefs6 = np.polyfit(sorted_positions, sorted_wavelengths, 6)
        



        if wl == 402:
            pi = 0

            if len(likelyroots)==1:
            
                optima402.append(likelyroots)
            
        elif wl == 637:
            pi = 1
 
            if len(likelyroots)==1:
            
                optima637.append(likelyroots)           
            
        elif wl == 856:

            if len(likelyroots)==1:
            
                optima856.append(likelyroots)

            pi = 2

        axarr[pi].plot(sorted_positions, sorted_wavelengths, label = "Fiber "+str(fibernr), color=colors[fibernr-1])
        axarr[pi].plot(x, np.polyval(polcoefs, x), linestyle='dashed', color='black')
        
        if len(likelyroots)==1:
        
            axarr[pi].vlines(likelyroots, 0, 10, linestyle='dotted', color='black')
            #print(likelyroots)
        #axarr[pi].plot(x, np.polyval(polcoefs4, x), linestyle='dashed', label='O(4)')
        #axarr[pi].plot(x, np.polyval(polcoefs6, x), linestyle='dashed', label='O(6)')
        
        
        axarr2[fiber_index].plot(sorted_positions, sorted_wavelengths, label = str(wl)+' nm', color=colors[pi**2+2*pi])



#print(optima402, np.mean(optima402))
#print(optima637, np.mean(optima637))        
#print(optima856, np.mean(optima856))        


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


axarr[0].plot(x, np.polyval(polcoefs_402, x), linestyle='dashed', color='yellow')
axarr[1].plot(x, np.polyval(polcoefs_637, x), linestyle='dashed', color='yellow')
axarr[2].plot(x, np.polyval(polcoefs_856, x), linestyle='dashed', color='yellow')


axarr[0].plot(sorted_positions, sorted_averages_402, label = "Averaged", color='blue', linestyle='dotted')
axarr[1].plot(sorted_positions, sorted_averages_637, label = "Averaged", color='blue', linestyle='dotted')
axarr[2].plot(sorted_positions, sorted_averages_856, label = "Averaged", color='blue', linestyle='dotted')

axarr[0].vlines(np.mean(optima402), 0, 10, linestyle='dotted', color='red')
axarr[1].vlines(np.mean(optima637), 0, 10, linestyle='dotted', color='red')
axarr[2].vlines(np.mean(optima856), 0, 10, linestyle='dotted', color='red')



axarr[0].vlines(likelyroot_402, 0, 10, linestyle='dotted', color='blue')
axarr[1].vlines(likelyroot_637, 0, 10, linestyle='dotted', color='blue')
axarr[2].vlines(likelyroot_856, 0, 10, linestyle='dotted', color='blue')


axarr[0].legend(loc='best', fancybox=True, framealpha=0.5, fontsize=8)
axarr2[0].legend(loc='best', fancybox=True, framealpha=0.5, fontsize=8)

axarr[1].set_ylabel(r'$\mathrm{peakwidth\, [nm]}$')
axarr2[int(len(used_fibers)/2)].set_ylabel(r'$\mathrm{peakwidth\, [nm]}$')
    
axarr[2].set_xlabel(r'$\mathrm{translation\, [um]}$')

plt.show()