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


def determine_focus_positions(peakwidths_dict):
    optima = {'averaged':{}, 'all':{}}
    optima['averaged'] = {'temperatures':[], 'mean402':[], 'mean637':[], 'mean856':[], 'std402':[], 'std637':[], 'std856':[]} 
    
    for t in sorted(peakwidths_dict.keys()):
        
        optima['averaged']['temperatures'].append(t)
        
        optima['all'][t] = {402:[], 637:[], 856:[]}
        

        pos_sorting_indices = np.argsort(peakwidths_dict[t]['positions'])
        sorted_positions = np.array(peakwidths_dict[t]['positions'])[pos_sorting_indices]

        
        for fibernr in sorted(list(set(peakwidths_dict[t].keys()).difference(set(['positions'])))):
        
            
            for wl in sorted(peakwidths_dict[t][fibernr].keys()):
        
                
                sorted_wavelengths = np.array(peakwidths_dict[t][fibernr][wl])[pos_sorting_indices]
                   
                
                polcoefs = np.polyfit(sorted_positions, sorted_wavelengths, 6)
                
                derivative = np.polyder(polcoefs)
                roots = np.roots(derivative)
                realroots = roots[np.isreal(roots)]
                
                likelyroots = realroots[np.where(np.logical_and(realroots>=sorted_positions[0], realroots<=sorted_positions[-1]))]

                #If there's more than 1 real root in the domain, we take the root that belongs to the absolute minimum within the domain
                optima['all'][t][wl].append(likelyroots[np.argmin(np.polyval(polcoefs, likelyroots))])
                    
        for wlt in optima['all'][t].keys():
        
            optima['averaged']['mean'+str(wlt)].append(np.mean(optima['all'][t][wlt]))
            optima['averaged']['std'+str(wlt)].append(np.std(optima['all'][t][wlt]))


    return optima


temperatures = [0, 10, 20, 30, 40]

peakwidths = pickle.load( open( "laser_peak_widths_T20_final_start.p", "rb" ) )

for t in set(temperatures).difference(set([20])):

    peakwidths.update(pickle.load( open( "laser_peak_widths_T"+str(t)+"_final.p", "rb" ) ))




peakwidths_round2 = pickle.load( open( "laser_peak_widths_T20_final_end.p", "rb" ) )
peakwidths_round2.update(pickle.load( open( "laser_peak_widths_T0_round2.p", "rb" ) ))



optima = determine_focus_positions(peakwidths)
optima_round2 = determine_focus_positions(peakwidths_round2)

linear_regression_positions_402 = []
linear_regression_positions_637 = []
linear_regression_positions_856 = []
linear_regression_temperatures = []


plt.figure()

for t in sorted(optima['all'].keys()):
    
    linear_regression_positions_402.append(optima['all'][t][402])
    linear_regression_positions_637.append(optima['all'][t][637])
    linear_regression_positions_856.append(optima['all'][t][856])
    linear_regression_temperatures.append(t*np.ones_like(optima['all'][t][402]))
    
    
    plt.scatter(t*np.ones_like(optima['all'][t][402]), optima['all'][t][402], color=colors[0])
    plt.scatter(t*np.ones_like(optima['all'][t][637]), optima['all'][t][637], color=colors[3])
    plt.scatter(t*np.ones_like(optima['all'][t][856]), optima['all'][t][856], color=colors[8])


for t in optima_round2['all'].keys():

    plt.scatter(t*np.ones_like(optima_round2['all'][t][402]), optima_round2['all'][t][402], color=colors[0], marker='X')
    plt.scatter(t*np.ones_like(optima_round2['all'][t][637]), optima_round2['all'][t][637], color=colors[3], marker='X')
    plt.scatter(t*np.ones_like(optima_round2['all'][t][856]), optima_round2['all'][t][856], color=colors[8], marker='X')



plt.errorbar(optima['averaged']['temperatures'], optima['averaged']['mean402'], yerr=optima['averaged']['std402'], fmt='-x', color=colors[0], label='402 nm', ecolor='black', capthick=2)
plt.errorbar(optima['averaged']['temperatures'], optima['averaged']['mean637'], yerr=optima['averaged']['std637'], fmt='-x', color=colors[3], label='637 nm', ecolor='black', capthick=2)
plt.errorbar(optima['averaged']['temperatures'], optima['averaged']['mean856'], yerr=optima['averaged']['std856'], fmt='-x', color=colors[8], label='856 nm', ecolor='black', capthick=2)

#print(np.array(linear_regression_temperatures).flatten())
#print(

polcoefs_402 = np.polyfit(np.array(linear_regression_temperatures).flatten(), np.array(linear_regression_positions_402).flatten(), 1)
polcoefs_637 = np.polyfit(np.array(linear_regression_temperatures).flatten(), np.array(linear_regression_positions_637).flatten(), 1)
polcoefs_856 = np.polyfit(np.array(linear_regression_temperatures).flatten(), np.array(linear_regression_positions_856).flatten(), 1)

plt.plot(linear_regression_temperatures, np.polyval(polcoefs_402, linear_regression_temperatures), linestyle='dashed', color='black')
plt.plot(linear_regression_temperatures, np.polyval(polcoefs_637, linear_regression_temperatures), linestyle='dashed', color='black')
plt.plot(linear_regression_temperatures, np.polyval(polcoefs_856, linear_regression_temperatures), linestyle='dashed', color='black')

plt.legend(loc='best', fancybox=True, framealpha=0.5, frameon=False)   


for i in [0,1]:
    print(polcoefs_402[i], polcoefs_637[i], polcoefs_856[i])    


plt.ylabel(r'$\mathrm{Actuator\,position\, [}\mu \mathrm{m]}$')

plt.xlabel(r'$\mathrm{Temperature\, [}^{\circ}\mathrm{C]}$')
plt.show()    
laser_peak_positions = [402, 637, 856]


'''

image_data20 = imread('../Basler spectra/Lasers/ClimateChamberT20_pos2500.tiff').astype(np.float32)


for t in [0,20]:#temperatures:

    print(t)

    image_data = imread('../Basler spectra/Lasers/ClimateChamberT'+str(t)+'_pos2500_round2.tiff').astype(np.float32)
    analyze = ASpec(temp=t, lasers=True)
    analyze.initialize(image=image_data20)
    
    pixelpos_dict = analyze.pixelpositions_of_peaks(image_data, threshold=25, fitpeaks=laser_peak_positions)

    pixelpositions_dict[t] = pixelpos_dict


#pickle.dump(pixelpositions_dict, open( 'peak_positions_0_20C_round2.p', "wb" ) )


all_pixelpositions_dict = {}

for t in temperatures:

    print(t)

    image_data = imread('../Basler spectra/Lasers/ClimateChamberT'+str(t)+'_pos2500.tiff').astype(np.float32)
    analyze = ASpec(temp=t, lasers=True)
    analyze.initialize(image=image_data20)
    
    pixelpos_dict = analyze.pixelpositions_of_peaks(image_data, threshold=25, fitpeaks=laser_peak_positions)

    all_pixelpositions_dict[t] = pixelpos_dict
'''

pixelpositions_dict = pickle.load( open( "peak_positions_0_20C_round2.p", "rb" ) )
all_pixelpositions_dict = pickle.load( open( "peak_shifts_0-40C.p", "rb" ) )

f, axarr = plt.subplots(3, sharex=True, figsize=(10,6))
f2, axarr2 = plt.subplots(7, sharex=True, figsize=(10,14))


#f.suptitle(r"T = "+str(temp)+"$^\circ$C")


axarr[0].set_title('420 nm')
axarr[1].set_title('637 nm')
axarr[2].set_title('856 nm')



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
                axarr2[j].scatter(t, pos-all_pixelpositions_dict[0][k]['positions'][i], label=str(laser_peak_positions[i])+'nm', color=colors[i**2+2*i])


for t in sorted(pixelpositions_dict.keys()):

        for j, k in enumerate(sorted(pixelpositions_dict[t].keys())):
            print(k)
            

        
            
            
            for i, pos in enumerate(pixelpositions_dict[t][k]['positions']):
                print(pos)
                axarr[i].scatter(t, pos-all_pixelpositions_dict[0][k]['positions'][i], color=colors[k-1], marker='X')
                axarr2[j].scatter(t, pos-all_pixelpositions_dict[0][k]['positions'][i], color=colors[i**2+2*i], marker='X')




        


axarr[0].legend(loc='best', fancybox=True, framealpha=0.5, fontsize=8)
axarr2[0].legend(loc='best', fancybox=True, framealpha=0.5, fontsize=8)
                    
                    

#axarr[1].set_ylabel(r'$\mathrm{Position\, [um]}$')
axarr[1].set_ylabel(r'$\mathrm{Position\, [pixels]}$')

axarr[2].set_xlabel(r'$\mathrm{Temperature\, [}^{\circ}\mathrm{C]}$')

#axarr2[3].set_ylabel(r'$\mathrm{Position\, [um]}$')
axarr2[3].set_ylabel(r'$\mathrm{Position\, [pixels]}$')

axarr2[6].set_xlabel(r'$\mathrm{Temperature\, [}^{\circ}\mathrm{C]}$')


plt.show()










