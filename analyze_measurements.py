import numpy as np
import glob
import os
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
from sklearn import linear_model
prop_cycle = plt.rcParams['axes.prop_cycle']
plt.rcParams.update({'errorbar.capsize': 2})
colors = prop_cycle.by_key()['color']



#ENTER THE TRANSLATION STAGE POSITION FOR WHICH YOU WANT TO ANALYZE THE FWHM OF THE LASER PEAKS AT DIFFERENT TEMPERATURES HERE
#Translation stage position for which we're going to determine the peakshift. Generally we choose the position that is closest to the optimal one for 20 C
translationstage_position = 0

# If the files 'peak_positions_all_temperatures.p' and '(focus_)laser_peak_widths_all_temperatures.p' are already in the dictionary, then set Dictionaries_made to True
Dictionaries_made = True

#Have you used the actuator for the measurements? If yes, set actuator = True, else set actuator = False
actuator = True


#This function determines the optimal focus position of the motorized translation stage on which the camera is mounted. At a given temperature, per laser wavelength, we look at the width of the laserpeak as a function of translation stage position. Then we fit a polynomial to the curve of each fiber, and determine the minimum of that curve. Finally we average the minima over the fibers, to determine the optimal focus position.


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


laser_peak_positions = [402, 637, 856]

all_directories = os.listdir()

temperatures = []

for p in all_directories:
    if p[-1] == 'C':
    
        try:
            temperatures.append(int(p[:-1]))
            
        except ValueError:
            temperatures.append(float(p[:-1]))

print("Analyzing the data for the following temperatures: ", temperatures)

#Choose a reference temperature (could also be another temperature as long as the temp argument and initialize image correspond to the same temperature)

Tref = min(temperatures)

print("Taking ", Tref, "C as the reference temperature")

if actuator is True:

    if Dictionaries_made is False:
        peakwidths = pickle.load( open( "./"+str(Tref)+"C/laser_peak_widths_T"+str(Tref)+".p", "rb" ) )
        
        for t in set(temperatures).difference(set([Tref])):
        
            peakwidths.update(pickle.load( open( "./"+str(t)+"C/laser_peak_widths_T"+str(t)+".p", "rb" ) ))
        
        pickle.dump(peakwidths, open( 'laser_peak_widths_all_temperatures.p', "wb" ) )
    else:
        peakwidths = pickle.load( open( "laser_peak_widths_all_temperatures.p", "rb" ) )
    
    
    
    optima = determine_focus_positions(peakwidths)
    
    
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
    
    
    plt.errorbar(optima['averaged']['temperatures'], optima['averaged']['mean402'], yerr=optima['averaged']['std402'], fmt='-x', color=colors[0], label='402 nm', ecolor='black', capthick=2)
    plt.errorbar(optima['averaged']['temperatures'], optima['averaged']['mean637'], yerr=optima['averaged']['std637'], fmt='-x', color=colors[3], label='637 nm', ecolor='black', capthick=2)
    plt.errorbar(optima['averaged']['temperatures'], optima['averaged']['mean856'], yerr=optima['averaged']['std856'], fmt='-x', color=colors[8], label='856 nm', ecolor='black', capthick=2)
    
    
    polcoefs_402 = np.polyfit(np.array(linear_regression_temperatures).flatten(), np.array(linear_regression_positions_402).flatten(), 1)
    polcoefs_637 = np.polyfit(np.array(linear_regression_temperatures).flatten(), np.array(linear_regression_positions_637).flatten(), 1)
    polcoefs_856 = np.polyfit(np.array(linear_regression_temperatures).flatten(), np.array(linear_regression_positions_856).flatten(), 1)
    
    plt.plot(linear_regression_temperatures, np.polyval(polcoefs_402, linear_regression_temperatures), linestyle='dashed', color='black')
    plt.plot(linear_regression_temperatures, np.polyval(polcoefs_637, linear_regression_temperatures), linestyle='dashed', color='black')
    plt.plot(linear_regression_temperatures, np.polyval(polcoefs_856, linear_regression_temperatures), linestyle='dashed', color='black')
    
    plt.legend(loc='best', fancybox=True, framealpha=0.5, frameon=False)   
    
    
    for i in [0,1]:
        print("Linear regression "+str(i)+" order coefficient for focus shift of 402 nm = ", polcoefs_402[i])
        print("Linear regression "+str(i)+" order coefficient for focus shift of 637 nm = ", polcoefs_637[i])
        print("Linear regression "+str(i)+" order coefficient for focus shift of 856 nm = ", polcoefs_856[i])
    
    plt.ylabel(r'$\mathrm{Actuator\,position\, [}\mu \mathrm{m]}$')
    
    plt.xlabel(r'$\mathrm{Temperature\, [}^{\circ}\mathrm{C]}$')
    plt.show()    



if Dictionaries_made is False:
    #Here we create a dictionary for the pixel positions of the laser peaks on the detector at all temperatures. For a given position of the translation stage, we look how the pixel positions of the laser peaks shift with temperature.
    #We only have to generate the dictionary once, which takes some time due to the calculations of the peak positions, and save it such that we can easily load the generated dictionary later, without having to re-do the peak position calculations.
    all_pixelpositions_dict = {}
    focus_peakwidths = {}
    
    
    image_data_reference = imread('./'+str(Tref)+'C/ClimateChamberT'+str(Tref)+'_pos'+str(translationstage_position)+'.tiff').astype(np.float32)
    analyze = ASpec(temp=Tref, lasers=True)
    analyze.initialize(image=image_data_reference)    
    
    for t in temperatures:
        focus_peakwidths[t] = {}
        image_data = imread('./'+str(t)+'C/ClimateChamberT'+str(t)+'_pos'+str(translationstage_position)+'.tiff').astype(np.float32)

     
        for f in analyze.used_fibers:    
        
            #Create an empty array to which we add the position of the actuator
            focus_peakwidths[t]['positions'] = []
            #Create empty arrays to which we add the peakwidth for different laser wavelenghts corresponding to each actuator position
            focus_peakwidths[t][f] = {402:[], 637:[], 856:[]}
      
                    
        
        pixelpos_dict = analyze.pixelpositions_of_peaks(image_data, threshold=25, fitpeaks=laser_peak_positions)
    
        wl_dict = analyze.backwards_spectral_fitting(image_data, resolution=1)
        
        analyze.determine_peak_widths(wl_dict, focus_peakwidths[t])
        
        all_pixelpositions_dict[t] = pixelpos_dict
    
    
    
    pickle.dump(focus_peakwidths, open( 'focus_laser_peak_widths_all_temperatures.p', "wb" ) )
    pickle.dump(all_pixelpositions_dict, open( 'peak_positions_all_temperatures.p', "wb" ) )


all_pixelpositions_dict = pickle.load( open( "peak_positions_all_temperatures.p", "rb" ) )
focus_peakwidths = pickle.load( open( "focus_laser_peak_widths_all_temperatures.p", "rb" ) )



f, axarr = plt.subplots(3, sharex=True, figsize=(10,6))
f2, axarr2 = plt.subplots(3, sharex=True, figsize=(10,6))
f.suptitle('Shift in peak position')
f2.suptitle('Peak widths')

axarr[0].set_title('402 nm')
axarr[1].set_title('637 nm')
axarr[2].set_title('856 nm')


axarr2[0].set_title('402 nm')
axarr2[1].set_title('637 nm')
axarr2[2].set_title('856 nm')

temperatures = []
shifts_402 = []
shifts_637 = []
shifts_856 = []


for t in sorted(all_pixelpositions_dict.keys()):

        for j, fibernr in enumerate(sorted(all_pixelpositions_dict[t].keys())):

            for i, pos in enumerate(all_pixelpositions_dict[t][fibernr]['positions']):


                if i==0:
                    temperatures.append(int(t))
                    shifts_402.append(pos-all_pixelpositions_dict[Tref][fibernr]['positions'][i])

                if i==1:
                    shifts_637.append(pos-all_pixelpositions_dict[Tref][fibernr]['positions'][i])

                if i==2:
                    shifts_856.append(pos-all_pixelpositions_dict[Tref][fibernr]['positions'][i])                
                
                if i==0 and t==Tref:
                    axarr[i].scatter(int(t), pos-all_pixelpositions_dict[Tref][fibernr]['positions'][i], label='Fiber'+str(fibernr), color=colors[fibernr-1])


                else:
                    axarr[i].scatter(int(t), pos-all_pixelpositions_dict[Tref][fibernr]['positions'][i], color=colors[fibernr-1])
                    
            
            for wl in sorted(focus_peakwidths[t][fibernr].keys()):
                #Make sure that each wavelength is plotted in the correct subplot.
                if wl == 402:
                    pi = 0
                elif wl == 637:
                    pi = 1
                elif wl == 856:
                    pi = 2
                if pi==0 and t==Tref:
                    axarr2[pi].scatter(int(t), focus_peakwidths[t][fibernr][wl], label = "Fiber "+str(fibernr), color=colors[fibernr-1])
                else:
                    axarr2[pi].scatter(int(t), focus_peakwidths[t][fibernr][wl], color=colors[fibernr-1])

axarr[0].legend(loc='best', fancybox=True, framealpha=0.5, fontsize=8)
axarr2[0].legend(loc='best', fancybox=True, framealpha=0.5, fontsize=8)


                   
 
model_ransac_402 = linear_model.RANSACRegressor(residual_threshold=0.5, max_trials=1000)#,min_samples=X.shape[0])
model_ransac_402.fit(np.array(temperatures)[:,np.newaxis], np.array(shifts_402))
model_ransac_637 = linear_model.RANSACRegressor(residual_threshold=0.5, max_trials=1000)#,min_samples=X.shape[0])
model_ransac_637.fit(np.array(temperatures)[:,np.newaxis], np.array(shifts_637))
model_ransac_856 = linear_model.RANSACRegressor(residual_threshold=0.5, max_trials=1000)#,min_samples=X.shape[0])
model_ransac_856.fit(np.array(temperatures)[:,np.newaxis], np.array(shifts_856))



polcoefs_402 = np.polyfit(np.array(temperatures).flatten(), np.array(shifts_402).flatten(), 1)
polcoefs_637 = np.polyfit(np.array(temperatures).flatten(), np.array(shifts_637).flatten(), 1)
polcoefs_856 = np.polyfit(np.array(temperatures).flatten(), np.array(shifts_856).flatten(), 1)








# Predict data of estimated models
line_X = np.linspace(0,40,41)
line_y_ransac_402 = model_ransac_402.predict(line_X[:, np.newaxis])
line_y_ransac_637 = model_ransac_637.predict(line_X[:, np.newaxis]) 
line_y_ransac_856 = model_ransac_856.predict(line_X[:, np.newaxis]) 

for i in np.arange(3):

    if i == 2:

        axarr[i].plot(line_X, -0.04*line_X, linestyle='solid', alpha=0.5, color='black', label='Theory')
        axarr[i].plot(line_X, line_y_ransac_402, linestyle='dashed', alpha=0.5, color='blue', label='Ransac 402 nm')
        axarr[i].plot(line_X, line_y_ransac_637, linestyle='dashed', alpha=0.5, color='red', label='Ransac 637 nm') 
        axarr[i].plot(line_X, line_y_ransac_856, linestyle='dashed', alpha=0.5, color='grey', label='Ransac 856 nm') 
        axarr[i].plot(line_X, np.polyval(polcoefs_402, line_X), linestyle='dotted', color='blue', label='Linear 402 nm')
        axarr[i].plot(line_X, np.polyval(polcoefs_637, line_X), linestyle='dotted', color='red', label='Linear 637 nm')
        axarr[i].plot(line_X, np.polyval(polcoefs_856, line_X), linestyle='dotted', color='grey', label='Linear 856 nm')
    
    else:

        axarr[i].plot(line_X, -0.04*line_X, linestyle='solid', alpha=0.5, color='black')
        axarr[i].plot(line_X, line_y_ransac_402, linestyle='dashed', alpha=0.5, color='blue')
        axarr[i].plot(line_X, line_y_ransac_637, linestyle='dashed', alpha=0.5, color='red') 
        axarr[i].plot(line_X, line_y_ransac_856, linestyle='dashed', alpha=0.5, color='grey') 
        axarr[i].plot(line_X, np.polyval(polcoefs_402, line_X), linestyle='dotted', color='blue')
        axarr[i].plot(line_X, np.polyval(polcoefs_637, line_X), linestyle='dotted', color='red')
        axarr[i].plot(line_X, np.polyval(polcoefs_856, line_X), linestyle='dotted', color='grey')

                    

axarr[2].legend(loc='best', fancybox=True, framealpha=0.5, fontsize=8)

#axarr[1].set_ylabel(r'$\mathrm{Position\, [um]}$')
axarr[1].set_ylabel(r'$\mathrm{Position\, [pixels]}$')

axarr[2].set_xlabel(r'$\mathrm{Temperature\, [}^{\circ}\mathrm{C]}$')




axarr2[1].set_ylabel(r'$\mathrm{FWHM\, [nm]}$')

axarr2[2].set_xlabel(r'$\mathrm{Temperature\, [}^{\circ}\mathrm{C]}$')

plt.show()







