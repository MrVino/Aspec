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
import os
import imageio
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

class ASpec(object):
    def __init__(self, image=None, lambda_min=350., nx=1920, ny=1200, pixelsize= 5.86, linesmm = 150, m=1, NA=0.1, grating_tilt_angle = 15., f=100., temp=20, lasers=False):
       
       
        self.temperature = temp
       
       
        #number of pixels in the x-direction
        self.nx = nx
        #number of pixels in the y-direction
        self.ny = ny
        #pixelsize (inserted in micron, converted here to nm)
        self.pxl = pixelsize*1.e3
        #grating groove spacing (in nm converted from lines/mm)
        self.d = 1.e6/linesmm
        #diffraction order
        self.m = m
        #Numerical Aperture of the fiber
        self.NA = NA
        #angle of incoming beam with respect to grating normal (theta) = angle incoming beam w.r.t. untilted grating + grating tilt angle
        self.theta = np.radians(grating_tilt_angle) #np.arcsin(NA)/2. + 
        #Camera mirror focal length (inserted in mm, converted here to nm)
        self.f = f*1.e6
        #wavelength range lower limit (used to position spectrum on detector)      
        self.lambda_min = lambda_min
        
        #Reference detector position (in nm)
        self.p0 = np.tan(self.exit_angle(lambda_min))*self.f

        #Are we looking at the three emission lines of the lasers (lasers = True) or at a ('full') spectrum (lasers=False). 
        #Used in the separate_fibers routine where we need to adjust the signal in the image for the lasers.
        self.lasers = lasers

        #Set the peak positions of the calibration filter (DON'T FORGET TO CHANGE WHEN FILTER IS RECALIBRATED)
        self.LiquidCalibrationFilter()
        
        #Determine how many fibers are used. This sets the number of fibers, self.n_fibers, and the polynomial coefficients 
        #that belong to each fiber, self.fiber_positions_polynomial_coefficients.   
        #if image is None:
        self.connect_camera()
        '''
            for image_data in self.take_snapshots(1):
                image = image_data
        '''                      

        #These values are used to identify each fiber based on its position on the detector
        #We assume that each fibers covers an equal number of rows specified by self.fiber_spacing.
        #The edge between two adjacent fibers is emperically determined.  
        self.fiber_spacing = 110 #rows
        self.fiber_edge_rows = np.arange(1150, 49, -self.fiber_spacing)

        print("Separating fibers")
        self.separate_fibers(image)

        #Open the files that contain the calibrated polynomial coefficients necessary for the spectral fitting of each fiber
        #Note that they are currently assumed to be in the same directory as from which this script is run.
        polynomial_files = glob.glob("polynomial_coefficients_fiber*")
    
        #Create a (nested) Python dictionary in which the coefficients are stored for each fiber. 
        #The dictionary uses key-value pairs, where the first key is the fiber number
        self.spectral_fitting_polynomial_coefficients = {}
        for f in polynomial_files:
            #Extract the fiber number from the filename and use it as key in the dictionary
            fibernr = int(f[29:-4])
            #Using this fiber number as a key we create a nested dictionary
            self.spectral_fitting_polynomial_coefficients[fibernr] = {}
            #In this nested dictionary the keys correspond to the order of the polynomial coefficient 
            #and the values are the calibrated polynomial coefficients that are necesarry to determine them
            self.spectral_fitting_polynomial_coefficients[fibernr]['0th order'] =  np.loadtxt(f)[0]
            self.spectral_fitting_polynomial_coefficients[fibernr]['1st order'] =  np.loadtxt(f)[1]


    def Gaussian(self, x, a, b, c, d):
        
        y = a*np.exp(-np.power((x - b), 2.)/(2. * c**2.)) + d
        
        return y



    def LiquidCalibrationFilter(self):
    
        #Peak positions in nm of Hellma liquid HoDi UV45 filter with serial no. 0028 (calibrated on 1-3-2018)
        #5 JUNE 2018 TW: I've temporarily removed the peak at 575 nm because it's currently blocked by the baffled edge of the filter in the test set-up
        #                Because that wouldn't leave enough peaks for acurate spectral fitting, I've added the (uncalibrated) peaks at 416, 468 and 794 nm.
        peak_wls = np.array([416., 468., 794, 444.20, 640.60, 740.00, 864.00]) #np.array([444.20, 640.60, 740.00, 864.00])# #np.array([444.20, 575.00, 640.60, 740.00, 864.00]) #241.20, 353.80
        #peak_wls = np.array([241, 278, 287, 347, 354, 361, 416, 444, 468, 482, 512, 522, 537, 575, 641, 732, 740, 794, 864, 900, 950, 1000, 610.])
        #Only use the once that are larger than our lower wavelength limit
        self.hodi_peaks = peak_wls[peak_wls>self.lambda_min]

   
    def exit_angle(self, wl):
        #Applying the grating equation 
        return np.arcsin(self.m*wl/self.d-np.sin(self.theta))
        

    def theoretical_position_on_detector(self, wl):
        #returns the theoretically expected position on the sensor in terms of pixels
        return (np.tan(self.exit_angle(wl))*self.f - self.p0)/self.pxl + ((20-self.temperature)*1.3e3)/self.pxl
                        

    def find_nearest(self, array,value):
        #Finds the element in 'array' that is nearest to 'value'. 
        #This routine is used to relate the expected position of a wavelength peak to its actual position
        idx = (np.abs(array-value)).argmin()
        return array[idx]

    def fiber_number(self, mean_row):
        #Identify the fiber number by the position of its brightest row on the detector
        return 10-np.argmax(np.isclose(self.fiber_edge_rows, mean_row, atol=(self.fiber_spacing-1)))

    def separate_fibers(self, image, width=100):
        #We stack the image along the x-axis in bins with a chosen width, summing the pixel values along the x-axis in the bin for each y-position. 
        #For each bin, the peakutils package is used to find the peaks along the y-axis, with the assumption that each peak correspond to a fiber. 
        #The position of each peak is saved and used to determine the curvature of the spectrum of each fiber. 
        
        #The startposition on the x-axis (position may be adjusted according to the width of the spectrum)
        x0 = 0
        #The endposition on the x-axis (position may be adjusted according to the width of the spectrum)
        x1 = self.nx
        
        #Number of bins along the x-axis with width=width,         
        num_steps = int(np.ceil((x1-x0)/width))
        #Empty array to add the center of each bin in the x-direction 
        xcs = []
        #Empty array to add the positions of the peaks
        ypositions = []
        #Empty array to add the number of peaks found
        npeaks = []
        
        image_copy = image.copy()
        if self.lasers is True:
            #We set all values higher than two standard deviations above the mean to 1, and the rest to 0. 
            #This makes it easier to separate the fibers in case we use the lasers for alignment (and only have three thin spectral lines in our image)
            image_copy.setflags(write=1)
            image_copy[image_copy<(np.mean(image)+2.*np.std(image))] = 0
            image_copy[image_copy>(np.mean(image)+2.*np.std(image))] = 1
        
        
        for i in range(num_steps):

            #Sum the pixel values over the binwidth (along the x-axis) for each y-value
            y = np.sum(image_copy[:, (i*width+x0):((i+1)*width+x0)], axis=1).astype(float)
            
            #Determine the position of the peak in the cross section along the y-axis
            ypos = np.array( peakutils.indexes(y, thres=5, min_dist=80, thres_abs=True) )
            
            #The center (x-position) of the current bin
            xc = np.array( (i+0.5)*width + x0 )
            
            
            plt.plot(xc*np.ones_like(ypos), ypos, 'ro')

            if len(ypos)>0:

                ypositions.append(ypos)
                xcs.append(xc)
                npeaks.append(len(ypos))
            
		
        #Sometimes the routine finds less or more peaks than the number of fibers, but mostly it identifies the right amount of peaks.        
        #We therefore assume that the value for npeaks that occurs most frequently is the actual number of fibers
        
        
        print(npeaks)
        
        self.n_fibers = int(np.bincount(npeaks).argmax())
        print("Number of fibers = ", self.n_fibers)

	    #turn the lists into arrays that can be used by the polyfit routine        
        xcs = np.array(xcs)
        ypositions = np.array(ypositions)
        
        
        #If the number of peaks found corresponds to the number of fibers used,
        #then we use the positions to determine the curvature of each spectrum 
        indices_all_fibers_found = np.where(np.isclose(npeaks, self.n_fibers))
        
        xcs = xcs[indices_all_fibers_found]
        ypositions_temp = ypositions[indices_all_fibers_found]
        ypositions = np.vstack(ypositions[indices_all_fibers_found])
        
        
        for j, p in enumerate(xcs):
        
            plt.plot(p*np.ones_like(ypositions_temp[j]), ypositions_temp[j], 'bo')
        
        
        
        # Fit a low order polynomial to the positions
        polcoefs = np.polyfit(xcs, ypositions, 1)
		
        
		#Plot the found fits
        x = np.linspace(x0, x1, 101)
       
        plt.imshow(image_copy, origin='lower', cmap='gist_gray',vmin=0, vmax=2)
        for i in range(self.n_fibers):
            y = np.polyval(polcoefs[:, i], x)	
            plt.plot(x, y, 'C0')
        
        plt.xlim(0,self.nx)
        plt.ylim(0, self.ny)
        


        x = np.arange(self.nx)
        
        self.used_fibers = []
         
        for i in range(self.n_fibers):
        
            #Determine the 'central' row of this fiber by taking the mean y-value of the peak positions on the y-axis
            #This peak positions are determined from the polynomial curve that is fitted to each fiber
            mean_y_position_fiber = np.mean(np.polyval(polcoefs[:, i], x))

            fibernr = self.fiber_number(mean_y_position_fiber)
            
            self.used_fibers.append(fibernr)
            
            print(fibernr)

        plt.show()
        self.fiber_positions_polynomial_coefficients = polcoefs


    def convert_pixel_to_wavelength(self, fibernr, pixels, row):
        #'polynomialception'
        #Use the saved polynomial coefficients to calculate, as a function of fiber and rownumber, the 0th and 1st order polynomial coefficients 
        #that are necessary to convert pixelposition to wavelength for that row.   
            
        return np.polyval([np.polyval(self.spectral_fitting_polynomial_coefficients[fibernr]['1st order'], row),np.polyval(self.spectral_fitting_polynomial_coefficients[fibernr]['0th order'], row)], pixels) 


    def convert_wavelength_to_pixel(self, fibernr, wavelengths, row):
        #'polynomialception'
        #Use the saved polynomial coefficients to calculate, as a function of fiber and rownumber, the 0th and 1st order polynomial coefficients 
        #that are necessary to convert pixelposition to wavelength and viceversa for that row. 
        #This is the inverse function of self.convert_pixel_to_wavelength
                     
        return  (wavelengths - np.polyval(self.spectral_fitting_polynomial_coefficients[fibernr]['0th order'], row))/np.polyval(self.spectral_fitting_polynomial_coefficients[fibernr]['1st order'], row)




    def determine_peak_widths(self, fitted_wavelengths_all_fibers, peakwidthdictionary):

       
        for fibernr in fitted_wavelengths_all_fibers.keys():


            print("Fiber", fibernr)
            #Determine the positions of the laser peaks
            peak_positions = np.array(peakutils.indexes(fitted_wavelengths_all_fibers[fibernr]['intensities'], thres=0.1, min_dist=60))
            base = peakutils.baseline(fitted_wavelengths_all_fibers[fibernr]['intensities'], 7)

         
            
            for pp in peak_positions:
                #print("peak = ", fitted_wavelengths_all_fibers[fibernr]['wavelengths'][pp])
                
                laserwl = self.find_nearest(np.fromiter(peakwidthdictionary[fibernr].keys(), dtype=int), fitted_wavelengths_all_fibers[fibernr]['wavelengths'][pp])

                halfmaximumbaseline = np.max(fitted_wavelengths_all_fibers[fibernr]['intensities'][pp-20:pp+20])/2 - base[pp-20:pp+20]/2

              
                

                #We fit a cubic spline to each laser emission peak - (peak_max/2) and then use its roots, i.e. where it's 0, to determine the positions
                #that indicate the Full Width Half Maximum (https://stackoverflow.com/questions/10582795/finding-the-full-width-half-maximum-of-a-peak)
                spline = UnivariateSpline(fitted_wavelengths_all_fibers[fibernr]['wavelengths'][pp-20:pp+20],fitted_wavelengths_all_fibers[fibernr]['intensities'][pp-20:pp+20]-halfmaximumbaseline, s=0)
                #print("Spline roots = ", spline.roots())
                splineroots = spline.roots()
                
                d = np.mean(base[pp-20:pp+20])
                popt, pcov = curve_fit(lambda x, a, b, c: self.Gaussian(x, a, b, c, d), fitted_wavelengths_all_fibers[fibernr]['wavelengths'][pp-20:pp+20], fitted_wavelengths_all_fibers[fibernr]['intensities'][pp-20:pp+20]-base[pp-20:pp+20], p0=[np.max(fitted_wavelengths_all_fibers[fibernr]['intensities'][pp-20:pp+20]-base[pp-20:pp+20]), fitted_wavelengths_all_fibers[fibernr]['wavelengths'][pp], 1])
                                            
                #print("FWHM, from Gaussian = ",popt[2]*2*np.sqrt(2*np.log(2)), 2.355*popt[2], ", from spline is = ", splineroots[-1]-splineroots[0])
                
                
                
                
                plt.plot(fitted_wavelengths_all_fibers[fibernr]['wavelengths'][pp-20:pp+20], fitted_wavelengths_all_fibers[fibernr]['intensities'][pp-20:pp+20])
                plt.plot(fitted_wavelengths_all_fibers[fibernr]['wavelengths'][pp-20:pp+20], spline(fitted_wavelengths_all_fibers[fibernr]['wavelengths'][pp-20:pp+20])+halfmaximumbaseline, linestyle='dashed')
                
                plt.plot(fitted_wavelengths_all_fibers[fibernr]['wavelengths'][pp-20:pp+20], self.Gaussian(fitted_wavelengths_all_fibers[fibernr]['wavelengths'][pp-20:pp+20], *popt, d), linestyle='dotted')
                
                #plt.plot(fitted_wavelengths_all_fibers[fibernr]['wavelengths'][pp-20:pp+20], base[pp-20:pp+20], label='base')
                #plt.plot(fitted_wavelengths_all_fibers[fibernr]['wavelengths'][pp-20:pp+20], halfmaximumbaseline, label='base')

                #plt.vlines(fitted_wavelengths_all_fibers[fibernr]['wavelengths'][pp], 0, np.max(fitted_wavelengths_all_fibers[fibernr]['intensities'][pp-20:pp+20]), linestyles='dotted', color='white')  
                
                #plt.vlines(fitted_wavelengths_all_fibers[fibernr]['wavelengths'][pp], 0, np.max(fitted_wavelengths_all_fibers[fibernr]['intensities'][pp-20:pp+20]), linestyles='dotted')
                #plt.hlines(np.max(halfmaximumbaseline), splineroots[-1], splineroots[0], linestyles='dashed', color='white')
                #plt.legend()
                #plt.show()  
  



                


  
                
                peakwidthdictionary[fibernr][laserwl].append(splineroots[-1]-splineroots[0])
                




                
                



    def backwards_spectral_fitting(self, image, lower_margin=30, upper_margin=40, threshold=25, resolution=2, onthefly=False):
        #Main routine to derive the spectrum of a single fiber. 
        #First the fibers are separated on the sensor based on their stacked intensity along the y-axis.
        #Then for each fiber, the average position of the peak intensity on the y-axis is taken as the central row.
        #For each row between central row - lower_margin and central row + upper_margin, where the median pixelvalue is above the threshold value, 
        #this routine fits a polynomial to the positions of the 5 calibrated peaks from the HoDi filter. 
        #With this polynomial fit, the pixel x-positions can be mapped to wavelength and backwards. 
        #Finally, the polynomial fit is used to map a preset array with wavelenghts to the pixel in each row that corresponds to that wavelength.
        #The pixel intensities found this way are averaged for every wavelength in the preset array. 
    
    
        print(np.max(image))
    
        #The preset wavelength array, running from self.lambda_min to 1000 nm with a spacing set by the resolution argument
        wls_mapped_backwards = np.arange(self.lambda_min, 1000+resolution, resolution)

        #Create an empty dictionary to which we add the fitted wavelengths and corresponding intensities for each fiber
        fitted_wavelengths_all_fibers = {}

        #Pixel position array (later used to convert to wavelength)
        x = np.arange(self.nx)
  
        for i in range(self.n_fibers):

            #Empty lists to store the x- (column) and y-coordinates (row) to which the wavelengths in wls_mapped_backwards are mapped
            mapping_rows = []
            mapping_columns = []

        
            #Determine the 'central' row of this fiber by taking the mean y-value of the peak positions on the y-axis
            #This peak positions are determined from the polynomial curve that is fitted to each fiber
            mean_y_position_fiber = np.mean(np.polyval(self.fiber_positions_polynomial_coefficients[:, i], x))
            
            #The fibernumber is necessary to load the appropriate polynomial coefficients and as a key to store the fitted spectrum in the dictionary
            fibernr = self.fiber_number(mean_y_position_fiber)
            
            print("Fiber", fibernr)            
            
            #The median pixel values for every row that belongs to this fiber
            rowmedians = np.sum(image[(int(mean_y_position_fiber)-lower_margin):(int(mean_y_position_fiber)+upper_margin),:], axis=1)

            #Determine between which rows the median pixel values are larger than the threshold
            try:
                loweredge =min(np.nonzero(rowmedians > threshold)[0])
                upperedge = max(np.nonzero(rowmedians > threshold)[0])

            except ValueError as e:
                print(rowmedians)
                raise Exception('Threshold value is too low') from e

            #Create an array with the rows over which to loop                        
            rows = np.arange(np.int(mean_y_position_fiber-lower_margin+loweredge), np.int(mean_y_position_fiber-lower_margin+upperedge))
            #rows = [np.int(mean_y_position_fiber)]
            print(len(rows), "rows, from row", rows[0],"to row",rows[-1])
            
            #Plot the rows between which the spectra are interpolated and stacked
            '''
            plt.hlines(mean_y_position_fiber, 0, self.nx, color = 'c0', linestyles='dashed')
            plt.hlines(int(mean_y_position_fiber)-lower_margin, 0, self.nx, color='yellow', linestyles='dashed')
            plt.hlines(int(mean_y_position_fiber)+upper_margin, 0, self.nx, color='yellow', linestyles='dashed')
            plt.hlines(rows[0], 0, self.nx, color='red', linestyles='dashed')
            plt.hlines(rows[-1], 0, self.nx, color='red', linestyles='dashed')
            '''
            for r in rows:
                
                if onthefly:
                
                    #Since the HoDi filter gives an absorption spectrum we take 255 - pixelvalue for each pixel in this row
                    #to determine the position of calibrated wavelength peaks (i.e. we convert minima to maxima).
                    #The peakutils.indexes is an existing python routine and returns the indeces of values corresponding to
                    #a local maximum (The indeces conveniently correspond to pixel position along the x-axis). 
                    peak_positions = np.array(peakutils.indexes((255-image[r,:]), thres=0.25, min_dist=60))
                        
                    
                    '''
                    for p in peak_positions:
                    
                        plt.vlines(p, 0, self.ny, linestyles='dashed', color='black', alpha=0.5)
                        #plt.text(p, 100, str(i), style='italic', bbox={'facecolor':'white', 'alpha':0.5}, ha='center')
                    
                    #plt.plot(255-image_data[r,:])
                    plt.plot(image[r,:])
                    plt.plot(mapped_row, color='black', linestyle='dotted')
                    '''
                    #An empty array to which we add the subset of the 5 calibrated absorption lines from all absorption lines found  
                    hodi_peak_positions = []
                
                    #We loop over all HoDi wavelength peak position
                    for wl in self.hodi_peaks:
                        #We compare the position of the found peaks to the theoretically expected positions of the HoDi peaks.
                        #The found peaks that are nearest to the theoretically expected peaks are identified as the HoDi peaks.
                        peak = self.find_nearest(peak_positions, self.theoretical_position_on_detector(wl))
                        hodi_peak_positions.append(peak)
                        
                        
                        #plt.vlines(peak, 0, self.ny, linestyles='dotted', color='blue', alpha=0.5)
                        #plt.text(peak, 50, str(i), style='italic', bbox={'facecolor':'white', 'alpha':0.5}, ha='center')            

                    #Convert to a numpy array for faster and easier array operations
                    hodi_peak_positions = np.array(hodi_peak_positions)
                
              
                    #Fit a polynomial with order pol_order to the positions
                    #This polynomial determines how the pixel positions are converted to wavelengths for this row
                    polcoefs = np.polyfit(hodi_peak_positions, self.hodi_peaks, 1)
                    
                    
                    #Convert the preset wavelengths to pixel positions (column = x-positions, row = y-positions)
                    #Both lists are equally long and the n-th entry of each list together give an x,y-position       
                    mapping_columns.append((wls_mapped_backwards - polcoefs[1])/polcoefs[0])
                    mapping_rows.append(np.full(len(wls_mapped_backwards),r))


                else:
                 
                    #Convert the preset wavelengths to pixel positions (column = x-positions, row = y-positions) using the saved calibrated polynomials
                    #Both lists are equally long and the n-th entry of each list together give an x,y-position 
                    mapping_columns.append(self.convert_wavelength_to_pixel(fibernr, wls_mapped_backwards, r))
                    mapping_rows.append(np.full(len(wls_mapped_backwards),r))
                
    
            #We need to flatten the pixel coordinates such that we have a 1-dimensional array for each fiber
            mapping_columns = np.array(mapping_columns).flatten()
            mapping_rows = np.array(mapping_rows).flatten()
            
            #The ndimage.map_coordinates Scipy function maps the wavelength to its corresponding pixel position for all rows belonging to this fiber
            #this gives us 1 long array in which the intensities per row are lined up 
            interpolated_intensities = ndimage.map_coordinates(image, [mapping_rows, mapping_columns])
            
            #We need to take the mean of all elements that are spaced len(wls_mapped_backwards) apart, as those elements are the intensities 
            #belonging to the same wavelength in different rows
            averaged_intensities = np.mean(interpolated_intensities.reshape(-1, len(wls_mapped_backwards)), axis=0)
                       
            #Add the wavelengths and mapped, averaged intensities of this fiber to a dictionary for all fibers
            fitted_wavelengths_all_fibers[fibernr] = {'wavelengths':wls_mapped_backwards, 'intensities':averaged_intensities}    
                        
        
        return fitted_wavelengths_all_fibers



    def determine_polynomial_coefficients(self, image, lower_margin=30, upper_margin=40, threshold=25, resolution=2, pol_order=2):
        #Routine to derive to derive a polynomial that fits the polynomial coefficients of each row for the whole sensor at once.
        #NOT OPERATIONAL BECAUSE THE SMILE AND KEYSTONE ABERRATIONS ARE NOT CONTINUOUS! 
        #This routine follows the backwards_spectral_fitting routine
    
        #The preset wavelength array, running from self.lambda_min to 1000 nm with a spacing set by the resolution argument
        wls_mapped_backwards = np.arange(self.lambda_min, 1000+resolution, resolution)
    
        #Create an empty dictionary to which we add the fitted wavelengths and corresponding intensities for each fiber
        fitted_wavelengths_all_fibers = {}
    

        #Pixel position array (later used to convert to wavelength)
        x = np.arange(self.nx)
        
        coefficients_0thorder = []
        coefficients_1storder = []
        fitted_rows = []        

        for i in range(self.n_fibers):

            #Determine the 'central' row of this fiber by taking the mean y-value of the peak positions on the y-axis
            #This peak positions are determined from the polynomial curve that is fitted to each fiber
            mean_y_position_fiber = np.mean(np.polyval(self.fiber_positions_polynomial_coefficients[:, i], x))
                        
            #The median pixel values for every row that belongs to this fiber
            rowmedians = np.median(image[(int(mean_y_position_fiber)-lower_margin):(int(mean_y_position_fiber)+upper_margin),:], axis=1)

            #Determine between which rows the median pixel values are larger than the threshold
            try:
                loweredge =min(np.nonzero(rowmedians > threshold)[0])
                upperedge = max(np.nonzero(rowmedians > threshold)[0])

            except ValueError as e:
                print(rowmedians)
                raise Exception('Threshold value is too low') from e

            #Create an array with the rows over which to loop                        
            rows = np.arange(np.int(mean_y_position_fiber-lower_margin+loweredge), np.int(mean_y_position_fiber-lower_margin+upperedge))
            print(len(rows), "rows, from row", rows[0],"to row",rows[-1])
            #rows = [np.int(mean_y_position_fiber)]
            for r in rows:
                
                #Since the HoDi filter gives an absorption spectrum we take 255 - pixelvalue for each pixel in this row
                #to determine the position of calibrated wavelength peaks (i.e. we convert minima to maxima).
                #The peakutils.indexes is an existing python routine and returns the indeces of values corresponding to
                #a local maximum (The indeces conveniently correspond to pixel position along the x-axis). 
                peak_positions = np.array(peakutils.indexes((255-image[r,:]), thres=0.25, min_dist=60))
                
                    
                #An empty array to which we add the subset of the 5 calibrated absorption lines from all absorption lines found  
                hodi_peak_positions = []
            
                #We loop over all HoDi wavelength peak position
                for wl in self.hodi_peaks:
                    #We compare the position of the found peaks to the theoretically expected positions of the HoDi peaks.
                    #The found peaks that are nearest to the theoretically expected peaks are identified as the HoDi peaks.
                    peak = self.find_nearest(peak_positions, self.theoretical_position_on_detector(wl))
                    hodi_peak_positions.append(peak)
                              

                #Convert to a numpy array for faster and easier array operations
                hodi_peak_positions = np.array(hodi_peak_positions)
            
          
                #Fit a polynomial with order pol_order to the positions
                #This polynomial determines how the pixel positions are converted to wavelengths for this row
                polcoefs = np.polyfit(hodi_peak_positions, self.hodi_peaks, 1)
                
                #Add the polynomial coefficients of this row to the arrays for the whole image              
                coefficients_0thorder.append(polcoefs[1])
                coefficients_1storder.append(polcoefs[0])
                fitted_rows.append(r)         
                
         
            
        #Convert the lists into arrays that we can work with
        coefficients_0thorder = np.array(coefficients_0thorder)
        coefficients_1storder = np.array(coefficients_1storder)
        fitted_rows = np.array(fitted_rows)
        
        #Fit a polynomial to the 0th and 1st order coefficients of the whole sensor    
        self.polcoefs_0thorder = np.polyfit(fitted_rows, coefficients_0thorder, pol_order)          
        self.polcoefs_1storder = np.polyfit(fitted_rows, coefficients_1storder, pol_order) 


        plt.figure()
                
        f, axarr = plt.subplots(2, 1, sharex=True)
            
        axarr[0].scatter(fitted_rows, coefficients_1storder)
        axarr[0].plot(np.arange(1200), np.polyval(self.polcoefs_1storder, np.arange(1200)))
        axarr[0].set_title('1st order coeff')   

        axarr[1].scatter(fitted_rows, coefficients_0thorder)
        axarr[1].plot(np.arange(1200), np.polyval(self.polcoefs_0thorder, np.arange(1200)))
        axarr[1].set_title('0th order coeff')
             
        plt.show()     
        f.savefig(str(self.n_fibers)+'fitted_polynomial_coefficients.png')
        print("0th order coefficients ", self.polcoefs_0thorder)
        print("1st order coefficients ", self.polcoefs_1storder)
        #Save the polynomial coefficients for the whole sensor
        np.savetxt('polynomial_coefficients.ini', (self.polcoefs_0thorder,self.polcoefs_1storder)) 
       
       
       
       
    def determine_polynomial_coefficients_per_fiber(self, image, lower_margin=30, upper_margin=40, threshold=25, resolution=2, pol_order=2):
        #Routine to derive to derive a polynomial that fits the polynomial coefficients of each row for each fiber separately
        #This routine follows the backwards_spectral_fitting routine 
    
        #The preset wavelength array, running from self.lambda_min to 1000 nm with a spacing set by the resolution argument
        wls_mapped_backwards = np.arange(self.lambda_min, 1000+resolution, resolution)
    
        #Create an empty dictionary to which we add the fitted wavelengths and corresponding intensities for each fiber
        fitted_wavelengths_all_fibers = {}
    
        #Pixel position array (later used to convert to wavelength)
        x = np.arange(self.nx)
        
        
        poldict = {}

        for i in range(self.n_fibers):
        


            
            coefficients_0thorder = []
            coefficients_1storder = []
            fitted_rows = []                

            #Determine the 'central' row of this fiber by taking the mean y-value of the peak positions on the y-axis
            #This peak positions are determined from the polynomial curve that is fitted to each fiber
            mean_y_position_fiber = np.mean(np.polyval(self.fiber_positions_polynomial_coefficients[:, i], x))
            fibernr = self.fiber_number(mean_y_position_fiber)
            poldict[fibernr] = {}         
            #The median pixel values for every row that belongs to this fiber
            rowmedians = np.median(image[(int(mean_y_position_fiber)-lower_margin):(int(mean_y_position_fiber)+upper_margin),:], axis=1)

            #Determine between which rows the median pixel values are larger than the threshold
            try:
                loweredge =min(np.nonzero(rowmedians > threshold)[0])
                upperedge = max(np.nonzero(rowmedians > threshold)[0])

            except ValueError as e:
                print(rowmedians)
                raise Exception('Threshold value is too low') from e

            #Create an array with the rows over which to loop                        
            rows = np.arange(np.int(mean_y_position_fiber-lower_margin+loweredge), np.int(mean_y_position_fiber-lower_margin+upperedge))
            print(len(rows), "rows, from row", rows[0],"to row",rows[-1])
            #rows = [np.int(mean_y_position_fiber)]

            for r in rows:
                
                #Since the HoDi filter gives an absorption spectrum we take 255 - pixelvalue for each pixel in this row
                #to determine the position of calibrated wavelength peaks (i.e. we convert minima to maxima).
                #The peakutils.indexes is an existing python routine and returns the indeces of values corresponding to
                #a local maximum (The indeces conveniently correspond to pixel position along the x-axis). 
                peak_positions = np.array(peakutils.indexes((255-image[r,:]), thres=0.25, min_dist=60))
                
                    
                #An empty array to which we add the subset of the 5 calibrated absorption lines from all absorption lines found  
                hodi_peak_positions = []
            
                #We loop over all HoDi wavelength peak position
                for wl in self.hodi_peaks:
                    #We compare the position of the found peaks to the theoretically expected positions of the HoDi peaks.
                    #The found peaks that are nearest to the theoretically expected peaks are identified as the HoDi peaks.
                    peak = self.find_nearest(peak_positions, self.theoretical_position_on_detector(wl))
                    hodi_peak_positions.append(peak)
                              

                #Convert to a numpy array for faster and easier array operations
                hodi_peak_positions = np.array(hodi_peak_positions)
            
          
                #Fit a polynomial with order pol_order to the positions
                #This polynomial determines how the pixel positions are converted to wavelengths for this row
                polcoefs = np.polyfit(hodi_peak_positions, self.hodi_peaks, 1)
                
                #Add the polynomial coefficients of this row to the arrays for this fiber              
                coefficients_0thorder.append(polcoefs[1])
                coefficients_1storder.append(polcoefs[0])
                fitted_rows.append(r)         
                
         
            
            
            #Convert the lists into arrays that we can work with
            coefficients_0thorder = np.array(coefficients_0thorder)
            coefficients_1storder = np.array(coefficients_1storder)
            fitted_rows = np.array(fitted_rows)
            
            #Fit a polynomial to the 0th and 1st order coefficients of this fiber                 
            polcoefs_0thorder = np.polyfit(fitted_rows, coefficients_0thorder, pol_order)          
            polcoefs_1storder = np.polyfit(fitted_rows, coefficients_1storder, pol_order) 
    
    
    
            poldict[fibernr]['0th order coefficients'] = coefficients_0thorder
            poldict[fibernr]['1st order coefficients'] = coefficients_1storder
            poldict[fibernr]['fitted rows'] = fitted_rows
            
            '''
            plt.figure()
                    
            f, axarr = plt.subplots(2, 1, sharex=True)
            f.suptitle('Fiber '+str(fibernr))    
            axarr[0].scatter(fitted_rows, coefficients_1storder)
            axarr[0].plot(np.arange(1200), np.polyval(polcoefs_1storder, np.arange(1200)))
            axarr[0].set_title('1st order coeff')   
    
            axarr[1].scatter(fitted_rows, coefficients_0thorder)
            axarr[1].plot(np.arange(1200), np.polyval(polcoefs_0thorder, np.arange(1200)))
            axarr[1].set_title('0th order coeff')
                 
            plt.show()     
            f.savefig('fitted_polynomial_coefficients_fiber_'+str(fibernr)+'.png')
            print("0th order coefficients ", polcoefs_0thorder)
            print("1st order coefficients ", polcoefs_1storder)
            '''

            #Save the polynomial coefficients for this fiber
            #np.savetxt("polynomial_coefficients_fiber"+str(fibernr)+".ini", (polcoefs_0thorder,polcoefs_1storder)) 
        return poldict            




    def pixelpositions_of_peaks(self, image, lower_margin=30, upper_margin=40, threshold=30):
        #Based on the spectral fitting routines, this routine returns the positions of the calibration peaks for each row. 
        #This routine can be used to determine the shift of the spectrum on the detector with changing temperature.
    

        #Remove peaks part of a 'double' peak as they can cause confusion with wavelength shift as a function of temperature 
        excluded_peaks = []#[444.2, 794.]
        sorter = np.argsort(self.hodi_peaks)
        indices_excluded_peaks = sorter[np.searchsorted(self.hodi_peaks, excluded_peaks, sorter=sorter)]       
        

        #Create an empty dictionary to which we add the fitted wavelengths and corresponding intensities for each fiber
        peak_positions_all_fibers = {}
    

        x = np.arange(self.nx)
  
        for i in range(self.n_fibers):
            

            mean_y_position_fiber = np.mean(np.polyval(self.fiber_positions_polynomial_coefficients[:, i], x))
            
            fibernr = self.fiber_number(mean_y_position_fiber)
            
            print("Fiber", fibernr)            
            
            #Create a nested dictionary in which we save the pixelpositions of the peaks and the corresponding wavelengths
            peak_positions_all_fibers[fibernr] = {}
            peak_positions_all_fibers[fibernr]['positions'] = []
            peak_positions_all_fibers[fibernr]['wavelengths'] = np.delete(self.hodi_peaks, indices_excluded_peaks).copy()

            
            #The median pixel values for every row that belongs to this fiber
            rowmedians = np.median(image[(int(mean_y_position_fiber)-lower_margin):(int(mean_y_position_fiber)+upper_margin),:], axis=1)
            #Determine between which rows the median pixel values are larger than the threshold          
            try:
                loweredge =min(np.nonzero(rowmedians > threshold)[0])
                upperedge = max(np.nonzero(rowmedians > threshold)[0])

            except ValueError as e:
                print(rowmedians)
                raise Exception('Threshold value is too low') from e

            #Create an array with the rows over which to loop                        
            rows = np.arange(np.int(mean_y_position_fiber-lower_margin+loweredge), np.int(mean_y_position_fiber-lower_margin+upperedge))
            #rows = [np.int(mean_y_position_fiber)]
            print(len(rows), "rows, from row", rows[0],"to row",rows[-1])
            
            for r in rows:
                           
                #Since the HoDi filter gives an absorption spectrum we take 255 - pixelvalue for each pixel in this row
                #to determine the position of calibrated wavelength peaks (i.e. we convert minima to maxima).
                #The peakutils.indexes is an existing python routine and returns the indeces of values corresponding to
                #a local maximum (The indeces conveniently correspond to pixel position along the x-axis). 
                peak_positions = np.array(peakutils.indexes((255-image[r,:]), thres=0.25, min_dist=60))
                '''  
                if fibernr == 10:     
                    for p in peak_positions:
                        plt.vlines(p, 0, self.ny, linestyles='dashed', color='black', alpha=0.5)
                        #plt.text(p, 100, str(i), style='italic', bbox={'facecolor':'white', 'alpha':0.5}, ha='center')
                    
                    #plt.plot(255-image_data[r,:])
                    plt.plot(image[r,:])
                '''
                    
                #An empty array to which we add the subset of the 5 calibrated absorption lines from all absorption lines found  
                hodi_peak_positions = []
            
                #We loop over all HoDi wavelength peak position
                for wl in self.hodi_peaks:
                    #We compare the position of the found peaks to the theoretically expected positions of the HoDi peaks.
                    #The found peaks that are nearest to the theoretically expected peaks are identified as the HoDi peaks.
                    #Remove peaks part of a 'double' peak as they can cause confusion with wavelength shift as a function of temperature 
                    if wl not in excluded_peaks:
                        peak = self.find_nearest(peak_positions, self.theoretical_position_on_detector(wl))
                        hodi_peak_positions.append(peak)
                        
                        '''
                        if fibernr == 10: 
                            plt.vlines(peak, 0, self.ny, linestyles='dotted', color=self.wavelength_to_rgb(wl), alpha=0.5)
                            plt.text(peak, 50, str(wl), style='italic', bbox={'facecolor':'white', 'alpha':0.5}, ha='center')
                        '''       
                '''
                if fibernr == 10: 
                    plt.show()            
                '''
                peak_positions_all_fibers[fibernr]['positions'].append(hodi_peak_positions)
                

            peak_positions_all_fibers[fibernr]['positions'] = np.array(peak_positions_all_fibers[fibernr]['positions'])

                        
        
        return peak_positions_all_fibers


        

    def sum_rows(self, image):
    
        return np.sum(image, axis=1)
        
    def sum_columns(self, image):
    
        return np.sum(image, axis=0)
        
    def connect_camera(self):
     
        available_cameras = pypylon.factory.find_devices()
  
        if len(available_cameras) != 1:
            if len(available_cameras)==0:
                sys.exit("Error: no camera found")
            else:
                while True:
                    print('Available cameras are:')
                    for i, c in enumerate(available_cameras):
                        print(i, c)
                    try:
                        choice = int(input("Choose camera number: "))
                    except ValueError:
                        print("Please choose a number")
                        continue
                
                    try:
                        cam = available_cameras[choice]
    
                        break
                    except IndexError:
                        print("Please choose a number that is on the list")
                        continue            
                
        else:
            cam = available_cameras[0]  
  
  
        print("Using", cam)
  
        
        # Create a camera instance
        print("Creating camera instance")        
        self.cam = pypylon.factory.create_device(cam)       
        # Open camera and grep some images
        print("Open camera")
        self.cam.open()
       
        self.cam.properties['ExposureTime'] = 100000.#100000.0#
        self.cam.properties['Gain'] = 5.0
        self.cam.properties['PixelFormat'] = 'Mono12'
        print("Set camera properties")

    def camera_properties(self):
        # We can still get information of the camera back
        print('Camera info of camera object:', self.cam.device_info)
        
        for key in self.cam.properties.keys():
            try:
                value = self.cam.properties[key]
            except IOError:
                value = '<NOT READABLE>'
        
            print('{0} ({1}):\t{2}'.format(key, self.cam.properties.get_description(key), value))
        

    def take_snapshots(self, N):
    
        return self.cam.grab_images(N)
    


    def wavelength_to_rgb(self, wavelength, gamma=0.5):
    
        '''This converts a given wavelength of light to an 
        approximate RGB color value. The wavelength must be given
        in nanometers in the range from 380 nm through 750 nm
        (789 THz through 400 THz).
    
        Based on code by Dan Bruton
        http://www.physics.sfasu.edu/astro/color/spectra.html
        '''
    
        wavelength = float(wavelength)
        if wavelength < 380:
            R = 0.0
            G = 0.0
            B = 1.0    
        elif wavelength >= 380 and wavelength <= 440:
            attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
            R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
            G = 0.0
            B = (1.0 * attenuation) ** gamma
        elif wavelength >= 440 and wavelength <= 490:
            R = 0.0
            G = ((wavelength - 440) / (490 - 440)) ** gamma
            B = 1.0
        elif wavelength >= 490 and wavelength <= 510:
            R = 0.0
            G = 1.0
            B = (-(wavelength - 510) / (510 - 490)) ** gamma
        elif wavelength >= 510 and wavelength <= 580:
            R = ((wavelength - 510) / (580 - 510)) ** gamma
            G = 1.0
            B = 0.0
        elif wavelength >= 580 and wavelength <= 645:
            R = 1.0
            G = (-(wavelength - 645) / (645 - 580)) ** gamma
            B = 0.0
        elif wavelength >= 645 and wavelength <= 750:
            attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
            R = (1.0 * attenuation) ** gamma
            G = 0.0
            B = 0.0
        else:
            R = 1.0
            G = 0.0
            B = 0.0
        R *= 255
        G *= 255
        B *= 255
        return '#%02x%02x%02x' % (int(R), int(G), int(B))

    def plot_wavelength(self, wl):
  
        wl_x = self.theoretical_position_on_detector(wl)
        plt.vlines(wl_x, 0, self.ny, linestyles='dashed', color=self.wavelength_to_rgb(wl), alpha=0.33)
        plt.text(wl_x, 1000, str(wl), style='italic', bbox={'facecolor':'white', 'alpha':0.5}, ha='center')    
        



if __name__ in ("__main__","__plot__"):
    
    #fileDir = os.path.dirname(os.path.realpath('__file__'))
    
    parser = argparse.ArgumentParser(description='Choose in which mode you want to use the script')

    parser.add_argument("--live", 
                        dest="live",
                        action='store_true', 
                        default = False,
                        help="Captures a live feed from the camera, instead of reading a saved image")
    parser.add_argument("--file", 
                      dest="file", 
                      default = '7fibers_3lasers_250000us_gain_NE10filter.tiff',
                      help="The captured image that you want to plot")
    parser.add_argument("-n", 
                      dest="n_fibers", 
                      default = 10,
                      type=int,
                      help="The number of fibers on the image")


    args = parser.parse_args()


    def fitfunc(x, b):
      
      return -0.04*x + b







    if args.live:
        #if one chooses to run this script 'live', then pypylon must be installed and can be imported.
        #To use this script with saved images, pypylon is not necessarily installed 
        import pypylon
        image_data = imread('ClimateChamberT20_pos2500.tiff').astype(np.float32)    
        plt.style.use('dark_background')
        analyze = ASpec(image=image_data, lasers=True)
        #analyze.camera_properties()
        peakwidths = {}
        
        while True:                
            try:
                temp = int(input("Give the temperature: "))
                
                peakwidths[temp] = {}
                
                for f in analyze.used_fibers:
                    print(f)
                    peakwidths[temp]['positions'] = []
                    peakwidths[temp][f] = {402:[], 637:[], 856:[]}
                
                break
            except ValueError:
                
                print("Please choose a number")
                continue
            
        try:
        
            while True:
            
                
                inp = input("Give the position (in um), type 'p' for plot, or any other non-numeric key to exit: ")
                
                
                if inp == 'p':
    
                    f, axarr = plt.subplots(3, sharex=True, figsize=(8,4))
                    f2, axarr2 = plt.subplots(len(analyze.used_fibers), sharex=True, figsize=(10,len(analyze.used_fibers)*3))
                    
                    f.suptitle(r"T = "+str(temp)+"$^\circ$C")
                    f2.suptitle(r"T = "+str(temp)+"$^\circ$C")
                    
                    axarr[0].set_title('420 nm')
                    axarr[1].set_title('637 nm')
                    axarr[2].set_title('856 nm')
            
                            
                    for fiber_index, fibernr in enumerate(sorted(analyze.used_fibers)):
                    
                        axarr2[fiber_index].set_title('Fiber '+str(fibernr))
                        
                        for wl in sorted(peakwidths[temp][fibernr].keys()):
            
                            if wl == 402:
                                pi = 0
                            elif wl == 637:
                                pi = 1
                            elif wl == 856:
                                pi = 2
            
            
                            #We sort the position array in case we have done additional measurements, for example around a found minimum.
                            pos_sorting_indices = np.argsort(peakwidths[temp]['positions'])
                            
                            sorted_positions = np.array(peakwidths[temp]['positions'])[pos_sorting_indices]
                            sorted_wavelengths = np.array(peakwidths[temp][fibernr][wl])[pos_sorting_indices]
                            
                            axarr[pi].plot(sorted_positions, sorted_wavelengths, label = "Fiber "+str(fibernr), color=colors[fibernr-1])
                            axarr2[fiber_index].plot(sorted_positions, sorted_wavelengths, label = str(wl)+' nm', color=colors[pi**2+2*pi])
                            
            
            
                    axarr[0].legend(loc='best', fancybox=True, framealpha=0.5, fontsize=8)
                    axarr2[0].legend(loc='best', fancybox=True, framealpha=0.5, fontsize=8)
                    
                    axarr[1].set_ylabel(r'$\mathrm{peakwidth\, [nm]}$')
                    axarr2[int(len(analyze.used_fibers)/2)].set_ylabel(r'$\mathrm{peakwidth\, [nm]}$')
                        
                    axarr[2].set_xlabel(r'$\mathrm{translation\, [um]}$')
                    
                    plt.show()
                    
                    continue                
                
                
                
                translation = float(inp)
                
                peakwidths[temp]['positions'].append(translation)
                
                for image_data in analyze.take_snapshots(1):
                    imageio.imsave('../Basler spectra/Lasers/Climate chamber/16bit/0-40/'+str(temp)+'C/ClimateChamberT'+str(temp)+'_pos'+str(inp)+'.tiff', image_data*16)#.astype(np.uint16)        
                    wl_int_dict = analyze.backwards_spectral_fitting(image_data, resolution=1)
                    analyze.determine_peak_widths(wl_int_dict, peakwidths[temp])
                    #filename = os.path.join(fileDir, '..\\"Basler spectra"\\"Climate chamber"\\ClimateChamberT'+str(temp)+'_pos'+str(inp)+'.tiff')
                    #imsave(filename, image_data)
                    
                    
                    
                    
                    
                    
                pickle.dump(peakwidths, open( 'laser_peak_widths_T'+str(temp)+'.p', "wb" ) )
            
            
            
            
        except ValueError as e:
            print(e)
            sys.exit("That's it, I'm done")




                    
    elif args.file:
    
    
        #Read the image from file and make sure that the pixels and their corresponding values are floats
        print("Reading in image...")


        image_data = imread(args.file).astype(np.float32)
        
        analyze = ASpec(image=image_data, lasers=True)
        
        temp = 20
        peakwidths = {temp:{}}

        peakwidths[temp]['positions'] = []

        peakwidths[temp]['positions'].append(2)


        
        #peakwidths[temp] = {}
                
        for f in analyze.used_fibers:
            print(f)
            peakwidths[temp][f] = {402:[], 637:[], 856:[]}
            
        wl_dict = analyze.backwards_spectral_fitting(image_data, resolution=1)
        
        print(peakwidths[temp])
        
        analyze.determine_peak_widths(wl_dict, peakwidths[temp])#, resolution=1)#, onthefly=True, threshold=15)
        
        
        print("----------")
        print(peakwidths[temp])

        f, axarr = plt.subplots(3, sharex=True, figsize=(10,6))
        f2, axarr2 = plt.subplots(len(analyze.used_fibers), sharex=True, figsize=(10,len(analyze.used_fibers)*3))
        
        f.suptitle(r"T = "+str(temp)+"$^\circ$C")
        f2.suptitle(r"T = "+str(temp)+"$^\circ$C")
        
        axarr[0].set_title('420 nm')
        axarr[1].set_title('637 nm')
        axarr[2].set_title('856 nm')

                
        for fiber_index, fibernr in enumerate(sorted(analyze.used_fibers)):
        
            axarr2[fiber_index].set_title('Fiber '+str(fibernr))
            
            for wl in sorted(peakwidths[temp][fibernr].keys()):

                if wl == 402:
                    pi = 0
                elif wl == 637:
                    pi = 1
                elif wl == 856:
                    pi = 2

                axarr[pi].scatter(peakwidths[temp]['positions'], peakwidths[temp][fibernr][wl], label = "Fiber "+str(fibernr), color=colors[fibernr-1])
                axarr2[fiber_index].scatter(peakwidths[temp]['positions'], peakwidths[temp][fibernr][wl], label = str(wl)+' nm', color=colors[pi**2+2*pi])
                


        axarr[0].legend(loc='best', fancybox=True, framealpha=0.5, fontsize=8)
        axarr2[0].legend(loc='best', fancybox=True, framealpha=0.5, fontsize=8)
        
        axarr[1].set_ylabel(r'$\mathrm{peakwidth\, [nm]}$')
        axarr2[int(len(analyze.used_fibers)/2)].set_ylabel(r'$\mathrm{peakwidth\, [nm]}$')
            
        axarr[2].set_xlabel(r'$\mathrm{translation\, [mm]}$')
        
        plt.show()
                
                      
                    