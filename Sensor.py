import numpy as np
import glob
import matplotlib.pyplot as plt
#import matplotlib.ticker as mtick
from scipy.misc import imread, imsave
from scipy import ndimage
import pypylon
import sys
#from scipy.signal import argrelextrema, savgol_filter
import argparse
import peakutils
import time
import astropy.stats as aps
from scipy.optimize import curve_fit


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
        if image is None:
            self.connect_camera()
            for image_data in self.take_snapshots(1):
                image = image_data

        #These values are used to identify each fiber based on its position on the detector
        #We assume that each fibers covers an equal number of rows specified by self.fiber_spacing.
        #The edge between two adjacent fibers is emperically determined.  
        self.fiber_spacing = 110 #rows
        self.fiber_edge_rows = np.arange(1150, 49, -self.fiber_spacing)

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
        if self.lasers == True:
            #We set all values higher than two standard deviations above the mean to 1, and the rest to 0. 
            #This makes it easier to separate the fibers in case we use the lasers for alignment (and only have three thin spectral lines in our image)
            image_copy.setflags(write=1)
            image_copy[image_copy<(np.mean(image)+2.*np.std(image))] = 0
            image_copy[image_copy>(np.mean(image)+2.*np.std(image))] = 1
        
        
        for i in range(num_steps):

            #Sum the pixel values over the binwidth (along the x-axis) for each y-value
            y = np.sum(image_copy[:, (i*width+x0):((i+1)*width+x0)], axis=1).astype(float)
            
            #Determine the position of the peak in the cross section along the y-axis
            ypos = np.array( peakutils.indexes(y, thres=0.1, min_dist=80, thres_abs=False) )
            
            #The center (x-position) of the current bin
            xc = np.array( (i+0.5)*width + x0 )
            
            
            plt.plot(xc*np.ones_like(ypos), ypos, 'ro')

            if len(ypos)>0:

                ypositions.append(ypos)
                xcs.append(xc)
                npeaks.append(len(ypos))
            
		
        #Sometimes the routine finds less or more peaks than the number of fibers, but mostly it identifies the right amount of peaks.        
        #We therefore assume that the value for npeaks that occurs most frequently is the actual number of fibers
        
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


    def spectral_fitting(self, image, pol_order = 1, lower_margin=30, upper_margin=40, threshold=25, binwidth=2, onthefly=False):
        #Alternative routine to derive the spectrum of a single fiber. The main routine is backwards_spectral_fitting
        #The pixel positions are converted to wavelengths and consequently all the rows belonging to a single fiber are stacked. 
        #First the fibers are separated on the sensor based on their stacked intensity along the y-axis with a separate routine.
        #Then for each fiber, the average position of the peak intensity along the y-axis is taken as the central row.
        #For each row between central row - lower_margin and central row + upper_margin, where the median pixelvalue is above the threshold value, 
        #this routine fits a polynomial to the positions of the calibrated peaks from the HoDi filter. 
        #The polynomial fit can be done either 'on the fly' or the saved and calibrated polynomial coefficients are used.
        #With this polynomial fit, the x-positions can be converted from pixel position to wavelength. 
        #Finally, the wavelengths (and their corresponding intensities) are binned in wavelength bins with width = binwidth nanometer. 
    
        #Create an empty dictionary to which we add the fitted wavelengths and corresponding intensities for each fiber
        fitted_wavelengths_all_fibers = {}
            
        #Pixel position array (later used to convert to wavelength)
        x = np.arange(self.nx)
  
        print("Polynomial order = ", pol_order)
        
        for i in range(self.n_fibers):

            #Empty array to store the derived wavelengths of all rows belonging to this fiber
            unsorted_wavelengths = []
            #Empty array to store the intensities of all rows belonging to this fiber
            unsorted_intensities = []
        
            #Determine the 'central' row of this fiber by taking the mean y-value of the peak positions on the y-axis
            #This peak positions are determined from the polynomial curve that is fitted to each fiber
            mean_y_position_fiber = np.mean(np.polyval(self.fiber_positions_polynomial_coefficients[:, i], x))

            fibernr = self.fiber_number(mean_y_position_fiber)
            
            print("Fiber", fibernr)
            print("Fitting spectrum on the fly = ", onthefly)    
            
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
            
            
                if onthefly:
                
                    #Since the HoDi filter gives an absorption spectrum we take 255 - pixelvalue for each pixel in this row
                    #to determine the position of calibrated wavelength peaks (i.e. we convert minima to maxima).
                    #The peakutils.indexes is an existing python routine and returns the indeces of values corresponding to
                    #a local maximum (The indeces conveniently correspond to pixel position along the x-axis). 
                    peak_positions = np.array(peakutils.indexes((255-image[r,:]), thres=0.25, min_dist=60))
                    
                    '''
                    for p in peak_positions:
                    
                        plt.vlines(p, 0, self.ny, linestyles='dashed', color='white', alpha=0.5)
                        plt.text(p, 100, str(i), style='italic', bbox={'facecolor':'white', 'alpha':0.5}, ha='center')
                    '''
                    #plt.plot(255-image_data[r,:])
                        
                    #An empty array to which we add the subset of the 5 calibrated absorption lines from all absorption lines found  
                    hodi_peak_positions = []
                
                    #We loop over all HoDi wavelength peak position

                    for wl in self.hodi_peaks:
                        #We compare the position of the found peaks to the theoretically expected positions of the HoDi peaks.
                        #The found peaks that are nearest to the theoretically expected peaks are identified as the HoDi peaks.
                        peak = self.find_nearest(peak_positions, self.theoretical_position_on_detector(wl))
                        hodi_peak_positions.append(peak)
                        
                        
                        #plt.vlines(peak, 0, self.ny, linestyles='dashed', color='blue', alpha=0.5)
                        #plt.text(peak, 50, str(i), style='italic', bbox={'facecolor':'white', 'alpha':0.5}, ha='center')            

                    #print("Locating HoDi peaks took", t7-t7, "secs")
                    #Convert to a numpy array for faster and easier array operations
                    hodi_peak_positions = np.array(hodi_peak_positions)
                             
                    #Fit a polynomial with order pol_order to the positions
                    #This polynomial determines how the pixel positions are converted to wavelengths for this row
                    polcoefs = np.polyfit(hodi_peak_positions, self.hodi_peaks, pol_order)
                    
                              
                    #Add the fitted wavelength positions of this row to the wavelengths that have been fitted for previous rows belonging to this fiber.    
                    unsorted_wavelengths.append(np.polyval(polcoefs, x))
    
                    #Add the pixel intensities of this row accordingly
                    unsorted_intensities.append(image_data[r,:])
                    
                    '''
                    plt.figure()
                    plt.plot(np.polyval(polcoefs, x), image_data[r,:])
                    for j, xpos in enumerate(hodi_peak_positions):
                        wl = np.polyval(polcoefs, xpos)
                        plt.vlines(wl, 0, 300, linestyles='dashed', color=self.wavelength_to_rgb(wl), alpha=0.33)
                        plt.text(wl, 300, str('%.2f' % wl), style='italic', bbox={'facecolor':'white', 'alpha':0.5}, ha='center')
                        plt.text(wl, 250, str('%.2f' % self.hodi_peaks[j]), style='italic', bbox={'facecolor':'white', 'alpha':0.5}, ha='center') 
                    
                    plt.savefig('spectrum_fiber_'+str(i)+'_'+str(pol_order)+'_orders'+'.png')
                    plt.close()
                    '''
                else:
                              
                    #Add the fitted wavelength positions of this row to the wavelengths that have been fitted for previous rows belonging to this fiber,
                    #using the saved calibrated polynomials    
                    unsorted_wavelengths.append(self.convert_pixel_to_wavelength(fibernr, x, r))
    
                    #Add the pixel intensities of this row accordingly
                    unsorted_intensities.append(image_data[r,:])
                    

            
            #We need to flatten the unsorted wavelengths (and intensities) such that we have a 1-dimensional array of length self.nx x #rows, 
            #instead of a matrix with dimensions self.nx x #rows

            unsorted_wavelengths = np.array(unsorted_wavelengths).flatten()
            unsorted_intensities = np.array(unsorted_intensities).flatten()
            
            #Create wavelength bins, ranging from the minimum to the maximam found wavelength
            bins = np.arange(int(round(min(unsorted_wavelengths)))-binwidth/2, int(round(max(unsorted_wavelengths)))+1-binwidth/2, binwidth)       
            
            #The numpy digitize routine bins the unsorted wavelengths according to the provided bins.
            #It returns the indices that map each wavelength to its corresponding bin
            binned_inds = np.digitize(unsorted_wavelengths, bins)
            
            #The indices given by the digitize routine are used to map the intensties to their corresponding bin
            #and take the mean value of that bin
            bin_means = np.array([unsorted_intensities[binned_inds == idb].mean() for idb in range(1, len(bins))])
                        
            #Remove the last element of the bins (such that the bins array has the same length as the bin_means array),
            #and shift by binwidth/2 such that the wavelengths correspond to the middle of the bin            
            binned_wavelengths = bins[:-1]+binwidth/2
            
            #Add the binned wavelengths and intensities of this fiber to a dictionary for all fibers
            fitted_wavelengths_all_fibers[fibernr] = {'wavelengths':binned_wavelengths, 'intensities':bin_means}

            '''
            for pwl in [347, 354, 361, 416, 444, 468, 482, 512, 522, 537, 575, 641, 732, 740, 794, 864]:
                plt.vlines(pwl, 0, 300, linestyles='dashed', color=self.wavelength_to_rgb(pwl), alpha=0.33)
                plt.text(pwl, 250, str(pwl), style='italic', bbox={'facecolor':'white', 'alpha':0.5}, ha='center') 
                
            '''
            
            '''
            for j, xpos in enumerate(hodi_peak_positions):
                wl = np.polyval(polcoefs, xpos)
                plt.vlines(wl, 0, 300, linestyles='dashed', color=self.wavelength_to_rgb(wl), alpha=0.33)
                plt.text(wl, 300, str('%.2f' % wl), style='italic', bbox={'facecolor':'white', 'alpha':0.5}, ha='center')
                plt.text(wl, 250, str('%.2f' % self.hodi_peaks[j]), style='italic', bbox={'facecolor':'white', 'alpha':0.5}, ha='center') 
            '''
            
            #plt.savefig('unfolded_spectrum_fiber_'+str(i)+'_'+str(pol_order)+'_orders'+'.png')
            #plt.close()

        #plt.show()
        
        return fitted_wavelengths_all_fibers  


    def backwards_spectral_fitting(self, image, lower_margin=30, upper_margin=40, threshold=25, resolution=2, onthefly=False):
        #Main routine to derive the spectrum of a single fiber. 
        #First the fibers are separated on the sensor based on their stacked intensity along the y-axis.
        #Then for each fiber, the average position of the peak intensity on the y-axis is taken as the central row.
        #For each row between central row - lower_margin and central row + upper_margin, where the median pixelvalue is above the threshold value, 
        #this routine fits a polynomial to the positions of the 5 calibrated peaks from the HoDi filter. 
        #With this polynomial fit, the pixel x-positions can be mapped to wavelength and backwards. 
        #Finally, the polynomial fit is used to map a preset array with wavelenghts to the pixel in each row that corresponds to that wavelength.
        #The pixel intensities found this way are averaged for every wavelength in the preset array. 
    
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
        self.cam = pypylon.factory.create_device(cam)       
        # Open camera and grep some images
        self.cam.open()
        
        self.cam.properties['ExposureTime'] = 100000.#100000.0#
        self.cam.properties['Gain'] = 5.0
        self.cam.properties['PixelFormat'] = 'Mono12'
        


    def camera_properties(self):
        # We can still get information of the camera back
        print('Camera info of camera object:', self.cam.device_info)
        '''
        for key in self.cam.properties.keys():
            try:
                value = self.cam.properties[key]
            except IOError:
                value = '<NOT READABLE>'
        
            print('{0} ({1}):\t{2}'.format(key, self.cam.properties.get_description(key), value))
        '''

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
    
    
    parser = argparse.ArgumentParser(description='Choose in which mode you want to use the script')

    parser.add_argument("--live", 
                        dest="live",
                        action='store_true', 
                        default = False,
                        help="Captures a live feed from the camera, instead of reading a saved image")
    parser.add_argument("--file", 
                      dest="file", 
                      default = 'Fiberguide_stacked_HoDi_absorption_7fibers.tiff',
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
        plt.style.use('dark_background')
        analyze = ASpec()        
        wavelengths = [444.20, 575.00, 610., 640.60, 740.00, 864.00]#[401.7, 610., 636.7 , 855.7]#
        
        for image_data in analyze.take_snapshots(1):
        
        
            #Select the row with the largest total photon flux
            row = np.argmax(analyze.sum_rows(image_data))
            print(row)
            #Save the image and cross section as an instance that can be overwritten in each frame
            h = plt.imshow(image_data, origin='lower', cmap='gist_gray',vmin=0, vmax=255)
            line, = plt.plot(image_data[row,:])
            #Plot the relevant wavelengths to align the spectrograph
            for wl in wavelengths:#analyze.hodi_peaks:##
                analyze.plot_wavelength(wl)
            #Indicate the row from which the spectrum is shown
            plt.hlines(row, 0, 1920, linestyles='dotted', color='blue', alpha=0.33)
            plt.xlim(0,1920) 
            plt.ylim(0,1200)
            plt.ion()
            try:
                while True:
                    #Continously take new snapshots to create a live image feed
                    for new_image in analyze.take_snapshots(1):
                        #Update the data stored in the previous image and line
                        #This saves memory space and the previous values are overwritten
                        h.set_data(new_image)
                        line.set_ydata(new_image[row,:])
                        plt.pause(0.01)
            
            except KeyboardInterrupt:
                sys.exit("That's it, I'm done")
                #Maybe work on some more functionalities here?
                '''
                try:
                    choice = int(input("Which parameter do you want to change?"))
                except ValueError:
                    print("Please choose a number")
                    continue
            
                try:
                    cam = available_cameras[choice]

                    break
                except IndexError:
                    print("Please choose a number that is on the list")
                    continue
                '''


                    
    elif args.file:
    
    
        #Read the image from file and make sure that the pixels and their corresponding values are floats
        print("Reading in image...")


        
        
        image_data = imread('Fiberguide_stacked_HoDi_absorption_7fibers.tiff').astype(np.float32)
        analyze = ASpec(image=image_data) 
        
        image_data_accl = imread('Fiberguide_stacked_HoDi_absorption_7fibers_acclimatised2.tiff').astype(np.float32)
        analyze_accl = ASpec(image=image_data_accl)         
        
        
        
        #test = analyze_accl.determine_polynomial_coefficients_per_fiber(image_data_accl)
        
        #stop
        
        wl_dict = analyze.backwards_spectral_fitting(image_data, resolution=1)#, onthefly=True, threshold=15)
        wl_dict_accl = analyze_accl.backwards_spectral_fitting(image_data_accl, resolution=1)#, threshold=5)

        plt.figure()
        
        for f in wl_dict:
            bwls = wl_dict[f]['wavelengths']
            bint = wl_dict[f]['intensities']
            bwls_accl = wl_dict_accl[f]['wavelengths']
            bint_accl = wl_dict_accl[f]['intensities']            
            #peak_positions_fiber = np.array(peakutils.indexes(np.nan_to_num(255-bint_50), thres=0.25, min_dist=5))
            
            plt.plot(bwls, bint, label='Fiber '+str(f))
            plt.plot(bwls_accl, bint_accl, label='Fiber '+str(f)+' (accl.)',color='black', linestyle='dashed', alpha=0.5)
            
            '''
            for wl in analyze2.hodi_peaks:#[416, 444, 468, 482, 512, 522, 537, 575, 641, 732, 740, 794, 864]:#

                wl_estimate = analyze2.find_nearest(bwls_50[peak_positions_fiber], wl)
                plt.vlines(wl, 0, 300, linestyles='dotted', color=analyze.wavelength_to_rgb(wl))                
                plt.vlines(wl_estimate, 0, 300, linestyles='dashed', color='black', alpha=0.5)
            '''
        '''
        for pwl in analyze.hodi_peaks:
            
            plt.vlines(pwl, 0, 300, linestyles='dotted', color='black', alpha=0.33)
        '''

        plt.legend(loc='best', fancybox=True, framealpha=0.5, frameon=False)    
        plt.show()           
        
  
        
        
        #image_data = imread(args.file).astype(np.float32)
        image_data0 = imread('Fiberguide_stacked_HoDi_absorption_7fibers_0degrees.tiff').astype(np.float32)
        analyze0 = ASpec(image=image_data0)
     
        image_data20 = imread('Fiberguide_stacked_HoDi_absorption_7fibers_20degrees.tiff').astype(np.float32)
        analyze20 = ASpec(image=image_data20, temp=20) 
        
        image_data50 = imread('Fiberguide_stacked_HoDi_absorption_7fibers_50degrees.tiff').astype(np.float32)
        analyze50 = ASpec(image=image_data50, temp=50) 

        pixelpos_dict0 = analyze0.pixelpositions_of_peaks(image_data0, threshold=10)
        pixelpos_dict20 = analyze20.pixelpositions_of_peaks(image_data20)
        pixelpos_dict50 = analyze50.pixelpositions_of_peaks(image_data50)
        



        '''
        wl_dict0 = analyze0.backwards_spectral_fitting(image_data0, resolution=1, threshold=10, onthefly=True)#, onthefly=True, threshold=15)
        wl_dict50 = analyze50.backwards_spectral_fitting(image_data50, resolution=1, onthefly=True)#, threshold=5)

        plt.figure()
        
        for f in wl_dict0:
            if f == 2:
                bwls0 = wl_dict0[f]['wavelengths']
                bint0 = wl_dict0[f]['intensities']
                bwls50 = wl_dict50[f]['wavelengths']
                bint50 = wl_dict50[f]['intensities']            
                #peak_positions_fiber = np.array(peakutils.indexes(np.nan_to_num(255-bint_50), thres=0.25, min_dist=5))
                
                plt.plot(bwls0, bint0, label='Fiber '+str(f)+' (0C)')
                plt.plot(bwls50, bint50, label='Fiber '+str(f)+' (50C)',color='black', linestyle='solid', alpha=0.5)
                


        plt.legend(loc='best', fancybox=True, framealpha=0.5, frameon=False)    
        plt.show() 
        '''
        

        for k in pixelpos_dict20.keys():
            print(k)
            #print(pixelpos_dict[k])
            #print(np.mean(pixelpos_dict[k], axis=0))
            
            
            mean20, median20, sigma20 = aps.sigma_clipped_stats(pixelpos_dict20[k]['positions'], axis=0)
            mean50, median50, sigma50 = aps.sigma_clipped_stats(pixelpos_dict50[k]['positions'], axis=0)
            
            #diff = (np.mean(pixelpos_dict[k]['positions'], axis=0)- np.mean(pixelpos_dict2[k]['positions'], axis=0))*analyze.pxl
            diff = (mean50-mean20)*analyze20.pxl
            
            print(diff, " nm")
            #print(pixelpos_dict[k]['wavelengths'])
            print('-------')
            print(np.mean(pixelpos_dict20[k]['positions'], axis=0))
            print(np.mean(pixelpos_dict50[k]['positions'], axis=0))
            if k in [9,10]:
                print(pixelpos_dict20[k]['positions'])
                print(pixelpos_dict50[k]['positions'])
            print('-------')
            plt.plot(pixelpos_dict20[k]['wavelengths'][np.argsort(pixelpos_dict20[k]['wavelengths'])], diff[np.argsort(pixelpos_dict20[k]['wavelengths'])]/1000., label='fiber'+str(k))
        
        
        plt.xlabel('wavelength [nm]')
        plt.ylabel('shift [um]')
        plt.legend(loc='best', fancybox=True, framealpha=0.5, frameon=False)      
        plt.show()






        plt.figure()
        
        for k in pixelpos_dict0.keys():
            print(k)
            
            if k == 2:
                #print(pixelpos_dict[k])
                #print(np.mean(pixelpos_dict[k], axis=0))
                
                
                mean0, median0, sigma0 = aps.sigma_clipped_stats(pixelpos_dict0[k]['positions'], axis=0)
                mean20, median20, sigma20 = aps.sigma_clipped_stats(pixelpos_dict20[k]['positions'], axis=0)
                mean50, median50, sigma50 = aps.sigma_clipped_stats(pixelpos_dict50[k]['positions'], axis=0)
                
                #diff = (np.mean(pixelpos_dict[k]['positions'], axis=0)- np.mean(pixelpos_dict2[k]['positions'], axis=0))*analyze.pxl
                
                diff20 = (mean20-mean0)*analyze0.pxl
                diff50 = (mean50-mean0)*analyze0.pxl
                
                
                sortwls = np.argsort(pixelpos_dict0[k]['wavelengths'])
                
                sorted_wls = pixelpos_dict0[k]['wavelengths'][sortwls]
                
                sorted_diff20 = diff20[sortwls]
                sorted_diff50 = diff50[sortwls]
                
                diff0 = np.full(len(diff20),0)
                
                
                for idx, val in enumerate(diff0):
                
                    plt.plot([0,20,50], np.array([val, diff20[idx], diff50[idx]])/1000., label=str(sorted_wls[idx]))
        
        
        plt.xlabel('Degrees [C]')
        plt.ylabel('shift [um]')
        plt.legend(loc='best', fancybox=True, framealpha=0.5, frameon=False)      
        plt.show()
            
            

        poldict0 = analyze0.determine_polynomial_coefficients_per_fiber(image_data0, threshold=10)
        poldict20 = analyze20.determine_polynomial_coefficients_per_fiber(image_data20)
        poldict50 = analyze50.determine_polynomial_coefficients_per_fiber(image_data50)
        
                
        f, axarr = plt.subplots(2, 1, sharex=True)
        #f.suptitle('Fiber '+str(fibernr))
        for k in poldict0.keys():
            if k == 2:
                print("plotting coefficients")
                print(poldict0[k]['fitted rows'])
                print(poldict0[k]['1st order coefficients'])
                axarr[0].scatter(poldict0[k]['fitted rows'], poldict0[k]['1st order coefficients'], color='blue')
                axarr[0].scatter(poldict20[k]['fitted rows'], poldict20[k]['1st order coefficients'], color='yellow')
                axarr[0].scatter(poldict50[k]['fitted rows'], poldict50[k]['1st order coefficients'], color='red')
        
        
        axarr[0].set_title('1st order coeff')   
        for k in poldict0.keys():
            if k == 2:
                print("plotting coefficients")
                #print(poldict0[k]['fitted rows'])
                #print(poldict0[k]['0th order coefficients'])
                
                polcoefs0, _ = curve_fit(fitfunc, poldict0[k]['fitted rows'], poldict0[k]['0th order coefficients']) 
                polcoefs20, _ = curve_fit(fitfunc, poldict20[k]['fitted rows'], poldict20[k]['0th order coefficients'])
                polcoefs50, _ = curve_fit(fitfunc, poldict50[k]['fitted rows'], poldict50[k]['0th order coefficients'])


                temppolcoefs0 = np.polyfit(poldict0[k]['fitted rows'], poldict0[k]['0th order coefficients'], 1)
                temppolcoefs20 = np.polyfit(poldict20[k]['fitted rows'], poldict20[k]['0th order coefficients'], 1)
                temppolcoefs50 = np.polyfit(poldict50[k]['fitted rows'], poldict50[k]['0th order coefficients'], 1)

                #print(temppolcoefs0, temppolcoefs20, temppolcoefs50)
            
                print(polcoefs0[0], polcoefs20[0], polcoefs50[0])
                print(polcoefs50[0]-polcoefs0[0], polcoefs50[0]-polcoefs20[0])
                
                
                totalshift = analyze0.pxl*(polcoefs50[0]-polcoefs0[0])/(0.35*1000)
                shift20_50 = analyze20.pxl*(polcoefs50[0]-polcoefs20[0])/(0.35*1000)
                
                print("Total shift = ", totalshift, " um")
                print("20 -> 50 C shift  = ", shift20_50, " um")
                        
                axarr[1].scatter(poldict0[k]['fitted rows'], poldict0[k]['0th order coefficients'], color='blue')
                axarr[1].plot(poldict0[k]['fitted rows'], fitfunc(poldict0[k]['fitted rows'], *polcoefs0), color='blue', linestyle='dashed')
                axarr[1].scatter(poldict20[k]['fitted rows'], poldict20[k]['0th order coefficients'], color='yellow')
                axarr[1].plot(poldict20[k]['fitted rows'], fitfunc(poldict20[k]['fitted rows'], *polcoefs20), color='yellow', linestyle='dashed')
                axarr[1].scatter(poldict50[k]['fitted rows'], poldict50[k]['0th order coefficients'], color='red')
                axarr[1].plot(poldict50[k]['fitted rows'], fitfunc(poldict50[k]['fitted rows'], *polcoefs50), color='red', linestyle='dashed')

        axarr[1].set_title('0th order coeff')
                
        plt.show()     
        f.savefig('polynomial_coefficients_temperature_comparison.png')
        
                    
    
        



        '''

        #image_data2 = imread('../Basler spectra/LaserDrivenLightSource/fiberguide/Climate chamber/50C/check_vibration/OrderFilter_HoDi_VIS_5000us_4.tiff').astype(np.float32)        
        #analyze2 = ASpec(image=image_data2)         
        #wl_dict = analyze.spectral_fitting(image_data, binwidth=1, pol_order = 1)
        wl_dict_20 = analyze.backwards_spectral_fitting(image_data, resolution=1, onthefly=True, threshold=15)
        #wl_dict_50 = analyze2.backwards_spectral_fitting(image_data2, resolution=1)#, threshold=5)

        plt.figure()
        
        for f in wl_dict_20:
            bwls_20 = wl_dict_20[f]['wavelengths']
            bint_20 = wl_dict_20[f]['intensities']
            #bwls_50 = wl_dict_50[f]['wavelengths']
            #bint_50 = wl_dict_50[f]['intensities']            
            #peak_positions_fiber = np.array(peakutils.indexes(np.nan_to_num(255-bint_50), thres=0.25, min_dist=5))
            
            plt.plot(bwls_20, bint_20, label='Fiber '+str(f)+' (20C)')
            #plt.plot(bwls_50, bint_50, label='Fiber '+str(f)+' (50C)',color='black', linestyle='dashed', alpha=0.5)
            
            
            for wl in analyze2.hodi_peaks:#[416, 444, 468, 482, 512, 522, 537, 575, 641, 732, 740, 794, 864]:#

                wl_estimate = analyze2.find_nearest(bwls_50[peak_positions_fiber], wl)
                plt.vlines(wl, 0, 300, linestyles='dotted', color=analyze.wavelength_to_rgb(wl))                
                plt.vlines(wl_estimate, 0, 300, linestyles='dashed', color='black', alpha=0.5)
            

        for pwl in analyze.hodi_peaks:
            
            plt.vlines(pwl, 0, 300, linestyles='dotted', color='black', alpha=0.33)


        plt.legend(loc='best', fancybox=True, framealpha=0.5, frameon=False)    
        plt.show()        
        '''
            
                      
                    