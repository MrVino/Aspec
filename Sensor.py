import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.ticker as mtick
from scipy.misc import imread, imsave
import pypylon
import sys
#from scipy.signal import argrelextrema, savgol_filter
import argparse
import peakutils
import time


class ASpec(object):
    def __init__(self, lambda_min=350., nx=1920, ny=1200, pixelsize= 5.86, linesmm = 150, m=1, NA=0.1, grating_tilt_angle = 14.5, f=100., nfibers=10):
       
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
        
        #The number of fibers used in the spectrograph
        self.n_fibers = nfibers

        #Set the peak positions of the calibration filter (DON'T FORGET TO CHANGE WHEN FILTER IS RECALIBRATED)
        self.LiquidCalibrationFilter()


    def LiquidCalibrationFilter(self):
    
        #Peak positions in nm of Hellma liquid HoDi UV45 filter with serial no. 0028 (calibrated on 1-3-2018)
        peak_wls = np.array([444.20, 575.00, 640.60, 740.00, 864.00]) #241.20, 353.80
        #peak_wls = np.array([241, 278, 287, 347, 354, 361, 416, 444, 468, 482, 512, 522, 537, 575, 641, 732, 740, 794, 864, 900, 950, 1000, 610.])
        #Only use the once that are larger than our lower wavelength limit
        self.hodi_peaks = peak_wls[peak_wls>self.lambda_min]

   
    def exit_angle(self, wl):
        #Applying the grating equation 
        return np.arcsin(self.m*wl/self.d-np.sin(self.theta))
        

    def theoretical_position_on_detector(self, wl):
        #returns the theoretically expected position on the sensor in terms of pixels
        return (np.tan(self.exit_angle(wl))*self.f - self.p0)/self.pxl
                        

    def find_nearest(self, array,value):
        #print(value)
        #print(array)
        #Finds the element in array that is nearest to value. 
        #This is routine is used to relate the expected position of a wavelength peak to its actual position
        idx = (np.abs(array-value)).argmin()
        return array[idx]


    def separate_fibers(self, image, width=25):
        #We stack the image along the x-axis in bins with a chosen width, taking the median pixel value for each y-position. 
        #For each bin, the peakutils package it used to find the peaks along the y-axis, with the assumption that each peak correspond to a fiber. 
        #The position of each peak is saved and used to determine the curvature of the spectrum of each fiber. 
        
        #The startposition on the x-axis (position may be adjusted according to width of spectrum)
        x0 = 0
        #The endposition on the x-axis (position may be adjusted according to width of spectrum)
        x1 = self.nx
        
        #Number of bins along the x-axis with width=width,         
        num_steps = int(np.ceil((x1-x0)/width))
        #Empty array to add the center of each bin in which 
        xcs = []
        #Empty array to add the positions of the peaks
        ypositions = []

        for i in range(num_steps):

            #The median (taking over the width) pixel value for each y-value
            y = np.median(image[:, (i*width+x0):((i+1)*width+x0)], axis=1)

            #Determine the position of the peak in the cross section along the y-axis
            ypos = np.array( peakutils.indexes(y, thres=0.25, min_dist=80) )
            
            #The center (x-position) of the current bin
            xc = np.array( (i+0.5)*width + x0 )
            
            #If the number of peaks found corresponds to the number of fibers used,
            #then we save the positions to determine the curvature of each spectrum 
            if len(ypos)==self.n_fibers:
            
                plt.plot(xc*np.ones_like(ypos), ypos, 'ro')

                ypositions.append(ypos)
                xcs.append(xc)
		


	    #turn the lists into arrays that can be used by the polyfit routine
        xcs = np.array(xcs)
        ypositions = np.vstack(ypositions)
        
        # Fit a low order polynomial to the positions
        polcoefs = np.polyfit(xcs, ypositions, 3)
		
		
		#Plot the found fits
        x = np.linspace(x0, x1, 101)

        for i in range(self.n_fibers):
            y = np.polyval(polcoefs[:, i], x)	
            plt.plot(x, y, 'C0')


        return polcoefs



    def spectral_fitting(self, image, pol_order = 1, lower_margin=30, upper_margin=30, threshold=25, binwidth=2):
        #Routine that stacks all the rows belonging to a single fiber. 
        #First the fibers are separated on the sensor based on their stacked intensity along the y-axis.
        #Then for each fiber, the average position of the peak intensity on the y-axis is taken as the central row.
        #For each row between central row - lower_margin and central row + upper_margin, where the median pixelvalue is above the threshold value, 
        #this routine fits a polynomial to the positions of the 5 calibrated peaks from the HoDi filter. 
        #With this polynomial fit, the x-positions can be converted from pixel position to wavelength. 
        #Finally, the wavelengths (and their corresponding intensities) are binned in wavelength bins with width = binwidth nanometer. 
    
    
        plot = False
    
        #Create an empty dictionary to which we add the fitted wavelengths and corresponding intensities for each fiber
        fitted_wavelengths_all_fibers = {}
    
        #Polynomial coefficients for the curve fitted to each fiber    
        polynomial_coefficients = self.separate_fibers(image)
        t5 = time.time()
        print("Separating fibers took", t5-t4, "secs")
        #Pixel position array (later used to convert to wavelength)
        x = np.arange(self.nx)
  
        f, axarr = plt.subplots(min(max(1,pol_order), 2), (pol_order+1)%min(max(1,pol_order), 2) + (pol_order+1)//min(max(1,pol_order), 2), sharex=True)

        total_deviation_all_peaks = 0
        total_deviation_hodi_peaks = 0
        


        mapping_rows = []
        mapping_columns = []


        print("Polynomial order = ", pol_order)
        if plot:
            plt.figure()
        tfibers = time.time()
        for i in range(self.n_fibers):
            #Empty matrix to store the polynomial coefficients of each row belonging to this fiber
            pol_coefs_matrix = []
            #Empty array to store the derived wavelengths of all rows belonging to this fiber
            unsorted_wavelengths = []
            #Empty array to store the intensities of all rows belonging to this fiber
            unsorted_intensities = []
        
            #Determine the 'central' rows of this fiber by taking the mean y-value of the 
            mean_y_position_fiber = np.mean(np.polyval(polynomial_coefficients[:, i], x))
            
            
            #The median pixel values for every row that belongs to this fiber
            rowmedians = np.median(image[(int(mean_y_position_fiber)-lower_margin):(int(mean_y_position_fiber)+upper_margin),:], axis=1)
            print(rowmedians)
            #Determine between which rows the median pixel values are larger than the threshold
            loweredge =min(np.nonzero(rowmedians > threshold)[0])
            upperedge = max(np.nonzero(rowmedians > threshold)[0])

            #Create an array with the rows over which to loop                        
            rows = np.arange(np.int(mean_y_position_fiber-lower_margin+loweredge), np.int(mean_y_position_fiber-lower_margin+upperedge))
            print(len(rows), "rows, from row", rows[0],"to row",rows[-1])
            #rows = [np.int(mean_y_position_fiber)]
            trows = time.time()
            for r in rows:
                
                #Since the HoDi filter gives an absorption spectrum we take 255 - pixelvalue for each pixel in this row
                #to determine the position of calibrated wavelength peaks (i.e. we convert minima to maxima).
                #The peakutils.indexes is an existing python routine and returns the indeces of values corresponding to
                #a local maximum (The indeces conveniently correspond to pixel position along the x-axis). 
                #The routine finds for local maxima than the 5 absorption lines that we are looking for.
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
                t6 = time.time()
                for wl in self.hodi_peaks:
                    #We compare the position of the found peaks to the theoretically expected positions of the HoDi peaks.
                    #The found peak that are nearest to the theoretically expected peaks are identified as the HoDi peaks.
                    peak = self.find_nearest(peak_positions, self.theoretical_position_on_detector(wl))
                    #peak = self.find_nearest(enhanced_peaks, self.theoretical_position_on_detector(wl))
                    hodi_peak_positions.append(peak)
                    
                    
                    #plt.vlines(peak, 0, self.ny, linestyles='dashed', color='blue', alpha=0.5)
                    #plt.text(peak, 50, str(i), style='italic', bbox={'facecolor':'white', 'alpha':0.5}, ha='center')            
                t7 = time.time()
                #print("Locating HoDi peaks took", t7-t7, "secs")
                #Convert to a numpy array for faster and easier array operations
                hodi_peak_positions = np.array(hodi_peak_positions)
            
          
                #Fit a polynomial with order pol_order to the positions
                #This polynomial determines how the pixel positions are converted to wavelengths for this row
                polcoefs = np.polyfit(hodi_peak_positions, self.hodi_peaks, pol_order)
                
                print("Polynomial coefficients:", polcoefs[0], polcoefs[1])
                #Add the polynomial coefficients of this row to the matrix for the whole fiber              
                pol_coefs_matrix.append(polcoefs)
                t8 = time.time()
                #print("Fitting the polynomial took", t8-t7, "secs")
                
                
                #Plotting routine for the found coefficients
                for idx, pc in enumerate(polcoefs):
                    

                    #print(r, pc)
                    if ((pol_order-idx) > 3) or ((pol_order-idx)==0):
                        suffix = 'th'
                    elif (pol_order-idx) == 1:
                        suffix = 'st'
                    elif (pol_order-idx) == 2:
                        suffix = 'nd'
                    elif (pol_order-idx) == 3:
                        suffix = 'rd'
                        
                    if pol_order>1:
                        axarr[(pol_order-idx)%2, (pol_order-idx)//2].scatter([r], [pc])
                        axarr[(pol_order-idx)%2, (pol_order-idx)//2].set_title(str(pol_order-idx)+suffix+' order coeff')
                    elif pol_order==1:
                        axarr[(pol_order-idx)].scatter([r], [pc])
                        axarr[(pol_order-idx)].set_title(str(pol_order-idx)+suffix+' order coeff')
                    else:
                        axarr.scatter([r], [pc])
                        axarr.set_title(str(pol_order-idx)+suffix+' order coeff')                                                
                  
                
                #Add the fitted wavelength positions of this row to the wavelengths that have been fitted for previous rows belonging to this fiber.    
                unsorted_wavelengths.append(np.polyval(polcoefs, x))
                #Add the pixel intensities of this row accordingly
                unsorted_intensities.append(image_data[r,:])
                
                if pol_order == 1:
                    wls_mapped_backwards = np.arange(self.lambda_min,max(unsorted_wavelengths),binwidth)
                    columns_mapped_backwards = (wls_mapped_backwards - polcoefs[1])/polcoefs[0]
                    rows_for_map = np.full(len(columns_mapped_backwards),r)
                
                
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
            
            #We need to flatten the unsorted wavelengths (and intensities) such that we have a 1-dimensional array of length self.nx x #rows, 
            #instead of a matrix with dimensions self.nx x #rows
            t9 = time.time() 
            print("Looping over rows took", t9-trows, "secs")
            unsorted_wavelengths = np.array(unsorted_wavelengths).flatten()
            unsorted_intensities = np.array(unsorted_intensities).flatten()
            t10 = time.time()
            print("Flattening arrays took", t10-t9, "secs")
                
            #sorted_inds = unsorted_wavelengths.argsort()
            #sorted_intensities = unsorted_intensities[sorted_inds]
            #sorted_wavelengths = unsorted_wavelengths[sorted_inds]
            
            #Create wavelength bins, ranging from the minimum to the maximam found wavelength
            bins = np.arange(int(round(min(unsorted_wavelengths)))-binwidth/2, int(round(max(unsorted_wavelengths)))+1-binwidth/2, binwidth)       
            
            #The numpy digitize routine bins the unsorted wavelengths according to the provided bins.
            #It returns the indices that map each wavelength to its corresponding bin
            binned_inds = np.digitize(unsorted_wavelengths, bins)
            
            #The indices given by the digitize routine are used to map the intensties to their corresponding bin
            #and take the mean value of that bin
            bin_means = np.array([unsorted_intensities[binned_inds == idb].mean() for idb in range(1, len(bins))])
                        
            t11 = time.time()
            print("Binning wavelengths and determining means took", t11-t10, "secs")
            #Remove the last element of the bins (such that the bins array has the same length as the bin_means array),
            #and shift by binwidth/2 such that the wavelengths correspond to the middle of the bin            
            binned_wavelengths = bins[:-1]+binwidth/2
            if plot:
                plt.plot(binned_wavelengths, bin_means)
            
            #Add the binned wavelengths and intensities of this fiber to a dictionary for all fibers
            fitted_wavelengths_all_fibers[i] = {'wavelengths':binned_wavelengths, 'intensities':bin_means}
            
            
            #Determine the peak positions in the binned spectrum
            peak_positions_fiber = np.array(peakutils.indexes(np.nan_to_num(255-bin_means), thres=0.25, min_dist=10))
            
            #Determine how much the peak position in the binned spectrum deviate from the actual peak position
            deviation_all_peaks = 0.
            deviation_hodi_peaks = 0.
            
            
            for wl in [416, 444, 468, 482, 522, 537, 575, 641, 740, 794, 864]:#self.hodi_peaks:
                
                #For each peak reference wavelength we look for the peak that is closest in wavelength position the binned spectrum
                wl_estimate = self.find_nearest(binned_wavelengths[peak_positions_fiber], wl)
                #Add the absolute value of the difference to the total sum of deviations
                deviation_all_peaks += abs(wl_estimate - wl)
                if plot:
                    plt.vlines(wl_estimate, 0, 300, linestyles='dashed', color='black', alpha=0.5)

                #plt.vlines(self.find_nearest(enhanced_peaks_fiber, wl), 0, 300, linestyles='dashed', color=self.wavelength_to_rgb(wl), alpha=0.33)
            



            
            for pwl in self.hodi_peaks:
                
                if plot:
                    plt.vlines(pwl, 0, 300, linestyles='dotted', color='black', alpha=0.33)
                
                wl_estimate_hodi = self.find_nearest(binned_wavelengths[peak_positions_fiber], pwl)
                deviation_hodi_peaks += abs(wl_estimate_hodi - pwl)                
                    
            #11 is the total number of known peak wavelengths in the HoDi filter, 5 is the number of calibrated peak wavelengths in the HoDi filter
            print("---------------------------------------------------")
            print("Average deviation for fiber", i, " and all peaks is", deviation_all_peaks/11.)
            print("Average deviation for fiber", i, " and only HoDi peaks is", deviation_hodi_peaks/5.)
            
            
            #Calculting the total deviation of all fibers
            total_deviation_all_peaks += deviation_all_peaks
            total_deviation_hodi_peaks += deviation_hodi_peaks
            t12 = time.time()
            print("Calculating mean deviation took", t12-t11, "secs")
            
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
        print("=======================================================")
        print("Total mean deviation for all peaks is", total_deviation_all_peaks/(11.*self.n_fibers))
        print("Total mean deviation for HoDi peaks is", total_deviation_hodi_peaks/(5.*self.n_fibers))
        print("=======================================================")
        t13 = time.time()
        print("Analyzing spectrum took", t13-tfibers, "secs")
        #plt.show()

        
        pol_coefs_matrix = np.array(pol_coefs_matrix)
                
        #Change the scaling of the y-axis 
        for idx in np.arange(pol_order+1):
        
        
            if pol_order > 1:
                axarr[idx%2, idx//2].set_ylim(min(pol_coefs_matrix[:,(pol_order-idx)])*0.99,max(pol_coefs_matrix[:,(pol_order-idx)])*1.01)
            elif pol_order == 1:
                axarr[idx].set_ylim(min(pol_coefs_matrix[:,(pol_order-idx)])*0.99,max(pol_coefs_matrix[:,(pol_order-idx)])*1.01) 
            else:
                axarr.set_ylim(min(pol_coefs_matrix[:,(pol_order-idx)])*0.99,max(pol_coefs_matrix[:,(pol_order-idx)])*1.01)
                
        f.tight_layout()        
        f.savefig(str(self.n_fibers)+'fibers_'+str(pol_order)+'polynomial_orders.png')
       
        
        return fitted_wavelengths_all_fibers#total_deviation_all_peaks/55., total_deviation_hodi_peaks/25.  
        
    def box_extraction(self, image):
        data = image
        #data = fits.getdata(filename)
        #data = data[100::,:]
        
        box_width = 2
        box_heigth = 100
        
        s = np.linspace(0, data.shape[1], data.shape[1]//box_width)
        
        
        separated_fibers = self.separate_fibers(image)
        
        num_orders = separated_fibers.shape[1]
        print("num_orders = ", num_orders)
        poly_order = separated_fibers.shape[0]-1
        print("Order of polynomial = ", poly_order)
        Ny = data.shape[0]
        Nx = data.shape[1]
        
        ccd_grid = make_pupil_grid([Nx,Ny],[Nx, Ny]).shift([Nx/2, Ny/2])
        echelle_spectrum = Field(data.ravel(), ccd_grid)
        
        flux = np.zeros((num_orders, s.size))
        order_offset = 27
        for i in [0,]:#range(num_orders):
        	order = i + order_offset
        	# Position
        	rx = s
        	ry = np.polyval(separated_fibers[:, order], s) + 1
        
        	imshow_field(echelle_spectrum, aspect='auto', vmin=0, vmax=5000)
        	plt.plot(rx, ry)
        	plt.show()
        	
        	masky = (ccd_grid.y < (ry.max()+20)) * ( ccd_grid.y > (ry.min()-20) )
        	masked_spectrum = echelle_spectrum[masky>0]
        	masked_grid = CartesianGrid(UnstructuredCoords([ccd_grid.x[masky>0], ccd_grid.y[masky>0]]))
        
        
        	start = time.time()
        	for si in range(s.size):
        		if si % 20 == 0:
        			print(si)
        		pixel_index = si
        
        		aperture = rectangular_aperture((box_width, box_heigth), (rx[pixel_index], ry[pixel_index]))(masked_grid)
        		flux[order, pixel_index] = np.sum(masked_spectrum * aperture)
        
        		#plt.subplot(1,2,1)
        		#imshow_field( np.log10(abs(echelle_spectrum)), vmin=0, vmax=3.5, aspect='auto')
        		#plt.xlim([rx[si]-10, rx[si]+10])
        		#plt.ylim([ry[si]-10, ry[si]+10])
        		#plt.subplot(1,2,2)
        		
        		#imshow_field(aperture, aspect='auto')
        		#plt.xlim([rx[si]-10, rx[si]+10])
        		#plt.ylim([ry[si]-10, ry[si]+10])
        
        		#plt.show()
        
        	end = time.time()
        	print("Elapsed {:g}".format(end-start))
        
        plt.plot(s, flux[order_offset])
        plt.show()





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
        
        self.cam.properties['ExposureTime'] = 400.0 
        


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
    
    
    tstart = time.time()
    parser = argparse.ArgumentParser(description='Choose in which mode you want to use the script')

    parser.add_argument("--live", 
                        dest="live",
                        action='store_true', 
                        default = False,
                        help="Captures a live feed from the camera, instead of reading a saved image")
    parser.add_argument("--file", 
                      dest="file", 
                      default = 'HDR_stacked_HoDi_absorption.tiff',
                      help="The captured image that you want to plot")
    parser.add_argument("-n", 
                      dest="n_fibers", 
                      default = 10,
                      type=int,
                      help="The number of fibers on the image")


    args = parser.parse_args()

    analyze = ASpec(nfibers=args.n_fibers)
    
    #Relevant wavelengths in nm: 401.7, 636.7, 855.7 correspond to the lasers used and 610 to the cutoff wavelength of the low-pass filter
    wavelengths = [401.7, 636.7, 855.7, 610.]#, 360., 950., 1000., 610.]

    spectrum_width_pixels =  analyze.theoretical_position_on_detector(max(wavelengths)) - analyze.theoretical_position_on_detector(min(wavelengths))
    spectrum_width_mm = spectrum_width_pixels*analyze.pxl/1.e6

    print(spectrum_width_pixels, spectrum_width_mm)


    if args.live:
        #if one chooses to run this script 'live', then pypylon must be installed and can be imported.
        #To use this script with saved images, pypylon is not necessarily installed 
        import pypylon
        plt.style.use('dark_background')
        
        analyze.connect_camera()
        analyze.camera_properties()
            
        
        for image_data in analyze.take_snapshots(1):
        
        
            #Select the row with the largest total photon flux
            row = np.argmax(analyze.sum_rows(image_data))
            print(row)
            #Save the image and cross section as an instance that can be overwritten in each frame
            h = plt.imshow(image_data, origin='lower', cmap='gist_gray',vmin=0, vmax=255)
            line, = plt.plot(image_data[row,:])
            #Plot the relevant wavelengths to align the spectrograph
            for wl in analyze.hodi_peaks:#wavelengths:##
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
        t0 = time.time()
        image_data = imread(args.file).astype(np.float32)
        t1 = time.time()
        print("took", t1-t0, "secs")
        #image_data_cont = imread("HDR_stacked_continuum.tiff").astype(np.float32)
        #image_data_hodi = imread("HDR_stacked_HoDi_absorption.tiff").astype(np.float32)
        #image_data = image_data_cont-image_data_hodi
        #Select the row with the largest total photon flux
        print("Determining Row with max signal")
        row = np.argmax(analyze.sum_rows(image_data))
        t2 = time.time()
        print("took", t2-t1, "secs")

        #NOTE that the imaged is flipped horizontally here. The origin of a saved image (0,0) is generally displayed in the upper left corner
        print("Plotting image...")
        plt.imshow(image_data, origin='lower', cmap='gist_gray',vmin=0, vmax=255)
        t3 = time.time()
        print("took", t3-t2, "secs")
        #Plot the spectrum of the row with the largest total photon flux
        #plt.plot(image_data[row,:])
        #Plot the theoretically expected position of relevant wavelengths
        print("Initializing HoDi wavelengths and plotting them")
        analyze.LiquidCalibrationFilter()
        for wl in analyze.hodi_peaks:#wavelengths:#
            analyze.plot_wavelength(wl)
        t4 = time.time()
        print("took", t4-t3, "secs")
        #Indicate from which row the spectrum is shown
        plt.hlines(row, 0, 1920, linestyles='dotted', color='blue', alpha=0.33)
        plt.xlim(0,1920) 
        plt.ylim(0,1200)
        #analyze.separate_fibers(image_data)
        all_orders = np.arange(4)+1
        performance = []
        #for po in all_orders:
        #performance.append(analyze.spectral_fitting(image_data, binwidth=1, pol_order = po))
        wl_dict = analyze.spectral_fitting(image_data, binwidth=2, pol_order = 1)
        #print(performance)
        tend = time.time()
        print("Total time taken", tend-tstart, "secs")
        plt.show()
        plt.figure()
        for f in wl_dict:
            bwls = wl_dict[f]['wavelengths']
            bint = wl_dict[f]['intensities']
            peak_positions_fiber = np.array(peakutils.indexes(np.nan_to_num(255-bint), thres=0.25, min_dist=10))
            
            plt.plot(bwls, bint, label='Fiber '+str(f))
            
            for wl in analyze.hodi_peaks:#[416, 444, 468, 482, 522, 537, 575, 641, 740, 794, 864]:#

                wl_estimate = analyze.find_nearest(bwls[peak_positions_fiber], wl)
                plt.vlines(wl, 0, 300, linestyles='dotted', color=analyze.wavelength_to_rgb(wl))                
                plt.vlines(wl_estimate, 0, 300, linestyles='dashed', color='black', alpha=0.5)

        '''    
        for pwl in analyze.hodi_peaks:
            
            plt.vlines(pwl, 0, 300, linestyles='dotted', color='black', alpha=0.33)
        '''
        plt.legend(loc='best', fancybox=True, framealpha=0.5, frameon=False)    
        plt.show()        
            
            
                      
                    