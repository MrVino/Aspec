import glob
import matplotlib.pyplot as plt
from scipy.misc import imread, imsave
import cv2
import numpy as np
np.set_printoptions(threshold=np.inf)
from skimage import io

def stack_spectra():
    #path = './Basler spectra/Only LED'
    #path = './Basler spectra/With lasers'
    #path = './Basler spectra/LaserDrivenLightSource/new'#/composite'
    path = '../Basler spectra/LaserDrivenLightSource/fiberguide/Climate chamber/Check after acclimatisation/check 2/'#Double fibers'
    darkpath = '../Basler spectra/LaserDrivenLightSource/fiberguide/Climate chamber/Check after acclimatisation/check 2/'#Double fibers/darks'
    #path = '../Basler spectra/LaserDrivenLightSource/fiberguide/7 fibers/With actuator/5dBgain/'#Double fibers'
    #darkpath = '../Basler spectra/LaserDrivenLightSource/fiberguide/7 fibers/With actuator/5dBgain/'#Double fibers/darks'    
    #darkpath = './Basler spectra/LaserDrivenLightSource/new/darks'#/darks'
    #dark_files = get_filepaths(full_path, '.FIT', 'dark')
    #bias = make_master_dark(dark_files)
    
    science_files = glob.glob(path+"/OrderFilter_HoDi*")
    dark_files = glob.glob(darkpath+"/OrderFilter_dark*")

    #science_files = glob.glob(path+"/NoFilter_HoDi_continuum**")
    #dark_files = glob.glob(darkpath+"/dark*")
    
    bias = make_master_dark(dark_files)
    
    print(science_files)
    science_data = []
    for f in science_files:
        print(f)
        file = imread(f).astype(np.uint8)
        file2 = cv2.imread(f, cv2.IMREAD_ANYDEPTH )
        
        print(np.max(file))
        print(np.max(file2))
        
        
        plt.imshow(file)	
        plt.show()
        reduced_image = file-bias
        #if 'NoFilter_HoDi_continuum34us_VIS_middle.tiff' in f:
            #print("flipping")
            #reduced_image = np.flip(reduced_image, 0)
        science_data.append(reduced_image)
        print(np.amax(reduced_image))
    
    science_data = np.array(science_data)

    #science_data -= np.median(science_data, axis=1, keepdims=True)
    stacked_data = np.sum(science_data, axis=0)#/len(science_files)
    
    stacked_data[stacked_data<0]=0
    print("Maximum pixel value = ", np.max(stacked_data))
    print("Maximum pixel value (uint8) = ", np.max(stacked_data).astype(np.uint8))
    plt.imshow(stacked_data.astype(np.uint8))	
    plt.show()
    #imsave('Fiberguide_stacked_HoDi_absorption_7fibers_withactuator.tiff', stacked_data)
    cv2.imwrite('Fiberguide_stacked_HoDi_absorption_7fibers_acclimatised2.tiff', stacked_data.astype(np.uint8))


def make_master_dark(dark_files):

	data = []
	for f in dark_files:
		print(f)				
		file = imread(f).astype(np.uint8)
		
		data.append(file)

	

	data = np.array(data)
	bias = np.median(data, axis=0)

	return bias





stack_spectra()