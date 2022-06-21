''' 
AOSLO systen
-------------------
For creating an AOSLO object matched to a user's system.


'''

import numpy
import yaml

from scipy.interpolate import interp1d

class AOSLO:
    
    ''' An AOSLO class. Instances of this class are AOSLO objects with 
    parameters specified in the system parameter yaml file
    
    '''

    
    def __init__(self, system_parameter_file):
        ''' Creates an instance of an AOSLO object with the settings in the 
        given system parameters yaml file
        
        Parameters
        ----------
        system_parameter_file : string
            The path to the system parameter file
        
        '''
        
        self.system_parameter_file = system_parameter_file
        
        # Open the system parameters file
        with open(self.system_parameter_file, 'r') as f:
            self.parameters = yaml.safe_load(f)
    
        # Calculate the field of view based on the number of scan lines, 
        # number of pixels per line and the pixel size
        self.fov = [self.parameters['pixel_size_arcmin_y_x'][0] * self.parameters['number_scan_lines_excluding_flyback'],
                    self.parameters['pixel_size_arcmin_y_x'][1] * self.parameters['number_pixels_per_line']]

        # Set up the scan pattern
        self.setScanners()

        
    def fastScan(self, time, blanking=None):
        '''Creates an array of sample positions across the fast (horizontal) 
        scan
        
        
        Parameters
        ----------
        time : array_like
            An array containing the time samples for one frame
        
            
        Returns
        -------
        fast_scan : array_like
            Fast scan (horizontal) sample positions (arcminutes) for one frame.
        
        '''

        # Create a sinusoidal scan pattern from 0 to number_pixels_per_line
        fast_scan = (self.parameters['number_pixels_per_line']-1) * (0.5 - 0.5 * numpy.cos(time * 2*numpy.pi * self.parameters['frequency_fast_scanner_Hz']))

        line_start = numpy.where(fast_scan == 0.0)[0]
        forward_scan = numpy.zeros((fast_scan.shape))
        for i in range(len(line_start)):
            forward_scan[line_start[i]:line_start[i] + self.parameters['number_pixels_per_line']] = 1
        
        
        return fast_scan, forward_scan

    def slowScanLinear(self, time):

        '''Creates an array of sample positions along the slow (vertical) 
        scan, including a flyback region
        
        
        Parameters
        ----------
        time : array_like
            An array containing the time samples for one frame
        
            
        Returns
        -------
        slow_scan : array_like
            Slow scan (vertical) sample positions (arcminutes) for one frame.
        
        '''
        
        # Create two linear ramps - one for the forward direction and one for
        # the flyback
        ramp = numpy.linspace(0,self.parameters['number_scan_lines_excluding_flyback']-1, 2*self.parameters['number_pixels_per_line']*self.parameters['number_scan_lines_excluding_flyback'])
        flyback = numpy.linspace(self.parameters['number_scan_lines_excluding_flyback']-2,1, 2*self.parameters['number_pixels_per_line']*(self.parameters['number_scan_lines_including_flyback']-self.parameters['number_scan_lines_excluding_flyback']))
        slow_scan = numpy.zeros((len(time)))
        slow_scan[:-len(flyback)] = ramp
        slow_scan[-len(flyback):] = flyback
        

        return slow_scan
    
    
    def setScanners(self):
        '''Sets up the spatio-temporal sampling array for this AOSLO's scanners
        
        '''
        
        # Calculate the number of samples in one frame
        self.n_samples = self.parameters['number_scan_lines_including_flyback'] * self.parameters['number_pixels_per_line'] * 2
        # Work out the time resolution based on the frame rate and number of
        # samples
        self.time_resolution = 1. / (self.parameters['frequency_fast_scanner_Hz'] * self.parameters['number_pixels_per_line'] * 2)
        
        # Set up arrays of time samples, slow scan sample positions and fast
        # scan positions
        self.time = numpy.linspace(0, self.n_samples-1, self.n_samples) * self.time_resolution
        self.slow = self.slowScanLinear(self.time)
        self.fast, self.scan_direction = self.fastScan(self.time)
        
        return
    
    def desinusoid(self, image):
        '''Desinusoids an image based on this AOSLO's resonant scanner
        
        
        Parameters
        ----------
        image : array_like
            An array containing the image with sinusoidal distortion
        
            
        Returns
        -------
        image_fast_desinusoid : array_like
            The desinusoided image
        
        '''

        # Define the horizontal samples (sinusoidal)
        x, forward_scan = self.fastScan(numpy.linspace(0, image.shape[1]-1, image.shape[1]) * self.time_resolution)
        x, forward_scan = self.fastScan(numpy.linspace(self.fast.min(), self.fast.max(), image.shape[1]) * self.time_resolution)
        
        
        # Specify a linear sampling grid
        x2=numpy.linspace(x[0], x[-1], len(x))
    
        # Create an array to containg the desinusoided image
        image_fast_desinusoid=numpy.zeros(image.shape, dtype=numpy.float64)
        
        # Interpolate along fast axis
        for i in range(image.shape[0]):
            # Interpolate the image with sinusoidal sampling
            f=interp1d(x,image[i])
            # Compute the pixel values for linear sampling
            image_fast_desinusoid[i] = f(x2)

        return image_fast_desinusoid
        
        
        
        
