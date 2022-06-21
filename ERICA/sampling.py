''' 
Sampling
-------------------
Functions for sampling a synthetic retina.
'''

import numpy
 
from numba import jit

#######################
# SCANNING FUNCTIONS #
######################


@jit(nopython=True)
def performScanning(retina_params_pixels, pixel_value_array, x, y, nstdev=6.0):
    '''Performs the scanning over a given retina with a given sampling pattern.
    This function uses number to speed up the operation but it could be 
    implemented using the multiprocessing module for parallelisation.
    
    Parameters
    ----------
    retina_params_pixels : array_like
        The retinal parameter array, rescaled to pixel units for the cone
        positions and sizes.
    pixel_value_array : array_like
        An array of pixel values in which the computed intensities are stored.
    idx : array_like
        The sample indices, i.e. an array of integers from 0 to number of 
        pixels minus 1.
    x : array_like
        The horizontal sample positions (in pixels), which includes the scanner
        position and the movement of the retina.
    y : array_like
        The vertical sample positions (in pixels), which includes the scanner
        position and the movement of the retina.
    nstdev : float (optional)
        The number of standard deviations (cone widths) to include when
        determining which samples will influenced by a particular cone.
    
        
    Returns
    -------
    pixel_value_array : array_like
        The pixel values after scanning the retina
        
    '''
    

    mean_y = retina_params_pixels[0]
    mean_x = retina_params_pixels[1]
    std_y = retina_params_pixels[2]
    std_x = retina_params_pixels[3]
    theta = retina_params_pixels[4]
    amp = retina_params_pixels[5]
    
    # Allow for the cone profile to be elliptical
    a = ((numpy.cos(theta) ** 2) / (2 * std_x ** 2)) + ((numpy.sin(theta) ** 2) / (2 * std_y ** 2))
    b = ((-numpy.sin(2 * theta)) / (4 * std_x ** 2)) + ((numpy.sin(2 * theta)) / (4 * std_y ** 2))
    c = ((numpy.sin(theta) ** 2) / (2 * std_x ** 2)) + ((numpy.cos(theta) ** 2) / (2 * std_y ** 2))
    
    # Iterating over cones is faster since there are always fewer cones than pixels.
    # For each each cone, find the samples during which that cone will be illuminated.
    # We take and samples within nstdev of the centre of the cone to be relevant.
    for i in range(mean_x.shape[0]):

        # Find the indices of the samples that will be within nstdev of the centre of the cone
        indices_relevant = numpy.where(
                ((x - mean_x[i])**2 <= (std_x[i] * nstdev)**2) &
                ((y - mean_y[i])**2 <= (std_y[i] * nstdev)**2))[0]
        
        indices_relevant = numpy.where(
                (numpy.absolute(x - mean_x[i]) <= numpy.absolute(std_x[i] * nstdev)) &
                (numpy.absolute(y - mean_y[i]) <= numpy.absolute(std_y[i] * nstdev)))[0]

        # Compute the intensity vs. sample index for this cone
        g0 = amp[i] * numpy.exp(-((a[i] * (x[indices_relevant] - mean_x[i]) ** 2) + (c[i] * (y[indices_relevant] - mean_y[i]) ** 2) + (2 * b[i] * (x[indices_relevant] - mean_x[i]) * (y[indices_relevant] - mean_y[i]))))
        
        # Add it to the pixel values array
        pixel_value_array[indices_relevant] += g0

    return pixel_value_array


def sampleRetina(retina_params_microns, eye_motion_arcmin, fast_scan_pixels, slow_scan_pixels, pixel_size_arcmin, microns_per_degree, eccentricity_deg=[0.0, 0.0], torsion=None):
    
    '''Combines the scanner positons and eye movementsand sets up scanning of
    the retina. All units are converted to pixels inside this function.
    
    Parameters
    ----------
    retina_params_pixels : array_like
        The retinal parameter array, rescaled to pixel units for the cone
        positions and sizes.
    eye_motion_arcmin : array_like
        The gaze positions of the eye. This should be an array of size (2,N)
        where N is the number of pixels in the image (the number of temporal
        samples)
    fast_scan_pixels : array_like
        The position of the fast (horizontal scanner) for each temporal sample.
        This should be an array of size (N)
    slow_scan_pixels : array_like
        The position of the slow vertical scanner) for each temporal sample.
        This should be an array of size (N)
    pixel_size_arcmin : float
        The size of each pixel in arcminutes.
    microns_per_degree : float
        The number of microns on the retinal suface per degree of visual angle.
        This will be determined by the axial length of the eye.
    
        
    Returns
    -------
    intensity : array_like
        The pixel values after scanning the retina. Values are not scaled.
    retina_params_pixels_scanned : array_like
        The ground truth data (retinal parameter array) for the field of view
        of this image.
    fast_indices: array_like
        The horizontal sample positions, including scanner motion, eye motion, 
        fixation eccentricity and imaging eccentricity
    slow_indices : array_like
        The vertical sample positions, including scanner motion, eye motion, 
        fixation eccentricity and imaging eccentricity
        
    '''
    
    
    # Convert retinal parameter array from microns to pixels (first 5 elements, 5 is rotation angle and 6 is reflectance)
    retina_params_pixels = numpy.copy(retina_params_microns)
    retina_params_pixels[:4] /= microns_per_degree
    retina_params_pixels[:4] *= 60.
    retina_params_pixels[[0,2]] /= pixel_size_arcmin[0]
    retina_params_pixels[[1,3]] /= pixel_size_arcmin[1]
    
    # Convert degrees of angluar motion to motion in pixels
    eye_motion_pixels = numpy.copy(eye_motion_arcmin)    
    eye_motion_pixels[0] /= pixel_size_arcmin[0]
    eye_motion_pixels[1] /= pixel_size_arcmin[1]
    
    # Make sure that (0,0) is the centre of the field of view    
    fast_scan_sampling = fast_scan_pixels - numpy.mean(fast_scan_pixels)
    slow_scan_sampling = slow_scan_pixels - numpy.mean(slow_scan_pixels)
    
    # The slow scanner runs from top (row 0) to bottom (row 511). Here we have
    # subtraced the mean such that the centre of the scan patch is the 
    # reference point.
    # In fundus view positive y values indicate an upward position on the 
    # retina, and so the slow scan sampling positions should be inverted
    # (i.e. run from +256 to -256 not -256 to +256)
    
    slow_scan_sampling = -slow_scan_sampling
    
    
    

    # A positive vertical movement of the eye is upwards and corresponds to a 
    # vertical downward movement of the retina, as seen in fundus view. This is
    # a relative vertical upward movement of the AOSLO beam. An upward movement
    # in fundus view corresponse to positive changes. So, vertically, the eye
    # movement must be added to the scanner position
    #    
    # A positive horizontal movement of the eye is rightwards and corresponds 
    # to a horizontal rightward movement of the retina, as seen in fundus view. 
    # This is a relative horizontal leftward movement of the AOSLO beam. So, a 
    # positive horizontal eye movement corresponds to a relative 
    # negative change in the scanner position. So, horizontally, the eye movement 
    # must be subtracted from the scanner position.

    # N.B. axis 0 is vertical (slow scan) and axis 1 is horizontal (fast scan)
    fast_indices = fast_scan_sampling - eye_motion_pixels[1]
    slow_indices = slow_scan_sampling + eye_motion_pixels[0]
    
    # As explained above, positive values in fundus view indicate upward and
    # rightward movement of the AOSLO beam. So, the imaging eccentricity 
    # defined in fundus view should be added to the scanner position data.
    fast_indices += eccentricity_deg[1] * 60 / pixel_size_arcmin[1]
    slow_indices += eccentricity_deg[0] * 60 / pixel_size_arcmin[0]

    
    ### Add torsion - Untested! ###
    if type(torsion) != type(None):
        fast_indices = fast_indices * numpy.cos(torsion*numpy.pi/180.) - slow_indices * numpy.sin(torsion*numpy.pi/180.)
        slow_indices = fast_indices * numpy.sin(torsion*numpy.pi/180.) + slow_indices * numpy.cos(torsion*numpy.pi/180.)
    ###

    # Find the cones that will be scanned, plus an additional 20 cone widths (standard deviations) beyond, to reduce computation time
    cones_scanned = numpy.where(
            (retina_params_pixels[1] > fast_indices.min() - numpy.mean(retina_params_pixels[3])*20) &
            (retina_params_pixels[1] < fast_indices.max() + numpy.mean(retina_params_pixels[3])*20) &
            (retina_params_pixels[0] > slow_indices.min() - numpy.mean(retina_params_pixels[2])*20) &
            (retina_params_pixels[0] < slow_indices.max() + numpy.mean(retina_params_pixels[2])*20))[0]
    
    retina_params_pixels_scanned = retina_params_pixels[:,cones_scanned].astype(float)
    
    # Compute the intensity data
    # Start with an empty array of pixel values (setting this up outside the scanning function is faster)
    pixel_values = numpy.zeros((fast_indices.shape[0]), dtype=float)

    # Scan the retina
    intensity = performScanning(retina_params_pixels_scanned, pixel_values, fast_indices, slow_indices)
    
    # Reference the cone locations to the start point of the scanner, so that 
    # they will be correct when displayed on the image
    retina_params_pixels_scanned[0] *= -1
    retina_params_pixels_scanned[0] -= -slow_indices[0]
    retina_params_pixels_scanned[1] -= fast_indices[0]

    # Return the intensity data and the ground truth data for this image
    return intensity, retina_params_pixels_scanned, fast_indices, slow_indices

def quickLookImage(retina_params_microns, eccentricity, field_of_view, microns_per_degree, pixel_size_arcmin, upsample_factor=1):
    
    '''Produces an image of the ground truth retina that doesn't account for
    diffraction, aberrations, noise, the pinhole or the scan pattern. This is 
    useful just to quickly check what the cone mosaic is going to look like.
    
    Parameters
    ----------
    retina_params_pixels : array_like
        The retinal parameter array, rescaled to pixel units for the cone
        positions and sizes.
    eccentricity : array_like
        The retinal eccentricity (vertical, horizontal) at which to image, in 
        degrees.
    field_of_view : array_like
        The field of view (vertical, horizontal) of the frame, in degrees.
    microns_per_degree : float
        The number of microns on the retinal suface per degree of visual angle.
        This will be determined by the axial length of the eye.
    pixel_size_arcmin : float
        The size of each pixel in arcminutes.
    
        
    Returns
    -------
    image : array_like
        The pixel values after scanning the retina. Values are not scaled.
    ground_truth_array : array_like
        The ground truth data (retinal parameter array) for the field of view
        of this image.
        
    '''
    
    # Convert the field of view from degrees to pixels
    field_of_view_pixels = [field_of_view[0] * 60 / pixel_size_arcmin[0], field_of_view[1] * 60 / pixel_size_arcmin[1]]
    
    # Set up the sampling grid
    a = field_of_view_pixels[0]*0.5
    b = field_of_view_pixels[1]*0.5
    c = int(round(field_of_view_pixels[0]))*1j * upsample_factor
    d = int(round(field_of_view_pixels[1]))*1j * upsample_factor
    y, x = numpy.mgrid[-a:a:c,-b:b:d]
    
    # Define the fast and slow scan sampling pattern
    fast_scan_pixels = x.astype(float).flatten()
    slow_scan_pixels = y.astype(float).flatten()
    
    # Set eye movement to be stationary
    eye_motion_arcmin = numpy.zeros((2, fast_scan_pixels.shape[0]))
    
    # Scan the retina
    intensity, ground_truth_array, fast_scan, slow_scan = sampleRetina(retina_params_microns, eye_motion_arcmin, fast_scan_pixels, slow_scan_pixels, pixel_size_arcmin, microns_per_degree, eccentricity)
    
    # Reshape the intensity stream into an image
    image = intensity.reshape(y.shape)
    
    return image, ground_truth_array