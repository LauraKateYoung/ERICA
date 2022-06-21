# -*- coding: cp1252 -*-
# Copyright (c) 2014, Durham University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without 
# modification, are permitted provided that the following conditions are 
# met:
#
# 1. Redistributions of source code must retain the above copyright 
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright 
# notice, this list of conditions and the following disclaimer in the 
# documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its 
# contributors may be used to endorse or promote products derived from 
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT 
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, 
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY 
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# THIS FILE COPYRIGHT (C) THE UNIVERSITY OF DURHAM 2014
# THIS FILE IS NOT FOR RE-DISTRUBUTION. ALL RIGHTS RESERVED

# EDIT AT YOUR OWN RISK

# 2014-07-08       Laura Young      Version 1.0

# 2021-06-01       Laura Young      Version 1.1
#  - Added an apodisation function to simulate the optical Stiles-crawford
#   effect.
#  - Removed the convolution function - this is now called from ERICA_toolkit
#  - Made Python 3 compatible.

'''


A module for convolving a point spread function (PSF) with an input image array.
The PSF and the input image array should have the same pixel scale (radians per
pixel). A description of the code can be found in the Supplementary materials
section of the following article (which should be cited if you use this code):
    
Young, L. K., and Smithson, H. E. (2014), Critical band masking reveals the 
effects of optical distortions on the channel mediating letter identification.
Frontiers in Psychology, 5:1060.


'''


from __future__ import division
import numpy
import sys

# Default variables for optional keyword arguments. If you always use
# the same values you could set them here to save changing the values at each
# each function call. If you want to use different values at each function call
# ensure that the same values are entered for all other functions, such as 
# when calculating the number of pixels required in the input image array.

ARRAY_SIZE = 512
HIGHEST_RADIAL_ORDER = 5
WAVELENGTH = 550.0e-9
OVERSAMPLING = 2
PUPIL_RADIUS = 1.25e-3
WAVEFRONT_TOL=7.0
PSF_FRACTION=0.1
N_BITS=8

class Zernike:
    ''' A Zernike class
    
    This class contains the coordinates, index mapping and Zernike
    matrix for a given array size (number of pixels accross the wavefront).
    
    Parameters
    ----------
    array_size : int, optional
        Size of an individual element (mode) the Zernike matrix, i.e. the
        number of pixels across the wavefront, which should be square. It
        is preferable for array_size to be a power of two to give a padded
        array size that is also of power two (if an even number is used 
        for the oversampling factor). This is for performance reasons when 
        computing the discrete Fourier tranform.
    order : int, optional
        The highest radial order to be calculated.
    print_indices : bool, optional
        If print_indices = True then the index with both its radial and
        angular order will be printed to the screen. The default value
        is False.
    
    Attributes
    ---------
    x : array_like
        The x cartesian coordinates for the given array size.
        
    y : array_like
        The y cartesian coordinates for the given array size.
        
    r : array_like
        The radial polar coordinates for the given array size.
        
    theta : array_like
        The angular polar coordinates for the given array size.
        
    zernike_matrix : array_like
        The Zernike matrix, of size (number of modes, array_size, 
        array_size), where the number of modes is determined by the
        highest radial order to be calculated.
        
    index_mapping : array_like
        The mapping between single index (as used in the zernike_matrix)
        and the double index notation. Indexing always begins at zero
        (piston) to maintain the standard single indexing convention
        defined by Thibos et al. (2000).
        
    
    '''

    def __init__(self,array_size=ARRAY_SIZE, order=HIGHEST_RADIAL_ORDER, 
                 print_indices=False, clip=True):
        ''' Initialises the Zernike class and performs the calculation.'''
        
        if type(array_size) != int:
            raise TypeError('array_size must be an integer')

        # Get the cartesian and polar coordinates.
        self.x, self.y, self.r, self.theta = coordinates(array_size)

        # Calculate the number of zernike modes that will be generated.
        number_zernikes = int(sum(range(order + 1, 0, -1)))
        
        # Create an appropriately sized array for the Zernike matrix and the
        # mapping from single to double indexing.
        self.zernike_matrix = numpy.zeros((number_zernikes, 
                                           array_size, array_size))
        self.index_mapping = numpy.zeros((number_zernikes, 2), numpy.int)
        self.clip = clip

        # For each mode evaluate the Zernike polynomial, n is the radial order
        # and m is the angular order of the mode.
        for n in range(order + 1):

            # If n is odd m starts from 1, otherwise it starts from 0.
            if (-1)**n == -1:
                m_start = 1
            elif (-1)**n == 1:
                m_start = 0
            
            # (-n <= m <= n) in steps of two.
            for m in range(m_start, n+1, 2):

                # For all values of m except zero, evaluate the Zernike 
                # polymonial for negative angular order.
                if m != 0:
                    
                    # Calculate the single index value (Eq. 5).
                    index = (n*(n+2) - m) // 2
                    self.index_mapping[index] = [n, -m]
                    
                    # Evaluate the Zernike polynomial.
                    self.zernike_matrix[index] = self.calculateZernike(n, -m)
                
                # For all values of m, evaluate the Zernike polynomial for 
                # positive angular order.
                
                # Calculate the single index value (Eq. 5).
                index = (n*(n+2) + m) // 2
                self.index_mapping[index] = [n, m]
                
                # Evaluate the Zernike polynomial.
                self.zernike_matrix[index] = self.calculateZernike(n, m)
        
        # Print the index mapping if print_indices = True.    
        if print_indices:
            sys.stdout.write('Index: n, m \n')
            for ind in range(number_zernikes):
                sys.stdout.write('%2i: %2i,%2i \n' % (ind, self.index_mapping[ind, 0], 
                                        self.index_mapping[ind, 1]))

    def normalisationFactor(self, n, m):
        '''Calculate the normalisation factor for mode (n, m) (Eq. 4).
        
        Parameters
        ----------
        n : int
            The radial order.
        m : int
            The angular order.
            
        Returns
        -------
        float
            The normalisation factor.
        
        '''
        
        # Calculate the Kronecker delta value.
        if m == 0:
            delta = 1.0
        else:
            delta = 0.0
            
        # Calculate the normalisation factor (Eq. 4).
        return numpy.sqrt(2*(n+1) / (1+delta))

    def calculateZernike(self, n, m):
        ''' Create an array containing the given Zernike mode (n, m).
        
        Parameters
        ----------
        n : int
            The radial order.
        m : int
            The angular order.
            
        Returns
        -------
        array_like
            An array containing the evaluated Zernike polynomial for the
            specified mode.
        
        '''
        
        # Calculate the radial component (Eq. 3).
        R = 0.0
        
        # Calculate the upper limit for the sum in Eq. 3.
        end = ((n-abs(m))//2) + 1
        
        # Calculate the sum in Eq. 3.
        for s in range(0, end, 1):
            
            R += ((-1)**s * factorial(n - s) * self.r**(n - 2*s)) / (
            factorial(s) * factorial(((n+abs(m))//2) - s) * 
            factorial(((n-abs(m))//2) - s)) 
        
        # Calculate the normalisation factor (Eq. 4).
        N = self.normalisationFactor(n, abs(m))

        # Evaluate the Zernike polynomial (Eq. 2).
        if m < 0:
            Z = N * R * numpy.sin(abs(m) * self.theta)
        elif m > 0:
            Z = N * R * numpy.cos(abs(m) * self.theta)
        else:
            Z = N * R
            
        if self.clip == True:

            # Remove values from outside of the unit circle (as Zernikes are 
            # defined over a unit circle).
            return numpy.where(self.r > 1.0, 0.0, Z)
        else:
            return Z
        

def factorial(n):
    '''Calculate the factorial of a number, n.'''
    
    f = 1
    while n > 0:
        f *= n
        n -= 1
        
    return f

def coordinates(array_size):
    '''Generate cartesian and polar coordinates. The coordinates are normalised
    to give a unit circle (r = 1) with theta = 0 along the positive x-axis.
    
    Parameters
    ----------
    array_size : int
        The size of the array containing an evaluated Zernike polynomial,
        (i.e. number of pixels across the wavefront). The array is square.
    
    Returns
    -------
    x : array_like
        The x cartesian coordinates for the given array size.
        
    y : array_like
        The y cartesian coordinates for the given array size.
        
    r : array_like
        The radial polar coordinates for the given array size.
        
    theta : array_like
        The angular polar coordinates for the given array size.
    
    '''
    
    # Create cartesian coordinates.
    y, x = numpy.ogrid[-1.0:1.0:array_size*1j, -1.0:1.0:array_size*1j]
    
    # Invert the y coordinates so that [-1, -1] is the bottom left corner.
    y = y[::-1]
    
    # Compute the radial and angular polar coordinates
    r = numpy.sqrt(x**2 + y**2)
    theta = numpy.angle(x + y*1j)
    
    return x, y, r, theta
    
def equivalentDefocus(rms_variance, pupil_radius=PUPIL_RADIUS):
    '''Calculates the equivalent defocus in Diopters that produces the 
    wavefront variance specified by rms_variance.
    
    Parameters
    ----------
    rms_variance : float
        The rms variance of the wavefront (in microns). For a single 
        Zernike mode this the same as the Zernike coefficient in rms 
        microns.
    pupil_radius : float, optional
        The radius of the eye's pupil (in metres).
    '''
    # Eq. 6.
    diopters = numpy.sqrt(3)*4*rms_variance / ((pupil_radius*1e3) ** 2)
    return diopters
        
def wavefront(coefficients, zernike_matrix):
    '''Calculates a wavefront given a Zernike matrix and a vector of Zernike
    coefficients (in rms microns).
    
    Parameters
    ----------
    coefficients : array_like
        An array containing the Zernike coefficients, check the index
        mapping to ensure the coefficienr values match the corresponding
        modes.
    zernike_matrix : array_like
        The zernike matrix generated by the Zernike class.
        
    Returns
    -------
    wavefront : array_like
        An array containing the wavefront phase map.
        
    '''

    # Reshape the Zernike matrix to (number modes, array_size**2).
    z = numpy.reshape(zernike_matrix, (zernike_matrix.shape[0], 
                      zernike_matrix.shape[1]*zernike_matrix.shape[2]))
    
    # Matrix multiply the coefficients and the Zernike matrix (Eq. 12).
    w = numpy.dot(coefficients, z)
    
    # Reshape the result back to (array_size, array_size).
    wavefront = numpy.reshape(w, (zernike_matrix.shape[1], 
                              zernike_matrix.shape[2]))

    return wavefront

def amplitudeFunctionCircle(array_size, fraction=1.0):
    '''Create a top-hat pupil function of unit radius.
    
    Parameters
    ----------
    array_size : int
        The size of the array containing an evaluated Zernike polynomial,
        (i.e. number of pixels across the wavefront). The array is square.
    fraction : float, optional
        The fractional size of the pupil, default is 1.0. This can be used to 
        simulate stopped down pupils.
    
    Returns
    -------
    pupil : array_like
        The pupil amplitude array
        
    '''

    # Get the coordinates.
    x, y, r, theta = coordinates(array_size)
    
    # Set indices that are inside a unit circle to 1 and outside to 0.
    pupil = numpy.where(r > fraction, 0.0, 1.0)

    return pupil

def amplitudeFunctionApodised(array_size, directionality_parameter=0.05, pupil_diameter_mm=5.0):
    '''Create an apodised pupil function for modelling the optical 
    Stiles-Crawford effect.
    
   Parameters
    ----------
    array_size : int
        The size of the array containing an evaluated Zernike polynomial,
        (i.e. number of pixels across the wavefront). The array is square.
    directionality_parameter : float, optional
        The exponent of the apodisation function (wavelength-dependent 
        directionality parameter). The default value is 0.05 mm^-2.
    pupil_diameter_mm : float, optional
        The diameter of the pupi in mm. The default value is 5 mm.
        
    Returns
    -------
    pupil : array_like
        The pupil amplitude array
        
    '''
    
    # Get the coordinates.
    x, y, r, theta = coordinates(array_size)
    
    r *= 0.5 * pupil_diameter_mm
    
    return 10**(-directionality_parameter*r**2)


def checkWavefrontSampling(wavefront, wavelength=WAVELENGTH, 
                           wavefront_tolerance=WAVEFRONT_TOL, print_warning=True):
    '''Checks that the wavefront sampling is high enough by checking that the
    gradient of the wavefront between adjacent pixels is less than
    wavelength/tolerance.
    
    Parameters
    ----------
    wavefront : array_like
        An array containing the wavefront phase map.
    wavelength : float, optional
        The wavelength of the monochromatic light used (in metres).
    wavefront_tolerance : float, optional
        The criterion for the gradient of the wavefront between adjacent
        pixels is given by wavelength/tolerance. The default value is 7,
        which is twice as stringent as the Marechal criterion.
        
    Returns
    -------
    wavefront_check : float
        The maximum gradient divided by the tolerance criterion. If
        wavefront_check > 1.0, the wavefront is undersampled 
        (gradient > criterion).
        
    '''
    
    # Shift the wavefront array by one pixel in each dimension.
    shift_x = numpy.roll(wavefront, -1, axis = 1)
    shift_y = numpy.roll(wavefront, -1, axis = 0)
    
    # Calculate the gradient of the wavefront using the difference quotient.
    gradient_x = wavefront[1:-1, 1:-1] - shift_x[1:-1, 1:-1]
    gradient_y = wavefront[1:-1, 1:-1] - shift_y[1:-1, 1:-1]
    
    # Create a circular mask that is 2 pixels smaller than the aperture to 
    # remove the gradients calculated at the edges of the wavefront, where 
    # there is a discontinuity.
    circle = amplitudeFunctionCircle(wavefront.shape[0] - 2) 
    
    # Find the maximum gradient in either dimension.
    max_gradient = max([abs(gradient_x * circle).max(),
                        abs(gradient_y * circle).max()])
    
    # Check calculate the maximum gradient divided by the tolerance.
    wavefront_check = max_gradient / (wavelength/wavefront_tolerance)
    
    # If this fraction is less than one then the tolerance criterion has not 
    # been met.
    if wavefront_check > 1.0 and print_warning==True:
        sys.stdout.write('Wavefront is undersampled by a factor of %.2f \n' \
                                                             %(wavefront_check))
        sys.stdout.write('Wavefront gradient between pixels is %.2f microns \
               (lambda / %.1f) \n' %(max_gradient * 1.0e6,
                                  wavelength / max_gradient))

    return wavefront_check
    
def padArray(data, oversampling=2, value=0.0):
    '''Pad an array with zeros (or value).To generate a PSF from a wavefront 
    the array containing the wavefront (data) must be oversampled by at least 
    a factor of two. An array that is larger than the data array by a factor of
    'oversampling' is created and filled with zeros (or a given value) and the 
    data is copied into the corner of the array.
    
    Parameters
    ----------
    data : array_like
        The data array that is to be padded.
    oversampling : int
        The factor increase in size of the array to be padded. When 
        generating a PSF is the factor by which the wavefront is 
        oversampled.
    value : float, optional
        The value with which to pad the array, i.e. the value to copy
        into the additional array elements. The default is zero.
        
    Returns
    -------
    padded : array_like
        The padded array.
        
    '''
    
    # Create an array that is larger than the data array by a factor of 
    # oversampling.
    padded=numpy.zeros(numpy.dot(oversampling,data.shape).astype(int),dtype=data.dtype)+value
    
    # Put the data into the array (it doesn't need to go in the middle).
    padded[:data.shape[0],:data.shape[1]]=data[:,:]
    
    return padded    

def PSF(wavefront,amplitude,wavelength=WAVELENGTH,oversampling=OVERSAMPLING, coherent=0):
    '''Calculates the PSF from the wavefront and amplitude function (eq. 7).
    
    Parameters
    ----------
    wavefront : array_like
        An array containing the wavefront (phase) in units of rms microns
        of optical path difference.
    amplitude : array_like
        An array containing the amplitude function. For uniform pupil
        illumination this is a top-hat function with a value of one inside 
        a unit circle and zero outside.
    wavelength : float, optional
        The wavelength of the monochromatic light used (in metres).
    oversampling : int, optional
        The factor by which to oversample the wavefront. This should be
        at least 2 and preferably an even number.
        
    Returns
    -------
    psf : array_like
        An array containing the point spread function, which is normalised
        such that the sum over the whole array is equal to one.
    
    '''

    # Pad the amplitude function with zeros.
    amplitude_padded = padArray(amplitude, oversampling)
        
    # The wavefront (phase) is given in units of optical path difference,  
    # convert to radians and then pad the array.
    wavefront_radians = padArray(2.0 * numpy.pi * wavefront / wavelength, 
                                 oversampling)

    # Calculate PSF as the Fourier transform of the aperture function (Eq. 7)
    psf_amplitude = numpy.fft.fft2(amplitude_padded * numpy.exp(-1.0j * 
                                   wavefront_radians))

    if coherent == 0:
        # Multiply by complex conjugate (modulus squared to give intensity, Eq. 7)
        psf_intensity = psf_amplitude * numpy.conjugate(psf_amplitude)

        # Shift the quadrants of the array so that the DC term is in the centre and
        # take the real part (for a real image)
        psf = numpy.fft.fftshift(psf_intensity).real

    # Divide by the sum so that the total intensity is equal to 1 (i.e. we 
    # don't lose any light)
    psf /= numpy.sum(psf)
    
    return psf
    
    
def checkPsfFov(psf, psf_fraction=PSF_FRACTION, bits=N_BITS, print_warning=True):
    '''Checks the field of view of the PSF to ensure that it is not wrapped,
    which can occur if there are not enough pixeld in the wavefront array. 
    The outer fraction of the PSF array is checked such that the total 
    intensity in that area is not larger than the bit depth with which 
    you wish to display images.
    
    Parameters
    ----------
    psf : array_like
       An array containing the point spread function, which is normalised
        such that the sum over the whole array is equal to one.
    psf_fraction : float, optional
        The outer fraction of the PSF to be checked, the default value is
        0.1.
    bits : int, optional
        The bit depth of the images you wish to display, the default
        value is 8.
        
    Returns
    -------
    intensity_check : float
        The total intensity within the area being checked.
    '''
    
    # Calculate the bit depth
    intensity_tolerance = 1 / 2**bits
    
    # Create a mask within which to check the total intensity
    check_area = numpy.ones(psf.shape)
    
    # Calculate the size of the centre of the mask, which is not summed.
    check_size = int(psf.shape[0] * (1-psf_fraction)/2)
    
    # Define the area that is not checked as the central area.
    centre = psf.shape[0] / 2
    check_area[int(centre - check_size):int(centre + check_size), int(centre - check_size):
               int(centre + check_size)] = 0.0
    
    # Apply the mask such that the PSF values in the check area are multiplied
    # by one and the rest are multiplied by zero.
    check = check_area * psf
    
    # Sum the total intensity in the check area.
    intensity_check = numpy.sum(check)
    
    # If the total intensity in the check area is more than the bit depth, the
    # PSF is too close to the edge and may be wrapped.
    if intensity_check > intensity_tolerance and print_warning==True:
        sys.stdout.write('PSF field of view is too small, intensity at edges contributes \
        %i grey values to an %i-bit image \n'% (intensity_check / 
                                             intensity_tolerance, bits))
        
    return intensity_check / intensity_tolerance
    
        
def checkSampling(wavefront, amplitude, array_size, oversampling,
                  wavefront_tolerance=WAVEFRONT_TOL, psf_fraction=PSF_FRACTION, 
                  bits=N_BITS, print_warning=True):
    '''Checks that the wavefront sampling is high enough based on the 
    wavefront gradient between adjacent pixels and on the field of view 
    in the PSF.
    
    Parameters
    ----------
    wavefront : array_like
        An array containing the wavefront (phase) in units of rms microns
        of optical path difference.
    amplitude : array_like
        An array containing the amplitude function. For uniform pupil
        illumination this is a top-hat function with a value of one inside 
        a unit circle and zero outside.
    array_size : int
        Size of an individual element (mode) the Zernike matrix, i.e. the
        number of pixels across the wavefront, which should be square. It
        is preferable for array_size to be a power of two to give a padded
        array size that is also of power two (if an even number is used 
        for the oversampling factor). This is for performance reasons when 
        computing the discrete Fourier tranform.
    oversampling : int
        The factor by which to oversample the wavefront. This should be
        at least 2 and preferably an even number.
    wavefront_tolerance : float, optional
        The criterion for the gradient of the wavefront between adjacent
        pixels is given by wavelength/tolerance. The default value is 7,
        which is twice as stringent as the Marechal criterion.
    psf_fraction : float, optional
        The outer fraction of the PSF to be checked, the default value is
        0.1.
    bits : int, optional
        The bit depth of the images you wish to display, the default
        value is 8.
        
    Returns
    -------
    wavefront_check : float
        The maximum gradient divided by the tolerance criterion. If
        wavefront_check > 1.0, the wavefront is undersampled 
        (gradient > criterion).
    intensity_check : float
        The total intensity within the area being checked.

    '''
    
    # Make the PSF with the specified oversampling.
    psf = PSF(wavefront, amplitude, oversampling=oversampling)
    
    # Check the wavefront gradient.
    wavefront_check = checkWavefrontSampling(wavefront, 
                                       wavefront_tolerance=wavefront_tolerance,
                                       print_warning=print_warning)
    
    # Check the PSF field of view.
    psf_check = checkPsfFov(psf, psf_fraction=psf_fraction, bits=bits, 
                            print_warning=print_warning)
    
    return wavefront_check, psf_check

def numberPixels(fov_arcmin, wavelength =WAVELENGTH, oversampling=OVERSAMPLING, 
                 pupil_radius=PUPIL_RADIUS):
    '''Calculates the number of pixels required in the input image array for 
    the given field of view in order to match the pixel scales of the PSF 
    and the input intensity pattern.
    
    Parameters
    ----------
    fov_armin : float
        The field of view of the stimulus image (in arcminutes).
    wavelength : float, optional
        The wavelength of the monochromatic light used (in metres).
    oversampling : int, optional
        The factor by which to oversample the wavefront. This should be
        at least 2 and preferably an even number.
    pupil_radius : float, optional
        The radius of the eye's pupil (in metres).
        
    Returns
    -------
    n : int
        The number of pixels across the image to match the pixels scales.
    
    '''
        
    # Eq. 11.
    fov_radians = fov_arcmin * numpy.pi / (180*60)
    pixel_scale = psfPixelScale(wavelength=wavelength, oversampling=oversampling, pupil_radius=pupil_radius)
    n = fov_radians / pixel_scale
    
    return int(round(n))

def psfPixelScale(wavelength=WAVELENGTH, oversampling=OVERSAMPLING, 
                  pupil_radius=PUPIL_RADIUS):
    '''Calculate the pixel scale in the PSF (radians per pixel)
    
    Parameters
    ----------
    wavelength : float, optional
        The wavelength of the monochromatic light used (in metres).
    oversampling : int, optional
        The factor by which to oversample the wavefront. This should be
        at least 2 and preferably an even number.
    pupil_radius : float, optional
        The radius of the eye's pupil (in metres).      
    '''
    # Eq. 9.
    #return 1.22*wavelength / (2*oversampling*pupil_radius)
    return wavelength / (2*oversampling*pupil_radius)
    



