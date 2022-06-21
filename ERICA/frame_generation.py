#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 19:47:48 2021

@author: laurayoung
"""

import numpy
import sys

from PIL import Image


from ERICA import aberration_rendering, ERICA_toolkit


def createPinhole(psf_size, pinhole_add, oversampling, ph_scale_up = 10.):
    '''Creates a pinhole mask.
    
    Parameters
    ----------
    psf_size : tuple
        A tuple containing the size for the pinhole mask (should match the size
        of the PSF array).
    pinhole_add : float
        The diameter of the confocal pinhole in units of Airy disk diameters.
    oversampling : float
        The oversampling factor used when computing the PSF.
    ph_scale_up : float, optional
        The upsampling factor to use when generating the pinhole. The result
        will be downsampled to match psf_size using an antialiasing filter.
    
    Returns
    -------
    pinhole : array_like
        An array containing the pinhole mask (1 in side the pinhole, 0 outside)
    
    '''    
    
    # Calculate the radius of the pinhole in pixels:
    # 1 ADD = 1.22 * wavelength / (2 * pupil_radius)
    # Therefore, pinhole radius (radians) = 1.22 * wavelength / (2 * pupil_radius)) * pinhold_add * 0.5
    # PSF pixel scale (radians) = wavelength / (2 * oversampling * pupil_radius)
    # So....
    pinhole_radius_pixels = 0.5 * pinhole_add * oversampling * 1.22

    # Define the centre of the pinhole array
    cenx_box = numpy.floor(psf_size[1] / 2.)
    ceny_box = numpy.floor(psf_size[0] / 2.)
    
    # Define radial coordinates, oversampled by a factor ph_scale_up
    y, x = numpy.ogrid[0:psf_size[0]-1:psf_size[0]*ph_scale_up*1j, 0:psf_size[1]-1:psf_size[1]*ph_scale_up*1j]
    r = numpy.sqrt((x-cenx_box)**2 + (y-ceny_box)**2)
    
    # Create a uniform circular mask with a radius equal to the pinhole size
    pinhole_scale_up = numpy.where(r <= pinhole_radius_pixels, 1., 0.)
    
    # Downsample it using an antialising filter to match the size of the PSF array
    pinhole_downsampled = numpy.asarray(Image.fromarray(pinhole_scale_up).resize(psf_size, Image.ANTIALIAS))
    
    # Clip the values so that they remain in the range 0->1
    pinhole = numpy.clip(pinhole_downsampled, 0., 1.)
    
    return pinhole

def computePSF(image_size, pixel_size_arcmin, wavelength, pupil_diameter, pinhole_add, zernike_coefficients=None, threshold=1/256., amplitude_function=aberration_rendering.amplitudeFunctionCircle, **kwargs):
    
    '''Adds the effects of aberrations and diffraction to an image - this 
    assumes that the point spread function (PSF) of the double pass system is the 
    autocorrelation of the single pass PSF.
    
    Parameters
    ----------
    image : array_like
        An array containing the image to add diffraction effects to.
    pixel_size_arcmin : float
        The size of each pixel in arcminutes. Normally stored in the
        system parameters file.
    wavelength : float
        The wavelength of the imaging light in metres. Normally stored in the
        system parameters file.
    pupil_diameter : float
        The diameter of the eye's pupil in metres. Normally stored in the
        system parameters file.
    pinhole_add : float
        The diameter of the confocal pinhole in units of Airy disk diameters.
    amplitude_function : function, optional
        The function to use for the pupil amplitude function. The default is a
        uniform circular pupil. You can pass your own function as an argument, 
        e.g. to replicate the Stiles-Crawford effect. This function should have
        a single input argument specifying the size of the (square) array and 
        should return an array containing the pupil amplitude function.

    Returns
    -------
    image_diffraction : array_like
        An array containing the image with the effects of diffraction added.
        
    '''
    
    # Convert the pixel size from arcminutes to radians
    pixel_scale_radians = numpy.pi*numpy.mean(pixel_size_arcmin)/(180*60.)
    
    # calculate the oversampling value required to sample the PSF with the specified pixel size
    oversampling = wavelength / (pixel_scale_radians * pupil_diameter)
    
    # The oversampling value should be at least two. The computation will continue but a warning will be printed.
    if oversampling<2.0:        
        sys.stdout.write('Warning: The image is undersampled (oversampling = %.2f), trying using a longer wavelength, smaller pixel size or smaller pupil diameter \n'%oversampling)

    # Calculate the required array size for the wavefront data
    array_size = int(max(image_size)/numpy.floor(oversampling))
    
    # Compute the pupil amplitude function
    pupil_amplitude_function = amplitude_function(array_size, **kwargs)
    
    if zernike_coefficients is None:
        
        # Compute the pupil phase function - an array of zeros for a diffraction-limited system
        wavefront = numpy.zeros((pupil_amplitude_function.shape))
    
    else:
        # Check what radial order is specified for the Zernike coefficients
        n_zernikes = zernike_coefficients.shape[0]
        
        # Note, this comes out with a value 1 higher because of zero indexing        
        radial_orders_plus_1 = 0
        while n_zernikes > 0:
            radial_orders_plus_1 += 1
            n_zernikes -= radial_orders_plus_1
        
        # Check if the number of zernikes matches the number for this order.
        n_zernikes_for_this_radial_order = sum(range(radial_orders_plus_1 + 1))
        if n_zernikes_for_this_radial_order != zernike_coefficients.shape[0]:
            raise ValueError('The number of zernike coefficients is not compatible with any radial order. Note that piston, tip and tilt are included so e.g. there are 21 Zernike coefficients up to 5th radial order')
            
        radial_orders = radial_orders_plus_1 - 1
        
        # Generate the matrix of Zernike polynomial data
        zernike_matrix = aberration_rendering.Zernike(array_size=array_size, order=radial_orders).zernike_matrix

        # Compute the pupil phase function
        wavefront = aberration_rendering.wavefront(zernike_coefficients, zernike_matrix)
    
    # Compute the single-pass PSF
    psf_single_pass_in = aberration_rendering.PSF(wavefront, pupil_amplitude_function, oversampling=oversampling, wavelength=wavelength)
    
    
    return psf_single_pass_in, oversampling

def confocalPSF(psf_single_pass_in, pinhole_add, oversampling, psf_single_pass_out=None):
    '''Computes the confocal PSF. See T. Wilson (2011). Resolution and optical 
    sectioning in the confocal microscope, Journal of Microscopy, Vol. 244, 
    Pt 2 2011, pp. 113–121.
    
    Parameters
    ----------
    psf_single_pass_in : array_like
        An array containing the single pass PSF (into the eye). This is H1 in
        Wilson (2011), where H1 = |h1|^2
    pinhole_add : float
        The diameter of the confocal pinhole in units of Airy disk diameters.
    oversampling : float
        The oversampling factor used when computing the PSF.

    Returns
    -------
    psf_confocal: array_like
        An array containing the confocal PSF.
    pinhole: array_like
        An array containing the pinhole mask.
    '''
    
    
    # Create the pinhole mask. This is D in Wilson (2011).
    pinhole = createPinhole(psf_single_pass_in.shape, pinhole_add, oversampling)
    
    # If not specified, compute the output PSF. This is H2 in Wilson (2011)
    # where H2 = |h2|^2
    if psf_single_pass_out is None:
        # The output PSF is flipped horizontally and vertically. 
        psf_single_pass_out = psf_single_pass_in[::-1,::-1]
    
    # Compute the output PSF. This is H2eff in Wilson (2011), where H2eff = |h2eff|^2 (Equation 17)
    psf_out = ERICA_toolkit.convolve2D(psf_single_pass_out, pinhole)
    # Compute the confocal PSF.
    psf_confocal = psf_single_pass_in * psf_out
    
    return psf_confocal, pinhole


def diffractionOnly(image, pixel_size_arcmin, wavelength, pupil_diameter, pinhole_add, amplitude_function=aberration_rendering.amplitudeFunctionCircle, **kwargs):
    '''Adds the effects of diffraction to an image - this assumes that the 
    point spread function (PSF) of the double pass system is the 
    autocorrelation of the single pass PSF.
    
    Parameters
    ----------
    image : array_like
        An array containing the image to add diffraction effects to.
    pixel_size_arcmin : float
        The size of each pixel in arcminutes. Normally stored in the
        system parameters file.
    wavelength : float
        The wavelength of the imaging light in metres. Normally stored in the
        system parameters file.
    pupil_diameter : float
        The diameter of the eye's pupil in metres. Normally stored in the
        system parameters file.
    pinhole_add : float
        The diameter of the confocal pinhole in units of Airy disk diameters.
    amplitude_function : function, optional
        The function to use for the pupil amplitude function. The default is a
        uniform circular pupil. You can pass your own function as an argument, 
        e.g. to replicate the Stiles-Crawford effect. This function should have
        a single input argument specifying the size of the (square) array and 
        should return an array containing the pupil amplitude function.

    Returns
    -------
    image_diffraction : array_like
        An array containing the image with the effects of diffraction added.
        
    '''
    
    psf_single_pass_in, oversampling = computePSF(image.shape, pixel_size_arcmin, wavelength, pupil_diameter, pinhole_add, zernike_coefficients=None, amplitude_function=amplitude_function, **kwargs)
    
    # Create the PSF and pinhole
    psf, pinhole = confocalPSF(psf_single_pass_in, pinhole_add, oversampling, psf_single_pass_out=None)
    
    # Make sure that the field of view of the PSF matches the field of view of the image. 
    # Crop such that the centre of the PSF remains in the centre so that the convolve image is not displaced.

    psf_use = numpy.copy(psf)
    if psf_use.shape[0] < image.shape[0]:
        psf_size = numpy.copy(psf_use.shape[0])
        psf_use_ = numpy.zeros((image.shape[0], psf_use.shape[1]))
        psf_use_[:psf_use.shape[0]] = psf_use
        psf_use = numpy.roll(psf_use,int(numpy.ceil((image.shape[0] - psf_size) * 0.5)),axis=0)
        psf_use = numpy.copy(psf_use_)
    elif psf_use.shape[0] >= image.shape[0]:        
        psf_use = psf_use[int(numpy.floor(psf_use.shape[0] / 2.) - numpy.floor(image.shape[0] / 2.)):int(numpy.floor(psf_use.shape[0] / 2.) + numpy.ceil(image.shape[0] / 2.)),:]

        
    if psf_use.shape[1] < image.shape[1]:
        psf_size = numpy.copy(psf_use.shape[1])
        psf_use_ = numpy.zeros((psf_use.shape[0], image.shape[1]))
        psf_use_[:,:psf_use.shape[1]] = psf_use
        psf_use = numpy.roll(psf_use,int(numpy.ceil((image.shape[0] - psf_size) * 0.5)),axis=0)
        psf_use = numpy.copy(psf_use_)

    elif psf_use.shape[1] >= image.shape[1]:
        psf_use = psf_use[:,int(numpy.floor(psf_use.shape[1] / 2.) - numpy.floor(image.shape[1] / 2.)):int(numpy.floor(psf_use.shape[1] / 2.) + numpy.ceil(image.shape[1] / 2.))]

    # Perform convolution of image with confocal PSF
    image_diffraction = ERICA_toolkit.convolve2D(image, psf_use)

    return image_diffraction


