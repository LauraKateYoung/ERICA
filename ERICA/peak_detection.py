''' 
Peak detection
-------------------
Functions for finding characterising cells within ground truth retinae. These
functions are designed to detect high contrast peaks that all have a similar 
amplitude, such as output by the reaction-diffusion model. It has not been 
tested for use on synthetic or real AOSLO images where cells may vary in
reflectance or where images may contain noise or residual blur.
'''

import numpy

from scipy import ndimage
from sklearn.neighbors import NearestNeighbors
from skimage.feature import peak_local_max 
from scipy.ndimage.measurements import center_of_mass
from skimage.segmentation import watershed


from ERICA import ERICA_toolkit


def findNeighbours(peaks, number_neighbours=6):
    '''Finds the nearest neighbours of each peak (cones or reaction diffusion 
    spots) using sklearn.
    
    Parameters
    ----------
    peaks : array_like
        An (N, 2) array containing the locations of the peaks  where the
        coordinates for each of the N peaks is given as (y,x).
    number_neighbours : int
        The number of nearest neighbours to look for.

        
    Returns
    -------
    distances : array_like
        An (N, number_neighbours+1) array containing the distance between each
        of the N peaks and its neighbours. The peaks itself is included as the
        first index in each row.
    indices : array_like
        An (N, number_neighbours+1) array containing the indices of each of the
        N peaks and its neighbours. The peaks itself is included as the
        first index in each row.
        
    '''
    
    # Find the nearest neighbours
    neighbours = NearestNeighbors(n_neighbors=number_neighbours+1, algorithm='ball_tree').fit(peaks)
    
    # Calculate the distances to the neighbours
    distances, indices = neighbours.kneighbors(peaks)
    
    return distances, indices

def removeEdgePeaks(peaks, gap_size):
    '''Removes peaks from the edges of the defined area.
    
    Parameters
    ----------
    peaks : array_like
        An (N, 2) array containing the locations of the peaks  where the
        coordinates for each of the N peaks is given as (y,x).
    gap_size : float
        The boundary around the edge of the area of peaks from which to remove
        peaks.

        
    Returns
    -------
    peaks_no_edges : array_like
        An (N', 2) array containing containing the locations of the peaks, 
        where the coordinates for each of the N' peaks is given as (y,x) and N'
        is the number of peaks not considered to be at the edge.
    indices_no_edges : array_like
        A size N' array containing containing the indices in the original 
        'peaks' array of the peaks not considered to be at the edge.
        
    '''
    indices_no_edges = numpy.where((peaks[:,0]>peaks[:,0].min()+gap_size) 
                    & (peaks[:,0]<peaks[:,0].max()-gap_size) 
                    & (peaks[:,1]>peaks[:,1].min()+gap_size) 
                    & (peaks[:,1]<peaks[:,1].max()-gap_size))[0]
    peaks_no_edges = peaks[indices_no_edges]

    return peaks_no_edges, indices_no_edges

def findUniformPeaks(image):
    '''Detects peaks that have uniform amplitude. May not work well for peaks
    with variable amplitude.
    
    Parameters
    ----------
    image : array_like
        An 2D array containing the intensity data in which to detect peaks.
   
    Returns
    -------
    centroids : array_like
        An (N,2) array containing the locations of the peaks as (y,x) in pixels.
    estimated_spacing : float
        The estimated spacing between the peaks, given in pixels.
        
    '''
    
    # Estimate the spacing between peaks. If the array is larger than 500 x 500
    # just use the top 500 x 500 corner to save processing time.
    if image.shape[0]>500 and image.shape[1]>500:
        estimated_spacing = estimateSpacingUniformPeaks(image[:500,:500])
    else:
        estimated_spacing = estimateSpacingUniformPeaks(image)
    
    # Detect peaks in the binarised image, specifying the minimum distance 
    # Between peaks as half the estimated peak spacing.
    minimum_separation = int(numpy.floor(estimated_spacing * 0.5))
    max_filter_output = peak_local_max(image, indices=False, min_distance=minimum_separation)
    
    # Label the regions around each of the peaks
    markers = ndimage.label(max_filter_output)[0]
    labels = watershed(-image, markers, mask=image)
    
    # Find the centroid of each region
    centroids = numpy.asarray(center_of_mass(image, labels, range(1, numpy.max(labels)+1)))

    
    return centroids, estimated_spacing

def estimateSpacingUniformPeaks(image):
    '''Estimates the average spacing between peaks of uniform amplitude in the
    given image. If the image is not square it will be cropped to make it 
    square.
    
    Parameters
    ----------
    image : array_like
        An 2D array containing the intensity data in which to detect the
        spacing of peaks. Should be square or will be cropped.
   
    Returns
    -------
    estimated_spacing : float
        The estimated spacing between the peaks, given in pixels.
        
    '''
    
    # Make sure that the image is square, if not crop it so that it is
    if image.shape[0] > image.shape[1]:
        image_square = image[:image.shape[1]]
    elif image.shape[1] > image.shape[0]:
        image_square = image[:,:image.shape[0]]
    else:
        image_square = image
        
    image_square = image_square.astype(float)
    
    # Remove mean (DC component)
    image_square -= numpy.mean(image_square)
    
    # Compute the Fourier transform
    ft_image = abs(numpy.fft.fftshift(numpy.fft.fft2(image_square)))
    
    # Compute the radial profile of the Fourier transform, summed over all angles.
    radial_profile, radial_values = ERICA_toolkit.radialProfileSum(ft_image)
    
    # Find the peak in the Fourier transform (estimate of Yellot's ring)
    peak_location_index = numpy.where(radial_profile >= numpy.nanmax(radial_profile) * 0.99)[0][0]
    
    peak_location = radial_values[peak_location_index]
    
    # Convert the peak frequency to spacing
    estimated_spacing = image_square.shape[0]/float(peak_location)
    
    return estimated_spacing

 