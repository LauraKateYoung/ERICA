''' 
Generate retina
-------------------
Functions to generate a retina parameter array defining a synthetic retina
and containing the ground truth data.
'''

import numpy
import os
import sys

from scipy.special import erf

from ERICA import ERICA_toolkit, peak_detection

####################
# GLOBAL VARIABLES #
####################

# CONE SPACING
# Curcio et al (1990) J. Comp. Neuro., Vol. 292, 497-523 - Figure 6, raw data taken from:
# Andrew B. Watson; A formula for human retinal ganglion cell receptive field density as a function of visual field location. Journal of Vision 2014;14(7):15. doi: 10.1167/14.7.15.
# From Watson (2014): "We order the meridians temporal, superior, nasal, inferior, consistent with increasing polar angle in the visual field of the right eye, and assign them indexes of 1 to 4"
CONE_DENSITY_RAW = numpy.loadtxt(os.path.abspath(os.path.normpath(os.path.join('Data', 'Curcio_1990_Watson.csv'))), delimiter=',')

# Refractive indices taken from Meadway A and Sincich LC (2019), "Light reflectivity and interference in cone photoreceptors", Biomed. Opt. Express. 10(12):6531-6554
# Original data in A. W. Snyder and C. Pask (1973), "The  Stiles-Crawford effect-explanation and consequences", Vis. Res. Vol. 13, 1115-1137.
# Refractive index of inner segment
NI = 1.378
# Refractive index of outer segment
NO = 1.39
# Refractive index of surrounding medium
NS = 1.34

def fitCurcioData():
    ''' Fits a power law curve to cone spacing data, derived from Curcio 1990. 
    For simplitity, cone densities are averaged across the four meridians and 
    then converted to spacings using Williams and Coletta(1987)
    (J. Opt. Soc. Am. A, Vol. 4, No. 8 - Note 23).
    
    Returns
    -------
    fit_parameters : array_like
        The parameters (p) of the power-law fit 
        (p[0] * (x + p[1]) ** p[2] + p[3])
    position_deg : array_like
        The positions of the measurements made by Curcio in degrees
    average_cone_density : array_like
        The density of cones, averaged over the four meridians, in cones per mm
        as measured by Curcio.
    cone_spacing_um : array_like
        The spacing of cones in microns, calculated based on the 
        average_cone_density.
    
    '''
    
    # Curcio/Watson data are given in visual field, so we swap them relative to the description above.
    temporal_retina = numpy.where(CONE_DENSITY_RAW[4] !=0, CONE_DENSITY_RAW[4], numpy.nan)
    nasal_retina = numpy.where(CONE_DENSITY_RAW[2] !=0, CONE_DENSITY_RAW[2], numpy.nan)
    superior_retina = numpy.where(CONE_DENSITY_RAW[5] !=0, CONE_DENSITY_RAW[5], numpy.nan)
    inferior_retina = numpy.where(CONE_DENSITY_RAW[3] !=0, CONE_DENSITY_RAW[3], numpy.nan)
    position_deg = CONE_DENSITY_RAW[1]
    
    # Take the average across meridians
    average_cone_density = numpy.nanmean([temporal_retina, nasal_retina, superior_retina, inferior_retina],axis=0)
    
    # Calculate spacings
    cone_spacing_um = numpy.where(average_cone_density == 0.0, numpy.nan, 1000 * 1.0 / numpy.sqrt(2 * average_cone_density / numpy.sqrt(3.0)))
    
    # Remove any NaNs
    cone_spacing_um_not_nan = numpy.where(numpy.isnan(cone_spacing_um) == False)[0]
    cone_spacing_um = cone_spacing_um[cone_spacing_um_not_nan]
    position_deg = position_deg[cone_spacing_um_not_nan]

    # Fit the data with a power-law function
    fit_parameters = ERICA_toolkit.fitCurcio(position_deg, cone_spacing_um)
    
    return fit_parameters, position_deg, average_cone_density, cone_spacing_um

# Fit the Curcio data and determine the minimum cone spacing - this could be saved to file rather than ran each time the module loads.
CurcioFIT, CONE_POSITIONS, CONE_DENSITIES, CONE_SPACING_UM = fitCurcioData()
MIN_SPACING_UM = numpy.nanmin(CONE_SPACING_UM)

##########################
# CONE SPACING FUNCTIONS #
##########################


def spacingFromEccentricity(eccentricity_deg):
    '''Computes the expected cone spacing in microns for a given eccentricity
    in degrees, based on the Curcio et al. (1990) data and using linear 
    interpolation
    
    Parameters
    ----------
    eccentricity_deg : float
        The eccentricity in degrees
        
    Returns
    -------
    spacing : float
        The expected cone spacing
    
    '''

    spacing = ERICA_toolkit.functCurcio(numpy.sqrt(eccentricity_deg[0] ** 2 + eccentricity_deg[1] ** 2), CurcioFIT)
    
    return spacing


def foveatePoints(points, microns_per_degree, n_iter=5, n_neighbours=3, eccentricity=[0.0, 0.0]):
    '''Takes (approximately) evenly spaced points and scales them to match the 
    radial cone spacing profile calculated from Curcio et al. (1990) 
    
    Parameters
    ----------
    points : array_like
        An array containing the evenly spaced points - typically the output of 
        the reaction diffusion model.
    microns_per_degree : float
        The number of microns per degree on the retinal surface - stored in the
        system parameters file
    n_iter : int, optional
        To get the cone spacings to match histology their positions are
        iteratively rescaled for their eccentricity. This defines the number
        of iterations to run through. Previous testing shows that after 5
        iterations the positions of the cones do not change significantly.
    n_neighbours : int, optional
        The number of nearest neighbours to include in the calculation of cone 
        spacing. The default is 3.
        
    Returns
    -------
    foveated_points : array_like
        A transposed array of the locations of the points, scaled radially.
    '''
    
    # First rescale the points so that they all have the minimum cone spacing in microns.
    rescaled_points = scaleRetina(numpy.transpose(points), [0.0, 0.0], microns_per_degree)
    
    
        
    # Calculate the spacings of the cones
    original_spacing = determineSpacing(numpy.transpose(rescaled_points), n_neighbours=n_neighbours)
    
    # Ignore any cones at the edge of the mosaic and calculate the average cone spacing
    no_edge_cones, idx = peak_detection.removeEdgePeaks(numpy.transpose(rescaled_points), numpy.max(original_spacing))
    original_spacing = numpy.mean(original_spacing[idx])
    
    rescaled_points[0] += eccentricity[0] * microns_per_degree
    rescaled_points[1] += eccentricity[1] * microns_per_degree

    # Rescale the cone positions to achieve the required cone spacing    
    previous_points = numpy.copy(rescaled_points)
    previous_spacing = original_spacing
    for i in range(n_iter):
        # Calculate the eccentricity of each cone
        eccentricities_um = numpy.sqrt(previous_points[0] ** 2 + previous_points[1] ** 2)
    
        # Convert the eccentricity of each cone in microns to degrees
        eccentricities_deg = eccentricities_um / microns_per_degree
        
        # Calculate the amount to scale each cones position to give the required cone spacing
        amount_to_move_points = ERICA_toolkit.functCurcio(eccentricities_deg, numpy.copy(CurcioFIT)) / previous_spacing
            
        # make a copy of the original set of cone locations and scale those positions as calculated above
        new_points = numpy.copy(previous_points)
        
        new_points[0] *= amount_to_move_points
        new_points[1] *= amount_to_move_points
        
        # Copy these for the next iteration
        previous_points = numpy.copy(new_points)
        # Calculate the eccentricity of each cone
        eccentricities_um1 = numpy.sqrt(new_points[0] ** 2 + new_points[1] ** 2)
    
        # Convert the eccentricity of each cone in microns to degrees
        eccentricities_deg1 = eccentricities_um1 / microns_per_degree
        
        # Calculate the new cone spacings based on a fit to the new data points
        spacings = determineSpacing(numpy.transpose(new_points), n_neighbours=n_neighbours)
        ec1 = eccentricities_deg1[::100]
        sp1 = spacings[::100]
        fit_parameters = ERICA_toolkit.fitCurcio(ec1, sp1, p0=numpy.copy(CurcioFIT))
        new_spacing = ERICA_toolkit.functCurcio(eccentricities_deg1, fit_parameters)
        previous_spacing = numpy.copy(new_spacing)


    # Return the final iteration
    foveated_points = new_points
    
    return foveated_points


def determineSpacing(points, n_neighbours=3):
    '''Estimates the spacing of cones. This is used throughout ERICA and so 
    changing or replacing this function allows the global method of cone 
    spacing estimation to be updated.
    
    Parameters
    ----------
    points : array_like
        An array containing points or cones.
    n_neighbours : int, optional
        The number of nearest neighbours to include in the calculation. The
        default is 3.
    
    Returns
    -------
    spacing : array_like
        An array of the spacings of each of the points from its neighbours.
     
    '''
    
    # Calculate the distance between each cone and its n nearest neighbours
    distances, indices = peak_detection.findNeighbours(points, number_neighbours=n_neighbours)
    
    # Average the distance between its neigbours
    spacing = numpy.nanmean(distances[:,1:], axis=1)

    return spacing

def scaleRetina(points, eccentricity, microns_per_degree):   
    ''' Rescales the locations of evenly spaced points to match the Curcio et 
    al. (1990) cone spacing for the given eccentricity.
    
    Parameters
    ----------
    points : array_like
        An array containing the evenly spaced points - typically the output of 
        the reaction diffusion model.
    eccentricity : float
        The specified eccentricity
    microns_per_degree : float
        The number of microns per degree on the retinal surface - stored in the
        system parameters file
        
    Returns
    -------
    rescaled : array_like
        An array of the locations of the points, scaled for the given
        eccentricity.
        
    '''
    
    # Centre on the average position
    points[0] -= numpy.mean(points[0])
    points[1] -= numpy.mean(points[1])
    
    # Calculate the cone spacing - average of three nearest neighbours
    dist, ind = peak_detection.findNeighbours(numpy.transpose(points), number_neighbours=6)
    spacing = numpy.nanmean(numpy.nanmedian(dist[:,1:4],axis=1))

    # Calculate the expected spacing based on the specified eccentricity
    new_spacing = spacingFromEccentricity(eccentricity)
    
    # Rescale the cone locations to match the required spacing
    rescaled = new_spacing * numpy.copy(points)/spacing

    rescaled[0] += eccentricity[0]*microns_per_degree
    rescaled[1] += eccentricity[1]*microns_per_degree

    return rescaled

def centreFovea(points):
    ''' Shifts the locations of the cones such that the centre (0.0, 0.0) is at the
    location of lowest cone spacing
    
    Parameters
    ----------
    points : array_like
        An array containing the location of the cones - these points should be
        foveated
    
    Returns
    -------
    centred_points : array_like
        An array of the locations of the points, shifted.
        
    '''
    
    # Calculate the median spacing for each cone from its three nearest 
    # neighbours
    distances, indices = peak_detection.findNeighbours(numpy.transpose(points), number_neighbours=6)
    spacing = numpy.median(distances[:,1:4],axis=1)
    
    # Find the cone with the lowest spacing
    centre = numpy.argmin(spacing)
    
    # Subtract that cone's location from all other cone positions, so that it
    # is now at the centre (0.0, 0.0) of the mosaic
    
    centred_points = numpy.copy(points)
    centred_points[0] -= points[0,centre]
    centred_points[1] -= points[1,centre]
    
    return centred_points


#######################
# CONE SIZE FUNCTIONS #
#######################
    

def innerSegmentDiam(eccentricity):
    '''Calculates the cone inner segment diameter as a function of eccentrricity 
    (degrees). This is based on Equation 3 in Tyler (1985) "Analysis of visual 
    sensitivity II Peripheral retina and the role of photoreceptor dimensions", 
    J. Opt. Soc. Am. A, Vol. 2, No 3., pgs 393 - 398. This in turn is based on
    data from Polyak (1941), "The Retina", The University of Chicago Press, 
    Illinois.
    
    Parameters
    ----------
    eccentricity : float
        The eccentricity of the cone
    '''
    
    return 2.5 * (eccentricity + 0.2) ** (1 / 3.0)
    
def outerSegmentDiam(eccentricity):
    '''Calculates the cone outer segment diameter as a function of eccentrricity 
    (degrees). This is based on Equation 4 in Tyler (1985) "Analysis of visual 
    sensitivity II Peripheral retina and the role of photoreceptor dimensions", 
    J. Opt. Soc. Am. A, Vol. 2, No 3., pgs 393 - 398. This in turn is based on
    data from Polyak (1941), "The Retina", The University of Chicago Press, 
    Illinois.
    
    Parameters
    ----------
    eccentricity : float
        The eccentricity of the cone
    '''
    
    return 1.4 * (eccentricity + 0.2) ** (1 / 5.0)

def vNumber(radius, wavelength, ni=NI, ns=NS):
    '''Calculates the mode number of a step-index optical fibre based on its 
    core radius, the refractive indices and the wavelength of light.
    
    Parameters
    ----------
    radius : float
        The radius of the core of the optical fibre
    wavelength : float
        The wavelength of the imaging light - stored in the system parameters 
        file
    ni : float (optional)
        The refractive index inside the fibre. The default value is defined by
        the global variable NI = 1.378 (Meadway and Sincich, 2019) - the 
        refractive index of the inner segment.
    ns : float (optional)
        The refractive index media surrounding the fibre. The default value is 
        defined by the global variable NS = 1.34 (Meadway and Sincich, 2019) -
        the refractive index of the surrounding medium.
    '''
    
    return 2 * numpy.pi * radius * numpy.sqrt(ni ** 2 - ns ** 2) / wavelength

def modeDiameter(radius, wavelength, ni=NI, ns=NS, pm=0):
    
    '''Calculates the mode diameter of a step-index optical fibre based on its 
    core radius, the refractive indices and the wavelength of light.
    
    Parameters
    ----------
    radius : float
        The radius of the core of the optical fibre
    wavelength : float
        The wavelength of the imaging light - stored in the system parameters 
        file
    ni : float (optional)
        The refractive index inside the fibre. The default value is defined by
        the global variable NI = 1.378 (Meadway and Sincich, 2019) - the 
        refractive index of the inner segment.
    ns : float (optional)
        The refractive index media surrounding the fibre. The default value is 
        defined by the global variable NS = 1.34 (Meadway and Sincich, 2019) -
        the refractive index of the surrounding medium.
    pm : int (optional)
        A flag to indicate whether to use (1) or not use (0) the Petermann II 
        approximation. The default is 0 (use the unmodified Marcuse equation).
        
    Returns
    -------
    mode_stdev : float
        The standard deviation of the Gaussian (LP01 mode) intensity profile.
    '''
    
    # Marcuse equation
    v = vNumber(radius, wavelength=wavelength, ni=ni, ns=ns)
    
    if pm == 0:
        
        rad_factor  = 0.65 + (1.619 / (v**(3/2.0))) + (2.879 / (v**6))
    
    else:
        rad_factor  = 0.65 + (1.619 / (v**(3/2.0))) + (2.879 / (v**6)) - (0.016 + (1.561/(v**7)))
    
    # Calculate the 1/e^2 mode radius using the Marcuse equation
    mode_radius = rad_factor * radius
    
    # Convert 1/e^2 to FWHM
    # mode_FWHM = mode_radius * numpy.sqrt(2 * numpy.log(2.0))

    # Convert 1/e^2 to standard deviation
    mode_stdev = mode_radius  / 2.0
    
    return mode_stdev

def coneDiameter(eccentricities, microns_per_degree, wavelength, ni=NI, ns=NS, pm=0):
    '''Determines the expected cone size (standard deviation of its Gaussian
    reflectance profile) based on the Marcuse equation, with the 'fibre core' 
    diameter taken to be the cone inner segment diameter (Polyak, 1941)
    
    Parameters
    ----------
    eccentricities : array_like
        An array containing the eccentricities of the cones
    microns_per_degree : float
        The number of microns per degree on the retinal surface - stored in the
        system parameters file
    wavelength : float
        The wavelength of the imaging light - stored in the system parameters 
        file.
    ni : float (optional)
        The refractive index inside the fibre. The default value is defined by
        the global variable NI = 1.378 (Meadway and Sincich, 2019) - the 
        refractive index of the inner segment.
    ns : float (optional)
        The refractive index media surrounding the fibre. The default value is 
        defined by the global variable NS = 1.34 (Meadway and Sincich, 2019) -
        the refractive index of the surrounding medium.
    pm : int (optional)
        A flag to indicate whether to use (1) or not use (0) the Petermann II 
        approximation. The default is 0 (use the unmodified Marcuse equation).
    
    Returns
    -------
    interpolated_cone_density : array_like
        An array of interpolated cone density values.
        
    '''
    
    # Calculate the radius of the cone inner segment
    cone_radius = innerSegmentDiam(eccentricities) * 0.5
    
    # Calculate the mode diameter
    cone_size_stdev = modeDiameter(cone_radius*1e-6,wavelength, ni=ni, ns=ns, pm=pm)*1e6
    
    # At very low eccentricities the Marcuse equation produces unreliable 
    # results due to the small cone size and low V number. We set the size of 
    # those cones to be fixed at the minimum value.
    min_valid_eccentricity = eccentricities[numpy.argmin(cone_size_stdev)]
    valid_cone_size_stdev = numpy.where(eccentricities<=min_valid_eccentricity,cone_size_stdev.min(), cone_size_stdev)
    
    return valid_cone_size_stdev


####################################
# RETINA PARAMETER ARRAY FUNCTIONS #
####################################
    
def generateRetinaFromEccentricity(points, eccentricity, field_of_view, microns_per_degree, wavelength, foveate=0, reflectance_map=None, width_variation=None, mean_reflectance=None, std_reflectance=None, skew_location=None, skew_scale=None, skew_alpha=None, skew_amplitude=None, skew_offset=None, distribution='normal'):
    '''Creates an array of [cone y pos, cone x pos, cone x width (stdev), 
    cone y width (stdev), rotation angle (radians), reflectance] from the
    non-scaled positions (points) and the specified eccentricity.
    
    Parameters
    ----------
    points : array_like
        An array containing the locations of the spots, typically the output
        of the reaction diffusion model, which will be rescaled for a given 
        retinal eccentricity.
    eccentricty : array_like
        An array if size (2), giving the retinal eccentricity. Note that cone
        locations and sizes are scaled for a fixed eccentricity.
    microns_per_degree : float
        The number of microns per degree on the retinal surface - stored in the
        system parameters file
    reflectance_map : array_like (optional)
        A 2D amplitude map specifying the cone reflectance as a function of position on
        the retina. The default is to select cone reflectances from a normal
        distribution with mean mean_amp and standard deviation std_amp. Passing
        a 2D map will, instead calculate the average reflectance value at the
        cone's location in the amplitude map.
    width_variation : float (optional)
        To add some variability to the sizes of the cones, specify a value > 0.
        The cone sizes specified in cone_sizes_um are each multiplied by a 
        factor. For each cone, this factor is selected from a normal 
        distribution with a mean of 1.0 and a standard deviation of 
        width_variation The default is 0.0, i.e. no variability.
    mean_reflectance : float (optional)
        Cone reflectances are selected from a normal distribution, this 
        specifies the mean of that distribution. 
    std_reflectance : float (optional)
        Cone reflectances are selected from a normal distribution, this 
        specifies the standard deviation of that distribution. 
        
        
    Returns
    -------
    retina_params_microns : array_like
        The retina parameter array with spatical dimensions of microns
        
    '''
    if foveate == 1:
        foveated_points = foveatePoints(points, microns_per_degree)
        rescaled_cone_positions = centreFovea(foveated_points)
    else:
        rescaled_cone_positions = scaleRetina(numpy.transpose(points), eccentricity, microns_per_degree)
    
    cone_positions_window = rescaled_cone_positions[:,numpy.where(
    (rescaled_cone_positions[0] > (eccentricity[0] - 0.5 * field_of_view[0]) * microns_per_degree) 
    & (rescaled_cone_positions[0] < (eccentricity[0] + 0.5 * field_of_view[0]) * microns_per_degree) 
    & (rescaled_cone_positions[1] > (eccentricity[1] - 0.5 * field_of_view[1]) * microns_per_degree) 
    & (rescaled_cone_positions[1] < (eccentricity[1] + 0.5 * field_of_view[1]) * microns_per_degree))[0]]
    
    eccentricities = numpy.sqrt(cone_positions_window[0]**2 + cone_positions_window[1]**2) / microns_per_degree
    cone_sizes = coneDiameter(eccentricities, microns_per_degree, wavelength)
    cone_sizes_2d = numpy.repeat(cone_sizes[numpy.newaxis,:], 2, axis=0)
    retina_params_microns = retina(cone_positions_window , cone_sizes_2d, reflectance_map=reflectance_map, width_variation=width_variation, mean_reflectance=mean_reflectance, std_reflectance=std_reflectance, skew_location=skew_location, skew_scale=skew_scale, skew_alpha=skew_alpha, skew_amplitude=skew_amplitude, skew_offset=skew_offset, distribution=distribution)
    
    return retina_params_microns


def retina(cone_locations_um, cone_sizes_um, reflectance_map=None, width_variation=0.0, mean_reflectance=None, std_reflectance=None, skew_location=None, skew_scale=None, skew_alpha=None, skew_amplitude=None, skew_offset=None, distribution='normal'):
    '''Creates an array of [cone y pos, cone x pos, cone x width (stdev), 
    cone y width (stdev), rotation angle (radians), reflectance] given the 
    variabilities specified.
    
    Parameters
    ----------
    cone_locations_um : array_like
        An array containing the locations of the cones, typically the output
        of the reaction diffusion model, scaled for a particular retinal
        eccentricity.
    cone_sizes_um : array_like
        The sizes of the cones, calculated based on their eccentricity.
    reflectance_map : array_like (optional)
        A 2D amplitude map specifying the cone reflectance as a function of position on
        the retina. The default is to select cone reflectances from a normal
        distribution with mean mean_amp and standard deviation std_amp. Passing
        a 2D map will, instead calculate the average reflectance value at the
        cone's location in the amplitude map.
    width_variation : float (optional)
        To add some variability to the sizes of the cones, specify a value > 0.
        The cone sizes specified in cone_sizes_um are each multiplied by a 
        factor. For each cone, this factor is selected from a normal 
        distribution with a mean of 1.0 and a standard deviation of 
        width_variation The default is 0.0, i.e. no variability.
    mean_reflectance : float (optional)
        Cone reflectances are selected from a normal distribution, this 
        specifies the mean of that distribution. 
    std_reflectance : float (optional)
        Cone reflectances are selected from a normal distribution, this 
        specifies the standard deviation of that distribution. 
        
        
    Returns
    -------
    params : array_like
        The retina parameter array.
        
    '''

    # Add some random variability to the horizontal and vertical widths
    # of the cones if desired
    if width_variation > 0.:
    
        # Variation in diameter of cones
        w = numpy.random.normal(1.,width_variation, size=(cone_locations_um.shape[1]))
        cone_sizes_um[0] *= w
        cone_sizes_um[1] *= w


    # Cone reflectances are specified by the amp_map if given and if not, each 
    # cone has a reflectance selected from a normal distribution.
    if type(reflectance_map) == type(None):
        if distribution == 'normal':
            if type(mean_reflectance) == type(None) or type(std_reflectance) == type(None):
                sys.stdout.write('The mean and standard deviation of the normal distribution for cone reflectance selection must be specified')
            else:
                
                reflectance = numpy.random.normal(mean_reflectance, std_reflectance, size=(cone_sizes_um.shape[1]))
        elif distribution == 'skew':
            if type(skew_location) == type(None) or type(skew_scale) == type(None) or type(skew_alpha) == type(None) or type(skew_amplitude) == type(None) or type(skew_offset) == type(None):
                sys.stdout.write('All of the parameters of the skew normal distribution for cone reflectance selection must be specified')
            else:
                
                reflectance = selectSkew(cone_sizes_um.shape[1], [skew_location, skew_scale, skew_alpha, skew_amplitude, skew_offset])
        

    else:
        # Create a copy of the cone locations and shift them so that they begin at [0,0] and are only positive in value
        positive_locations = numpy.copy(cone_locations_um)
        positive_locations[0] -= positive_locations[0].min()
        positive_locations[1] -= positive_locations[1].min()
        
        # Clip the locations so that they are only within the amplitude map that has been specified.
        positive_locations[0] = numpy.clip(positive_locations[0],0,reflectance_map.shape[0]-1)
        positive_locations[1] = numpy.clip(positive_locations[1],0,reflectance_map.shape[1]-1)
    
        # Select the average value from the amplitude map in a window around the cone's location.
        reflectance = numpy.asarray([numpy.mean(reflectance_map[numpy.clip(int(round(positive_locations[0,i]-int(cone_sizes_um[0,i]*3))), 0, reflectance_map.shape[0]):numpy.clip(int(round(positive_locations[0,i]+int(cone_sizes_um[0,i]*3))), 0, reflectance_map.shape[0]),
    numpy.clip(int(round(positive_locations[1,i]-int(cone_sizes_um[1,i]*3))), 0, reflectance_map.shape[1]):numpy.clip(int(round(positive_locations[1,i]+int(cone_sizes_um[1,i]*3))), 0, reflectance_map.shape[1])]) for i in range(positive_locations.shape[1])])

    # Create the parameter array
    params = numpy.zeros((6,cone_locations_um.shape[1]))

    params[0] = cone_locations_um[0]
    params[1] = cone_locations_um[1]
    params[2] = cone_sizes_um[0]
    params[3] = cone_sizes_um[1]
    params[4] = numpy.random.uniform(0,2*numpy.pi, size=(params.shape[1]))
    params[5] = reflectance

    return params

def pdf(x):
    ''' The standard normal probability density function. This seems to run
    faster in a loop than the scipy version.
    
    Parameters
    ----------
    x : array_like
        Input variable.

    '''
    
    return 1/numpy.sqrt(2*numpy.pi) * numpy.exp(-x**2/2)

def cdf(x):
    ''' The standard normal cumulative distribution function. This seems to run
    faster in a loop than the scipy version.
    
    Parameters
    ----------
    x : array_like
        Input variable.

    '''
    return (1 + erf(x/numpy.sqrt(2))) / 2

def functSkew(x, p):
    ''' A skew normal distribution.
    
    Parameters
    ----------
    x : array_like
        Input variable.
    p : array_like
        The parameters of the skew normal distribution, given as:
        (location, scale, alpha, amplitude). These values can be converted to
        mean, variance, skewness and excess kurtosis using the 
        'calculatSkewCoefficients'.

    '''
    
    e=p[0]
    w=p[1]
    a=p[2]
    b = p[3]
    t = (x-e) / w
    c = p[4]
    return b * 2 / w * pdf(t) * cdf(a*t) + c


def selectSkew(n, params, bit_depth=8):
    '''Randomly selects numbers from a skew normal distribution
    
    Parameters
    ----------
    n : int
        The number of values to draw
    params : array_like
        The parameters of the skew normal distribution, given as:
        (location, scale, alpha, amplitude). These values can be converted to
        mean, variance, skewness and excess kurtosis using the 
        'calculatSkewCoefficients'.
        
    Returns
    -------
    values : array_like
        The numbers drawn from a skew normal distribution
    '''
    
    values = []
    
    while len(values)<n:
        # Select a random value, which should be in the range 0-Imax, where Imax is the number of gray levels in the image, with half-step resolution
        test_val = round(numpy.random.uniform(-0.4999999, 2**(bit_depth+1) + 0.4999999)) / (2**(bit_depth+1))
        
        # Compute the probability density for the random value
        p_test = functSkew(test_val, params)

        # Select a value from a random uniform distribution as a test. If it is less than or equal to our desired probability, we accept the value.
        sample = numpy.random.uniform(0,5)
        if sample<=p_test:
            values.append(test_val)

    return numpy.asarray(values)


