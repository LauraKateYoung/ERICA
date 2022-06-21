''' 
ERICA toolkit
-------------------
Useful functions called within other modules or within exampe Juypter notebooks.
'''

import numpy
import scipy

from  matplotlib import pyplot
from  matplotlib import collections
from scipy.optimize import leastsq, fmin

class minim:

    def __init__(self,funct,x,obs,p0,args=None,maxfev=100000,xtol=0.0001, ftol=0.0001):
        self.peval=funct
        self.obs=obs
        self.x=x
        self.param,self.fn,self.warnflag=self.fitSimplex(self.x,self.obs,p0,arg=args,maxfev=maxfev,xtol=xtol, ftol=ftol)
        self.sumSquares=numpy.sum(self.residuals(self.param,self.obs,self.x)**2)
        

    def residuals(self,p, obs, x,arg=None):
        if type(arg) != type(None):
            err = obs-self.peval(x,p,args=arg)
        else:
            err = obs-self.peval(x,p)
        return err

    def residualsSimplex(self,p, obs, x,arg=None):
        if type(arg) != type(None):
            err = obs-self.peval(x,p,args=arg)
        else:
            err = obs-self.peval(x,p)
        return numpy.sum(err**2)

    def fitLS(self,x,obs,p0,arg=None,maxfev=10000):
        plsq = leastsq(self.residuals, p0, args=(obs, x,arg), maxfev=maxfev )
        p=plsq[0]
        fn=self.peval(x,p)
        return p,fn

    def fitSimplex(self,x,obs,p0,arg=None,maxfev=10000,prnt=False,xtol=0.0001, ftol=0.0001):
        plsq,fopt,itera,funcalls,warnflag = fmin(self.residualsSimplex, p0, args=(obs, x,arg),maxiter=maxfev,maxfun=maxfev,full_output=True,disp=prnt,retall=prnt,xtol=xtol, ftol=ftol)
        p=plsq
        fn=self.peval(x,p)
        return p,fn,warnflag

    

def fitCurcio(x, data, p0=[0.0,0.0,0.0,0.0]):
    '''Performs fitting to the Curcio dataset
    
        Parameters
        ----------
        x : array_like
            A 1D array containing the eccenricity samples (degrees)
        data : float
            The Curcio data
        p0 : array_like
            The initial guess for the fit parameters
        
            
        Returns
        -------
        f.param : array_like
            The fit parameters
        
    '''
    
    p0[2] = 0.5
    p0[3] = data.min()
    f=minim(functCurcio,x,data,p0)
    return f.param

def functCurcio(x, p):
    
    '''Performs fitting to the Curcio dataset
    
        Parameters
        ----------
        x : array_like
            A 1D array containing the eccenricity samples (degrees)
        
        p : array_like
            The guess for the fit parameters
        
            
        Returns
        -------
        result.real : array_like
            The fit parameters
            
    '''  
      
    # The output could be complex so change the data type to match
    x2 = numpy.array(x, dtype=numpy.complex)

    result = p[0] * (x2 + p[1]) ** p[2] + p[3]
    
    return result.real

def functGaussian(x, mean, stdev):
    '''Calculates a Gaussian function
    
        Parameters
        ----------
        x : array_like
            A 1D array containing the samples
        mean : float
            The mean (cenre)r of the Gaussian
        stdev : float
            The standard deviation (width) of the Gaussian.
        
            
        Returns
        -------
        gaussian_profile : array_like
            The computed filter profile.
        
    '''
    gaussian_profile = numpy.exp(-(x - mean) ** 2 / (2 * stdev ** 2))
    
    return gaussian_profile


def radialProfileSum(data):
    '''Computes the radial profile of a 2D data set, where values are summed
    over all angles
    
        Parameters
        ----------
        data : array_like
            The 2D to data
        
            
        Returns
        -------
        radial_sum : array_like
            The computed radial profile.
        radial_values : array_like
            The radius values
            
        
    '''
    y, x = numpy.ogrid[-data.shape[0]/2:data.shape[0]/2:data.shape[0]*1j,-data.shape[1]/2:data.shape[1]/2:data.shape[1]*1j]
    r = numpy.sqrt(x**2 + y**2).round().astype(int)


    tbin = numpy.bincount(r.ravel(), data.ravel())
    nr = numpy.bincount(r.ravel()).astype(float)
    radial_sum = numpy.where(nr==0.0, numpy.nan, tbin / nr)
    
    radial_values = numpy.linspace(0,r.max(), r.max()+1)
    return radial_sum, radial_values


####################
# FILTER FUNCTIONS #
####################

def convolve2D(array_1, array_2):
    '''There are different ways to perform convolution. This is a wrapper
    function allowing the method of convolution to be changed globally 
    throughout ERICA. The original implementation uses scipy.signal.convolve, 
    which is fast and implements zero padding. It will automatically use either
    direct or FFT-based circular convolution depending on which is predicted to
    be faster based on the array sizes.
    
    Parameters
        ----------
    array_1 : array_like
        A 2D array containing one of the two arrays to be convolved
    array_1 : array_like
        A 2D array containing the other of the two arrays to be convolved
    
        
    Returns
    -------
    convolution : array_like
        The computed convolution.
    '''
    
    convolution = scipy.signal.convolve(array_1, array_2, mode='same')
    return convolution

def bandPass(frequencies, centre, width):
    '''Calculates a Gaussian bandpass filter profile
    
        Parameters
        ----------
        frequencies : array_like
            A 1D array containing the frequency samples
        centre : float
            The centre frequency of the filter. 
        width : float
            The full-width-half-maximum of the filter.
        
            
        Returns
        -------
        filter_profile : array_like
            The computed filter profile.
        
    '''

    # Calculate the standard deviation from the FWHM
    gaussian_std = width / (2.0 * numpy.sqrt(2.0 * numpy.log(2.0)))
    
    # The filter profile is repeated for positive and negative frequencies (+centre and -centre)    
    filter_profile = functGaussian(frequencies, centre, gaussian_std) + functGaussian(frequencies, -centre, gaussian_std)
    
    return filter_profile

def lowPass(frequencies, cutoff, order=5):
    '''Calculates a Butterworth low pass filter profile
    
        Parameters
        ----------
        frequencies : array_like
            A 1D array containing the frequency samples
        cutoff : float
            The -3dB cut-off frequency of the filter. 
        order : float, optional
            The order of the Butterworth filter (number of poles; default=5)
        
            
        Returns
        -------
        filter_profile : array_like
            The computed filter profile.
        
    '''


    filter_profile = numpy.sqrt( 1 / (1 + (frequencies / cutoff) ** (2 * order)))
    
    return filter_profile
 
def highPass(frequencies, cutoff, order=5):
    
    '''Calculates a Butterworth lhighpass filter profile
    
        Parameters
        ----------
        frequencies : array_like
            A 1D array containing the frequency samples
        cutoff : float
            The -3dB cut-off frequency of the filter. 
        order : float, optional
            The order of the Butterworth filter (number of poles; default=5)
        
            
        Returns
        -------
        filter_profile : array_like
            The computed filter profile.
        
    '''
    

    filter_profile = numpy.sqrt(1 / (1 + (cutoff / frequencies) ** (2 * order)))
    
    return filter_profile


def filter_1d(data, centre, sample_scale, width=None, typ='bp', order=5):
    '''Preforms filtering of 1D data 
    
        Parameters
        ----------
        data : array_like
            A 1D array containing the signal to be filtered
        centre : float
            The centre frequency of the filter if using a bandpass filter, or
            the cut off frequency if using a high or low pass filter. The units
            should be the inverse of those for the sample_scale (e.g. Hz).
        sample_scale : float
            The size of a data sample (e.g. 1 second).
        width : float, optional
            The full-width-half-maximum of the filter, if using a bandpass 
            filter.
        typ : string
            The type of filter which should be either 'bp' for bandpass filter,
            'hp' for high pass filter or 'lp' for low pass filter. High and low
            pass filters use the Butterworth filter.
        order : int, optional
            The order of the Butterworth filter (default is 5).
        
            
        Returns
        -------
        filtered_data : array_like
            The filtered data.
        filter_profile : array_like
            The computed filter profile.
        frequencies : array_like
            The frequency samples for the computed filter.
        
    '''
    
    if data.ndim != 1:
        raise TypeError('Data must by a 1D array')
    
    frequency_scale = 1.0 / (sample_scale * data.shape[0])
    
    frequencies = numpy.linspace(-numpy.floor(data.shape[0]/2.), numpy.floor(data.shape[0]/2. - 1.0), data.shape[0]) * frequency_scale
    
    if typ == 'bp':
        if width is None:
            raise TypeError('Optional argument "width" must be specified for a bandpass filter')
        filter_profile = bandPass(frequencies,centre, width)
    
    elif typ == 'hp':
        filter_profile = highPass(frequencies, centre, order=order)
    
    elif typ == 'lp':
        filter_profile = lowPass(frequencies, centre, order=order)

    
    # Use a Hamming window
    haming_window = numpy.hamming(data.shape[0])
    
    data_windowed = data * haming_window
            
    # Convolve images via Fourier transform and multiplication
    fft_data_windowed = numpy.fft.fftshift(numpy.fft.fft(data_windowed))
    fft_filtered = fft_data_windowed * filter_profile
    
    # Inverse Fourier transform
    data_windowed_filtered = numpy.fft.ifft(numpy.fft.ifftshift(fft_filtered)).real
    
    # Remove the effects of the Hamming window
    filtered_data = data_windowed_filtered / haming_window 
    
    return filtered_data, filter_profile,frequencies



    
    

def normalise(data, bits=8, mn=None, mx=None):
    '''Normalises data 
    
        Parameters
        ----------
        data : array_like
            The data to be normalised
        bits : int
            The bit depth of the data
        mn : float, optional
            The minimum value to normalise to. The default is the minimum
            value in the data.
        mx : float, optional
            The maximum value to normalise to. The default is the maximum
            value in the data.
            
       
        
            
        Returns
        -------
        data_norm_scaled : array_like
            The normalised data, scaled to the required bit depth
        
    '''
    
    data=data.astype(numpy.float64)
    if type(mn) == type(None):
        mn = data.min()
    if type(mx) == type(None):
        mx = data.max()
    data_norm = (data-mn)/(mx-mn)
    data_norm_scaled = (data_norm * (2**bits-1))

    return data_norm_scaled




#####################
# PLOTING FUNCTIONS #
#####################

# These are take from the weblink below
def colorline(x, y, current_axes, z=None, cmap=pyplot.get_cmap('jet'), norm=pyplot.Normalize(0.0, 1.0),
        linewidth=1, alpha=1.0, clp=255):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = numpy.linspace(0.0, 1.0*clp/255, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = numpy.array([z])

    z = numpy.asarray(z)

    segments = make_segments(x, y)
    lc = collections.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha, linestyle='solid')

    lcc = current_axes.add_collection(lc)

    return lcc

def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = numpy.array([x, y]).T.reshape(-1, 1, 2)
    segments = numpy.concatenate([points[:-1], points[1:]], axis=1)
    return segments


