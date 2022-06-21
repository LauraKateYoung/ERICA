''' 
Fixation simulation
-------------------
A simulation of fixational eye movements including drift, tremor and 
microsaccades.
'''


import numpy
import yaml

from scipy.optimize import fmin

from ERICA import ERICA_toolkit

class Fixation:
    
    '''A Fixation class. Instances of this class are oculomotor objects with 
    parameters specified in the fixation parameter yaml file.
    '''
    def __init__(self, fixation_parameter_file, sample_time, 
                 subsamp=64):
        
        '''Creates an instance of a Fixation object. Individual periods of 
        tremor, drift or microsaccades are generated with parameters 
        (amplitude, frequency etc) drawn from a normal distribution. The 
        fixation parameter file sets the defaul mean and standard deviation of
        those normal distributions.
        
        Parameters
        ----------
        fixation_parameter_file : string
            The path to the fixational eye movemennts parameter file
        sample_time : float
            The time between samples, which should match the AOSLO pixel time
        subsamp : int, optional
            The temporal downsampling rate. Eye movements are generated with a
            lower temporal resolution and then upsampled at the end. This
            speeds up data generation. This number should be small, and no
            larger than number of pixels per scan line, to avoid undersampling.
            The default is 64x.
            
        '''
                
        self.fixation_parameter_file = fixation_parameter_file
        
        if type(sample_time) != float:
            sample_time = float(sample_time)
        self.sample_time = sample_time
        
        if type(subsamp) != float:
            subsamp = int(round(subsamp))
        self.subsamp = subsamp
                 
    
        # Open the fixation parameters file
        with open(self.fixation_parameter_file, 'r') as f:
            self.parameters = yaml.safe_load(f)
                 
     
    def fixationalEyemovementTrace(self, total_time_s):
        
        ''' Generates a fixational eye movement trace
            
        Parameters
        ----------
        total_time_s : float
            The amount of time over which eye movements are generated (seconds)
        
        
            
        Returns
        -------
        mx : array_like
            Horizontal eye position data in arcminutes
        my : array_like
            Vertical eye position data in arcminutes
        time_samples : array_like
            Time in seconds
        fix_param : array_like
            Parameters for each inter-microsaccadic period (tremor + drift) 
            given as:
            [tremor amplitude (standard deviation, arcminutes), 
            tremor centre frequency (Hz),
            tremor bandwidth (Full Width Half Maximum, Hz), 
            drift amplitude (maximum deviation, arcminutes)]
        ms_param : array_like
            Parameters for each microsaccade given as: 
            [microsaccade amplitude (maximum deviation, arcminutes), 
            angle (direction from positive horizontal axis, radians),
            maximum speed (arcminutes/second)]
        
        '''
    
        # Make sure that the time and time resolution are floats    
        if type(total_time_s) != float:
            total_time_s = float(total_time_s)
        
        # The total number of samples is the total time divide by the time 
        # resolution, rounded up to the nearest integer.
        n_samples = int(numpy.ceil(total_time_s / self.sample_time))
        
        n_samples /= self.subsamp
        time_resolution = self.sample_time * self.subsamp
        
        # Generate some microsaccade intervals
        microsaccade_timepoints = self.realistic_ms_intervals(total_time_s)
        
        # Set up the lists that data will be appended to
        motion_horizontal = [[0.0]]
        motion_vertical = [[0.0]]
        fix_param=[]
        ms_param=[]
        microsaccade_marker = []
        
        # Start the sample counter
        sample_counter = 0
    
        # Iterate through each intermicrosaccadic interval and append eye movement
        # data.
        for i in range(len(microsaccade_timepoints)):
        
            # Current time point
            start_time = float(numpy.copy(sample_counter))*time_resolution
            
            # determine the start point for the next microsaccade and the time 
            # until it occurs
            microsaccade_start_sample = int(round(microsaccade_timepoints[i]/time_resolution))
            microsaccade_start_time = microsaccade_timepoints[i]
            intermicrosaccadic_time_interval = microsaccade_start_time-start_time
            number_samples_between_microsaccades = int(round(intermicrosaccadic_time_interval/time_resolution))
            
            # Generate fixational eye motion (tremor + drift) for the 
            # inter-microsaccadic interval and save the parameters of the movement
            fem, param_fem = self.eyeMotion_TremorDrift(time_resolution, number_samples_between_microsaccades)
            if sample_counter < n_samples:
                fix_param.append(param_fem)
            
            # Add the fixational eye motion to the motion trace, lining it up with
            # the last position. N.b. fem[:,1] corresponds to horizontal motion
            # and fem[:,0] to vertical motion
            
            motion_horizontal.append(fem[1,:microsaccade_start_sample] - fem[1,0] + motion_horizontal[-1][-1])
            motion_vertical.append(fem[0,:microsaccade_start_sample] - fem[0,0] + motion_vertical[-1][-1])
            
            sample_counter += len(motion_horizontal[-1])
            ms_start = numpy.copy(sample_counter)
            
            # If the motion is not centred, choose an angle for the next 
            # microsaccade such that it directs the eye (roughly) back towards the
            # centre. This is only to prevent the eye 'wandering off'.

            if motion_horizontal[-1][-1] != 0.0 and motion_vertical[-1][-1] != 0.0:
                
                # Determine the angular position at the end of the fixational 
                # motion
                current_angle = numpy.arctan(motion_vertical[-1][-1] / motion_horizontal[-1][-1])
                
                # Deal with the signs to give the conventional polar coordinate
                # (Angle increasing anticlockwise from the positive x-axis)
                if motion_horizontal[-1][-1]<0:
                    current_angle += numpy.pi
    
                # Add 180 degrees to direct the eye in the opposite direction.
                # N.b. This will be altered by a random amount so that the eye
                # does not simply go straight back to the centre every time.
                optimal_angle = current_angle + numpy.pi
            
            else:
                optimal_angle = None
    
            # Generate a microsaccade and save its parameters - use a large number 
            # of samples so as not to miss the start or end (this is cropped later)
            ms, param_m = self.eyeMotion_Microsaccade(time_resolution, optimum_angle=optimal_angle)
            
            # Find the data between the start and end of the generated microsaccade
            # - defined as the points at which the velocity exceeds 10%.
            ms_radial_profile = numpy.sqrt(ms[0]**2 + ms[1]**2)
            ms_velocity_profile = (ms_radial_profile[1:] - ms_radial_profile[:-1]) / time_resolution
            ms_samples = numpy.where(ms_velocity_profile > 0.01 * ms_velocity_profile.max())[0]
            ms_data = numpy.asarray(ms[:,ms_samples])

            # Add the microsaccade data to the trace, lined up to the last position
            motion_x_vals = ms_data[1] - ms_data[1,0] + motion_horizontal[-1][-1]
            motion_y_vals = ms_data[0] - ms_data[0,0] + motion_vertical[-1][-1]
            motion_horizontal.append(motion_x_vals)
            motion_vertical.append(motion_y_vals)
            
            # save the indices of the start and end of the microsaccade
            sample_counter += len(motion_horizontal[-1])
            if ms_start < n_samples and sample_counter < n_samples:
                microsaccade_marker.append([ms_start, sample_counter])
                ms_param.append(param_m)
            elif ms_start < n_samples and sample_counter >= n_samples:
                microsaccade_marker.append([ms_start, numpy.nan])
                ms_param.append(param_m)
    
        mx = numpy.concatenate(motion_horizontal)[:int(n_samples)]
        my = numpy.concatenate(motion_vertical)[:int(n_samples)]
        microsaccade_marker = numpy.asarray(microsaccade_marker)
        ms_param = numpy.asarray(ms_param)
        fix_param = numpy.asarray(fix_param)
        
        # Upsample the data back to the original time_resolution and return the 
        # data and parameters
        mx = numpy.repeat(mx,self.subsamp)
        my = numpy.repeat(my,self.subsamp)
        try:
            microsaccade_marker = microsaccade_marker * self.subsamp
        except:
            microsaccade_marker = numpy.zeros((1,2))
        
        # Generate the time samples
        time_samples = numpy.linspace(0, mx.shape[0]-1, mx.shape[0]) * self.sample_time
        
        return mx, my, time_samples, fix_param, ms_param, microsaccade_marker.astype(int)
    
    def realistic_ms_intervals(self, total_time_s):
        
        '''Estimates intervals between microsaccades based on a normal distribution.
        
        Parameters
        ----------
        total_time_s : float
            The amount of time over which eye movements are generated (seconds)
        
                
        Returns
        -------
        intervals : list
            A list of time points for microsaccade onsets
     
        '''
        
        intervals = []
        time_counter = 0.0
        
        # Keep choosing an interval from a normal distribution and adding it to the
        # time counter until the total time has been reached
        while time_counter < total_time_s:
            this_interval = numpy.random.normal(self.parameters['Microsaccade_interval_seconds_MEAN'], self.parameters['Microsaccade_interval_seconds_STD'])
            if this_interval > self.parameters['Minimum_microsaccade_interval']:
                time_counter += this_interval
                intervals.append(time_counter)
        return intervals
    

    def eyeMotion_Tremor(self, time_resolution, n_samples):
        ''' Generates the parameters of the current period of tremor by 
        selecting the amplitude, centre frequency and bandwidth from normal
        distrubitions, and generates the corresponding position data.
        
        Parameters
        ----------
        time_resolution : float
            The time between samples for motion generation. This will 
            be longer than the AOSLO pixel time by a factor equal to the 
            rate of subsampling.
        n_samples : int
            The number of samples to generate. This is total time divided by
            the time_resolution.
            
        Returns
        -------
        tremor_motion : array_like
            An array containing the eye position (arcminutes) for the current
            period of tremor
        current_tremor_parameters : list
            A list of parameters for the current period of tremor - 
            [amplitude (arcmin), centre frequency (Hz), bandwidth (Hz)]
     
        '''
        
        tremor_amp = numpy.random.normal(loc=self.parameters['Tremor_amplitude_arcmin_MEAN'], scale=self.parameters['Tremor_amplitude_arcmin_STD'])
        tremor_freq = numpy.random.normal(loc=self.parameters['Tremor_centre_frequency_MEAN'], scale=self.parameters['Tremor_centre_frequency_STD'])
        tremor_bd = numpy.random.normal(loc=self.parameters['Tremor_bandwidth_MEAN'], scale=self.parameters['Tremor_bandwidth_STD'])
                
        current_tremor_parameters = [tremor_amp, tremor_freq, tremor_bd]
        tremor_motion = tremor(tremor_freq, tremor_bd, tremor_amp, time_resolution, n_samples)
        
        return tremor_motion, current_tremor_parameters
    
    def eyeMotion_Drift(self, time_resolution, n_samples):
        ''' Generates the parameters of the current period of drift by 
        selecting the amplitude from a normal distrubition and generates the
        corresponding position data.
        
        Parameters
        ----------
        time_resolution : float
            The time between samples for motion generation. This will 
            be longer than the AOSLO pixel time by a factor equal to the 
            rate of subsampling.
        n_samples : int
            The number of samples to generate. This is total time divided by
            the time_resolution.
            
        Returns
        -------
         drift_motion : array_like
            An array containing the eye position (arcminutes) for the current
            period of drift
        drift_amp : float
            The amplitude (arcmin) of the current period of drift.
     
        '''
        
        
        drift_amp = numpy.random.normal(loc=self.parameters['Drift_amplitude_arcmin_MEAN'], scale=self.parameters['Drift_amplitude_arcmin_STD'])
        drift_motion = drift(drift_amp , time_resolution, n_samples)
        
    
        return drift_motion, drift_amp
    
    def eyeMotion_TremorDrift(self, time_resolution, n_samples):
        
        ''' Generates the parameters of the current period of combined tremor 
        and drift by selecting the parameters from normal distrubitions, and 
        generates the corresponding position data.
        
        Parameters
        ----------
        time_resolution : float
            The time between samples for motion generation. This will 
            be longer than the AOSLO pixel time by a factor equal to the 
            rate of subsampling.
        n_samples : int
            The number of samples to generate. This is total time divided by
            the time_resolution.
            
        Returns
        -------
        motion_tremor_drift : array_like
            An array containing the eye position (arcminutes) for the current
            period of combined tremor and drift.
        current_tremor_drift_parameters : list
            The parameters of the current period of combined tremor and drift-
            [tremor amplitude (arcmin), tremor centre frequency (Hz), 
            tremor bandwidth (Hz), drift amplitude (armcmin)]
     
        '''
        
        tremor_motion, [tremor_amp, tremor_freq, tremor_bd] = self.eyeMotion_Tremor(time_resolution, n_samples)
        drift_motion, drift_amp = self.eyeMotion_Drift(time_resolution, n_samples)
        
        motion_tremor_drift = tremor_motion + drift_motion
        current_tremor_drift_parameters = [tremor_amp, tremor_freq, tremor_bd, drift_amp]
        
        return motion_tremor_drift, current_tremor_drift_parameters
      
        
        
    def eyeMotion_Microsaccade(self, time_resolution, optimum_angle=None, optimal_angle_std=numpy.pi/20., max_speed=None):
        
        ''' Generates the parameters of the current microsaccade by selecting 
        the parameters from normal distrubitions, and generates the 
        corresponding position data.
        
        Parameters
        ----------
        time_resolution : float
            The time between samples for motion generation. This will 
            be longer than the AOSLO pixel time by a factor equal to the 
            rate of subsampling.
        optimum_angle : float (optional)
            The angle (radians) of the microsaccade, usually optimised to maintain a
            locus around the centre of fixation
        optimal_angle_std : float (optional)
            The angle of the microsaccade is selected from a normal 
            distribution centred on optimum_angle with a standard deviation
            equal to optimal_angle_std.
        max_speed : float (optional)
            If specified, the peak velocity (magnitiude) of the microsaccade
            will be set to this value. If max_speed = None, the peak velocity
            will be determined from the main sequence.
        
            
        Returns
        -------
        ms_motion : array_like
            An array containing the eye position (arcminutes) for the current
            microsaccade.
        ms_parameters : list
            The parameters of the current microsaccade-
            [microsaccade amplitude (arcmin), microsaccade amplitude (radians), 
            peak speed (arcmin/s)]
     
        '''
        
        ms_amp = numpy.random.normal(loc=self.parameters['Microsaccade_amplitude_arcmin_MEAN'], scale=self.parameters['Microsaccade_amplitude_arcmin_STD'])
        
        # If the optimal angle is not specified, choose a random angle
        if type(optimum_angle) == type(None):
            angle = numpy.random.uniform(0,numpy.pi*2)
    
        # If the optical angle is specified, choose an angle from a normal 
        # distribution with some small variation
        else:
            angle = numpy.random.normal(optimum_angle, optimal_angle_std)
    
        # if the maximum speed is not specified, select it based on the main 
        # sequence - amplitude (deg) is linearly related to the maximum speed 
        # (deg/s) in log space
        if type(max_speed) == type(None):
            
            # Log of the amplitude in degrees
            l = numpy.log10(ms_amp/60.)
    
            # Estimate the maximum speed from the main sequence 
            ms = l + numpy.log10(self.parameters['Main_sequence_factor'])
    
            # Select a maximum speed (arcmin/s) from a normal distribution centred on this
            # estimate with some variability
            max_speed = 60*numpy.random.normal(loc=10**ms, scale=self.parameters['Main_sequence_STD'])
        
        
        ms_motion = microsaccade(ms_amp, angle, max_speed, time_resolution, n_samples=None)
        ms_parameters = [ms_amp, angle, max_speed]
    
    
        return ms_motion, ms_parameters
    

def microsaccade(amplitude, angle, max_speed, time_resolution, n_samples=None):
    ''' Generates a microsaccade
        
        Parameters
        ----------
        amplitude : float
            The amplitude of the microsaccade (arcmin) - the distance traveled
            from the start to the end
        angle : float
            The angular direction of the microsaccade (radians)
        max_speed :float
            The maximum speed of the microsaccade (arcmin/second)
        time_resolution : float
            The time between motion samples (should match the AOSLO pixel clock)
        n_samples : int (optional)
            The number of samples in the microsaccade. This need to be high
            enough that the start and end of the microsaccade are not missed.
            If n_samples = None, it is estimated based on the estimated
            microsaccade duration.
        
            
        Returns
        -------
        positions : array_like
            The vertical and horizontal eye position data (arcmin) for the
            current microsaccade
       
        '''

    if n_samples is None:
        n_samples = int(numpy.ceil((amplitude * 10. / max_speed) / time_resolution))
    
    time = numpy.linspace(0, n_samples-1, n_samples) * time_resolution
    
    # Ensure that the microsaccade duration is set such that the specified 
    # speed and amplitude are generated
    def funct(duration, t):
        std = numpy.sqrt(1./(2 * numpy.log(100.0))) * duration / 2.
        v2 = ERICA_toolkit.functGaussian(t, t[int(round(t.shape[0] / 2.))], std)
        v2 /= v2.max()
        v2 *= max_speed
        return  v2
    
    def funct_error(duration, t):
        v2 = funct(duration, t)
        n = numpy.where(v2 >= 0.01 * v2)[0]
        dist = numpy.cumsum(v2[n] * time_resolution)
        err = (dist.max() - dist.min()) - amplitude
        return  numpy.sum(err ** 2)
    
    p0 = [amplitude * 2 / max_speed]
    plsq, fopt, itera, funcalls, warnflag = fmin(funct_error, p0, args=(time,),maxiter=10000, maxfun=10000, full_output=True, disp=False, retall=False, xtol=0.001, ftol=0.0001)

    # Generate the motion profile using a cumulative Gaussian distribution
    velocity = funct(plsq[0], time)
    dist = numpy.cumsum(velocity * time_resolution)

    # Orient the microsaccade according to the specified angle
    sx = dist * numpy.cos(angle)
    sy = dist * numpy.sin(angle)

    positions = numpy.zeros((2, sx.shape[0]))
    positions[0] = sy
    positions[1] = sx
    
    return positions


def tremor(centre, width, amplitude_arcmin, time_resolution, n_samples):
    ''' Generates a period of tremor
        
        Parameters
        ----------
        centre : float
            The centre frequency of the period of tremor (Hz) 
        width : float
            The bandwidth of the period of tremor (Hz) 
        amplitude_arcmin :float
            The amplitude (standard deviation) of the period of tremor (arcmin)
        time_resolution : float
            The time between motion samples (should match the AOSLO pixel clock)
        n_samples : int
            The number of samples in the period of tremor. 
        
            
        Returns
        -------
        positions : array_like
            The vertical and horizontal eye position data (arcmin) for the
            current period of tremor.
       
        '''

    # Start with uniformly distributed random positions
    sx = numpy.random.uniform(0,1,size=(n_samples))
    sy = numpy.random.uniform(0,1,size=(n_samples))
       
    # Filter the positions according to the specified band pass filter
    sx2, fx, ff = ERICA_toolkit.filter_1d(sx,centre,time_resolution, width=width, typ='bp')
    sy2, fy, ff = ERICA_toolkit.filter_1d(sy,centre,time_resolution, width=width, typ='bp')

    # Scale the filtered positions to the specified amplitude
    r = numpy.sqrt(sx2**2 + sy2**2)
    amp = numpy.std(r)
    positions = numpy.zeros((2,sx2.shape[0]))
    positions[0] = amplitude_arcmin * sy2/amp
    positions[1] = amplitude_arcmin * sx2/amp
    
    return positions
    

def driftPowerSpectrum(f): 
    ''' Generates a the 1/f^2 power spectrum for drift 
        
        Parameters
        ----------
        f : array_like
            The temporal frequencies to calculate the spectrum for (Hz)
        time_resolution : float
            The time between motion samples (should match the AOSLO pixel clock)
        n_samples : int
            The number of samples in the period of tremor. 
        
            
        Returns
        -------
        spectrum : array_like
            The power at the specified frequencies.
       
        '''
    spectrum = numpy.where(f==0,0,1./(f**2))    

        
    return spectrum  


def drift(amplitude_arcmin_drift, time_resolution, n_samples):
    ''' Generates a period of drift
        
        Parameters
        ----------
        amplitude_arcmin_drift :float
            The amplitude  of the period of drift (arcmin)
        time_resolution : float
            The time between motion samples (should match the AOSLO pixel clock)
        n_samples : int
            The number of samples in the period of tremor. 
        
            
        Returns
        -------
        positions : array_like
            The vertical and horizontal eye position data (arcmin) for the
            current period of drift.
       
        '''
    # Start with uniformly distributed random positions
    sx_real = numpy.random.uniform(0,1,size=(n_samples*2))
    sy_real = numpy.random.uniform(0,1,size=(n_samples*2))

    # Create a temporal filter based on the drift power spectrum
    freq_scale = 1.0/(time_resolution * n_samples)
    frequencies = numpy.linspace(-numpy.floor(n_samples / 2.), numpy.floor((n_samples-1.) / 2.), n_samples * 2) * freq_scale  
    frequency_spectrum = driftPowerSpectrum(frequencies)
    frequency_spectrum = numpy.where(frequencies==0.0, 0, frequency_spectrum)
    frequency_spectrum /= frequency_spectrum.sum()

    # Filter the random positions
    ft_x = numpy.fft.fftshift(numpy.fft.fft(sx_real-numpy.mean(sx_real)))
    fft_x = ft_x * frequency_spectrum

    ft_y = numpy.fft.fftshift(numpy.fft.fft(sy_real-numpy.mean(sy_real)))
    fft_y = ft_y*frequency_spectrum
    
    drft_x = numpy.fft.ifft(numpy.fft.ifftshift(fft_x)).real
    drft_y = numpy.fft.ifft(numpy.fft.ifftshift(fft_y)).real
    
    drft_x = drft_x[int(drft_x.shape[0]/2):int(drft_x.shape[0]/2)+n_samples]
    drft_y = drft_y[int(drft_y.shape[0]/2):int(drft_y.shape[0]/2)+n_samples]

    # Scale the positions to the specified amplitude (difference between start
    # and end of drift)
    amp = numpy.sqrt((drft_x[-1]-drft_x[0])**2 + (drft_y[-1]-drft_y[0])**2)

    sx = drft_x * (amplitude_arcmin_drift/amp) 
    sy = drft_y * (amplitude_arcmin_drift/amp)
    
    positions = numpy.zeros((2, n_samples))
    positions[1] = sx
    positions[0] = sy

    return positions
 


    




