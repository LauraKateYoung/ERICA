''' 
Reaction diffusion
-------------------
Implements a reaction-diffusion model for generating a self-organising mosaic
of spots.
'''

import numpy
from numba import jit

@jit
def reactionDiffusionIterator(u, v, coefficients, Dv, idx, steps):
    '''Iterates through the reaction diffusion equations for the specified
    number of time steps
    
    Parameters
    ----------
    v : array
        The concentration of chemical V.
    u : array
        The concentration of chemical U.
    coefficients : list
        The coefficients for the reaction diffusion model (Du, F, k).
    Dv : float
        The diffusion coefficient for chemical V. 
    idx : int
        The current iteration number.
    steps : int
        The number of iterations to complete.
            
    Returns
    -------
    v : array
        The concentration of chemical V.
    u : array
        The concentration of chemical U.
    idx : int
        The current iteration number (1).
        
        
    '''

    # Assign the specified values to the coefficients
    (Du, f, k) = coefficients

    
    for i in range(steps):
        # Compute the Laplacians of u and v
        Laplace_u = (u[0:-2,1:-1] + u[1:-1,0:-2] - 4*u[1:-1,1:-1] + u[1:-1,2:] + u[2:  ,1:-1] )
        Laplace_v = (v[0:-2,1:-1] + v[1:-1,0:-2] - 4*v[1:-1,1:-1] + v[1:-1,2:] + v[2:  ,1:-1] )

        # Equation 3
        du_dt = (Du*Laplace_u - u[1:-1,1:-1] * v[1:-1,1:-1] * v[1:-1,1:-1] + f *(1-u[1:-1,1:-1]))
        
        # Equation 4
        dv_dt = (Dv*Laplace_v + u[1:-1,1:-1] * v[1:-1,1:-1] * v[1:-1,1:-1] - (f+k)*v[1:-1,1:-1])

        # Update the concentrations
        u[1:-1,1:-1] += du_dt
        v[1:-1,1:-1] += dv_dt
        
        idx += 1

    return u, v, idx

@jit
def reactionDiffusionStart(coefficients, Dv, height, width, start_size=20, u0=0.50, v0=0.25, amp_u=0.05, amp_v=0.05, foveated=False, Dv_gradient=0.0):
    '''Set up the starting conditions for reaction diffusion model

    Parameters
    ----------
    coefficients : list
        The coefficients for the reaction diffusion model (Du, F, k).
    Dv : float
        The diffusion coefficient for chemical V .
    height : int, optional
        The height of the reaction-diffusion system.
    width : int, optional
        The width of the reaction-diffusion system.
    start_size : int, optional
        The size of the 'seed' to start the reaction-diffusion system.
        This is the half width of a square area (in pixels) over which the
        chemicals are initially deposited. For a foveated system it starts
        with the chemicals in the centre. Otherwise, the system starts with
        9 x 9 seeds evenly spaced with a random offset. This allows a 
        stable pattern to be achieved faster.
    u0 : float, optional
        The starting concentration of chemical U within the seeded area.
    v0 : float, optional
        The starting concentration of chemical U within the seeded area.
    amp_u : float, optional
        The amplitude of the initial random perturbation (chemical U). This
        should be small compared to the starting concentration.
    amp_uv: float, optional
        The amplitude of the initial random perturbation (chemical U)This
        should be small compared to the starting concentration.
    foveated : bool, optional
        A flag to indicate whether a foveated mosaic is to be generated.
    Dv_gradient : float, optional
        The diffusion coefficient for chemical V at the edge of the system
        (only used for a foveated mosaic).
            
    Returns
    -------
    v : array
        The concentration of chemical V
    u : array
        The concentration of chemical U
    idx : int
        The current iteration number (1)
    '''
    
    # Set up the arrays for the concentrations of chemicals U and V
    u = numpy.zeros((height+2, width+2))
    v = numpy.zeros((height+2, width+2))
    u[1:-1,1:-1]=1.0
    
    # Assign the specified values to the coefficients
    (Du, f, k) = coefficients
    
    # If the mosaic is not to be foveated (the normal operation), start the
    # system with 9 x 9 seeds. This is faster.
    if foveated == False:
        
        dx = width/10.
        dy = height/10.
        
        if start_size>dx/4.:
            start_size = dx/4.
        start_size = int(round(start_size))
        
        # randomly offset the seeds
        nudge_x = numpy.random.uniform(0,dx/4.)
        nudge_y = numpy.random.uniform(0,dy/4.)

        for i in range(0,9):
            for j in range(0,9): 
                
                u[int(dy*(i+0.5+nudge_y))-start_size:int(dy*(i+0.5+nudge_y))+start_size, int(dx*(j+0.5+nudge_x))-start_size:int(dx*(j+0.5+nudge_x))+start_size] = u0
                v[int(dy*(i+0.5+nudge_y))-start_size:int(dy*(i+0.5+nudge_y))+start_size, int(dx*(j+0.5+nudge_x))-start_size:int(dx*(j+0.5+nudge_x))+start_size] = v0
        
    # If the mosaic is to be foveated, start with a single seed in the centre        
    elif foveated == True:
        
        u[height/2-start_size:height/2+start_size,width/2-start_size:width/2+start_size] = u0
        v[height/2-start_size:height/2+start_size,width/2-start_size:width/2+start_size] = v0

    # Add a small random pertubration
    u[1:-1,1:-1] += amp_u * numpy.random.random((height,width))
    v[1:-1,1:-1] += amp_v * numpy.random.random((height,width))
    
    # Run through one iteration to initiate the arrays
    idx = 0
    for i in range(1):
         # Compute the Laplacians of u and v
        Laplace_u = (u[0:-2,1:-1] + u[1:-1,0:-2] - 4*u[1:-1,1:-1] + u[1:-1,2:] + u[2:  ,1:-1] )
        Laplace_v = (v[0:-2,1:-1] + v[1:-1,0:-2] - 4*v[1:-1,1:-1] + v[1:-1,2:] + v[2:  ,1:-1] )

        # Equation 3
        du_dt = (Du*Laplace_u - u[1:-1,1:-1] * v[1:-1,1:-1] * v[1:-1,1:-1] + f *(1-u[1:-1,1:-1]))
        
        # Equation 4
        dv_dt = (Dv*Laplace_v + u[1:-1,1:-1] * v[1:-1,1:-1] * v[1:-1,1:-1] - (f+k)*v[1:-1,1:-1])

        # Update the concentrations
        u[1:-1,1:-1] += du_dt
        v[1:-1,1:-1] += dv_dt
        
        idx += 1

    return u, v, idx

    
def runReactionDiffusion(coefficients, steps, height, width, start_size=20, u0=0.50, v0=0.25, amp_u=0.05, amp_v=0.05, foveated=False, Dv_gradient=0.0):
    '''Runs through the reation-diffusion simulation

    Parameters
    ----------
    coefficients : list
        The coefficients for the reaction diffusion model (Du, F, k).
    steps : int
        The number of iterations to complete.
    height : int, optional
        The height of the reaction-diffusion system.
    width : int, optional
        The width of the reaction-diffusion system.
    start_size : int, optional
        The size of the 'seed' to start the reaction-diffusion system.
        This is the half width of a square area (in pixels) over which the
        chemicals are initially deposited. For a foveated system it starts
        with the chemicals in the centre. Otherwise, the system starts with
        9 x 9 seeds evenly spaced with a random offset. This allows a 
        stable pattern to be achieved faster.
    u0 : float, optional
        The starting concentration of chemical U within the seeded area.
    v0 : float, optional
        The starting concentration of chemical U within the seeded area.
    amp_u : float, optional
        The amplitude of the initial random perturbation (chemical U). This
        should be small compared to the starting concentration.
    amp_uv: float, optional
        The amplitude of the initial random perturbation (chemical U)This
        should be small compared to the starting concentration.
    foveated : bool, optional
        A flag to indicate whether a foveated mosaic is to be generated.
    Dv_gradient : float, optional
        The diffusion coefficient for chemical V at the edge of the system
        (only used for a foveated mosaic).
            
    Returns
    -------
    output : array
        The final concentration of chemical V

    '''
    # Assign the specified values to the coefficients
    (Du, Dv, f, k) = coefficients
    
    # If the mosaic is to be foveated, generate a radially varying Dv
    if foveated == True:
        if Dv_gradient!=0.0:
        
            # Set up radial coordinates
            y,x = numpy.indices((height,width)).astype(float)
            y-=float(height)/2.
            x-=float(width)/2.
            radial = numpy.sqrt(x**2+y**2)
            
            # Scale as specified
            radial *= Dv_gradient
            radial += Dv
            Dv = numpy.copy(radial)
    
    # Make a new list of coefficients with Dv removed (this is necessary for the JIT compiler to function
    coefficients = (Du, f, k)
    
    # Set up the system
    u, v, idx = reactionDiffusionStart(coefficients, Dv, height, width, start_size=start_size, u0=u0, v0=v0, amp_u=amp_u, amp_v=amp_v, foveated=foveated, Dv_gradient=Dv_gradient)
    
    # Divide the number of time steps in to multiples of 100. JIT compliation
    # is faster after the first call. Subsequent (identical) function calls
    # will be faster, so lots of small repetitions is best.
    reps = int(round(steps/100.))
    steps = 100
    for i in range(reps):
        u, v, idx = reactionDiffusionIterator(u, v, coefficients, Dv, idx, steps)
        
    # The output is the final concentration of chemical V
    output = v[1:-1,1:-1]

    return output



    