'''
Exercise 5.10 in Newman (Page 173)
'''
import numpy as np
import matplotlib.pyplot as plt

#-----Part B-----#
def func_rescale(func, xp=None, range=None):
    """
    From Blanton (2024) Computational Physics Jupyter Notebook on Integration:

    Returns values at xp of a rescaled function for new limits of -1 to 1
    
    Parameters
    ----------
    func : function(float)
        Function that we want to rescale

    xp : np.float32 or np.float64
        input parameter (in new limits' coordinates)

    range : list or np.array
        [2] low and high limits of range to map to -1 to 1

    Returns
    -------
    fn : np.float32 or np.float64
        output of rescaled function
"""
    x = range[0] + 0.5 * (range[1] - range[0]) * (xp + 1.)
    return((range[1] - range[0]) * 0.5 * func(x))

def get_period(a, V, m=1, N=20):
    """
    a: float | int | numpy.ndarray(float | int)
        Starting amplitude of the oscillating particle
    V: func(x)
        Function that gives the potential of a particle for a location x.
        The function V should only take one parameter.
    m: float
        Mass of the particle.
    N: int
        Number of sample points used for Gaussian quadrature.

    Returns:
    a: numpy.ndarray[float]
        Array with initial amplitude value(s)
    T: numpy.ndarray[float]
        Array with corresponding periods
    """

    #Get a into the form of an array
    if isinstance(a, float) or isinstance(a, int):
        a = np.array([a], dtype=np.float64)
    else:
        a = np.array(a, dtype=np.float64)

    #Set up our output array
    T = np.zeros_like(a, dtype=np.float64)

    #Get the Legendre roots and associated weights to use for Gaussian quadrature
    xp, weights = np.polynomial.legendre.leggauss(N)
    
    #Loop through the a array and compute T(a) for each
    i = 0
    while i < a.size:
        print(i)
        #Define the expression we want to integrate
        def int_formula(x):
            return np.sqrt(1 / (V(a[i]) - V(x)))
        
        #Set up range so we can feed it into func_rescale
        range = np.array([0, a[i]], dtype=np.float64)
        # print(func_rescale(int_formula, xp, range))

        #Compute the integral by dotting the rescaled function values with the weights
        gauss_integral = (func_rescale(int_formula, xp, range) * weights).sum()
        print(gauss_integral)
        T[i] = np.sqrt(8 * m) * gauss_integral

        i += 1

    return a, T

def part_b(m=1):
    #Plot the T(a) curve
    a, T = get_period(a=np.linspace(0, 2, 100), V=lambda x: np.power(x, 4))
    plt.plot(a, T)
    plt.title("Period of the $V(x) = x^4$ Oscillator vs. Starting Amplitude")
    plt.xlabel("Amplitude (m)")
    plt.ylabel("Period (s)")
    plt.savefig("period_curve.eps", format="eps")
    plt.show()

part_b()