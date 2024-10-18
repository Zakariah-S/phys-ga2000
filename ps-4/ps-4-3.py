'''
Exercise 5.13 in Newman (Page 182)
Then: (d) Perform the calculation using Gauss-Hermite quadrature (scipy can give you the right roots and weights to use). 
Can you make an exact evaluation (meaning zero approximation error) of the integral?
'''
import numpy as np
import matplotlib.pyplot as plt
from math import factorial

#-----Part A-----#
def H(n, x):
    '''
    Returns the nth Hermite polynomial, evaluated at the value(s) x.
    n: int
        Number of the Hermite polynomial.
    x: float | numpy.ndarray(float)
    '''
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    else:
        return 2 * x * H(n-1, x) - 2 * (n-1) * H(n-2, x)

def nth_ho(n, x):
    #Return the nth wavefunction of the 1-D harmonic oscillator, evaluated at x
    return (1/np.sqrt(np.power(2, n) * factorial(n) * np.sqrt(np.pi))) * np.exp(-0.5 * np.square(x)) * H(n, x)

def part_a(Ns = np.arange(0, 4), xlims=np.array([-4, 4])):
    x = np.linspace(xlims[0], xlims[1], 100)
    plt.hlines(0, xlims[0], xlims[1], color='k')
    for N in Ns:
        plt.plot(x, nth_ho(N, x), label=f"N = {N}")

    #checking accuracy
    # plt.plot(x, (1/np.sqrt(np.power(2, 0) * factorial(0) * np.sqrt(np.pi))) * (1) * np.exp(-0.5*x**2), ls='dashed', label='expected n=0')
    # plt.plot(x, (1/np.sqrt(np.power(2, 1) * factorial(1) * np.sqrt(np.pi))) * (2*x) * np.exp(-0.5*x**2), ls='dashed', label='expected n=1')
    # plt.plot(x, (1/np.sqrt(np.power(2, 2) * factorial(2) * np.sqrt(np.pi))) * (4*x**2 - 2) * np.exp(-0.5*x**2), ls='dashed', label='expected n=2')
    # plt.plot(x, (1/np.sqrt(np.power(2, 3) * factorial(3) * np.sqrt(np.pi))) * (8*x**3 - 12*x) * np.exp(-0.5*x**2), ls='dashed', label='expected n=3')

    plt.legend()
    plt.title("Wavefunctions of the 1-D Harmonic Oscillator")
    plt.xlabel("x")
    plt.ylabel("$\psi$")
    plt.savefig("ho_curves.eps", format='eps')
    plt.show()

# part_a()

#-----Debugging-----#
def check_hermites():
    x = np.linspace(-4, 4, 200)
    print(x)
    print(H(0, x))
    for i in [0, 1, 2, 3]:
        plt.plot(x, H(i, x), label=f"Calculated $H_{i}$")
    plt.plot(x, np.ones_like(x), label="$H_0$", ls='dashed')
    plt.plot(x, 2*x, label="$H_1$", ls='dashed')
    plt.plot(x, 4*x**2 - 2, label="$H_2$", ls='dashed')
    plt.plot(x, 8*x**3 - 12*x, label="$H_3$", ls='dashed')
    plt.legend()
    plt.show()
# check_hermites()

#-----Part B-----#
def part_b(N = 30, xlims=np.array([-10, 10])):
    x = np.linspace(xlims[0], xlims[1], 1000)
    plt.hlines(0, xlims[0], xlims[1], color='k')
    plt.plot(x, nth_ho(N, x))

    plt.hlines(0, xlims[0], xlims[1], color='k')
    plt.title(f"{N}th Wavefunction of the 1-D Harmonic Oscillator")
    plt.xlabel("x")
    plt.ylabel("$\psi$")
    plt.savefig("ho_curve.eps", format='eps')
    plt.show()

# part_b()

#-----Part C-----#

def nth_ho_no_const(n, x):
    #Returns nth wavefunction but without the constant factor in the front
    return np.exp(-0.5 * np.square(x)) * H(n, x)

def func_rescale_infinite(func, xp=None, q=1):
    """
    Returns values at xp of a rescaled function for new limits of -1 to 1, where the function is even and has 
    initial range [0, infinity]

    Parameters
    ----------
    func : function(float)
        Function that we want to rescale. Should have the initial range [0, infinity].

    xp : np.float32 or np.float64
        input parameter (in new limits' coordinates)

    range : list or np.array
        [2] low and high limits of range to map to -1 to 1

    Returns
    -------
    fn : np.float32 or np.float64
        output of rescaled function
"""
    x = q * (1. + xp) / (1. - xp)
    return (2.*q / np.square(1.-xp)) * func(x)

def get_ex_x_sq(n, N=100, q=2.3):
    '''
    Evaluate <x^2> for the nth harmonic oscillator wavefunction, using Gaussian quadrature.
    n: int
        Number of the wavefunction, with energy E(n) = hw(n + 1/2).
    N: int
        Number of sample points used for Gaussian quadrature.
    
    Returns:
    ex_x_sq: float
        Expected value of x^2 for the wavefunction.
    rms_x: float
        Square root of the expected value of x^2 for the wavefunction.
    '''

    #Get the Legendre roots and associated weights to use for Gaussian quadrature
    xp, weights = np.polynomial.legendre.leggauss(N)

    #Define the expression we want to integrate (square of the nth ho wavefunction * x^2)
    def int_formula(x):
        return np.square(x * nth_ho_no_const(n, x))

    #Compute the integral (from 0 to infinity) by dotting the rescaled function values with the weights
    gauss_integral = (func_rescale_infinite(int_formula, xp, q=q) * weights).sum()

    #Multiply by the prefactors and by 2 to get Ex[x^2]
    ex_x_sq = (1/(np.power(2., n) * factorial(n) * np.sqrt(np.pi))) * 2 * gauss_integral

    #Get rms[x]
    rms_x = np.sqrt(ex_x_sq)
    
    print(f"<x^2> = {ex_x_sq}\nsqrt(<x^2> = {rms_x})")

# get_ex_x_sq(n=5, q=5.5)
# >> <x^2> = 5.500000000000023
# >> x-rms = 2.3452078799117198

#-----Part D-----#

def get_ex_x_sq_gh(n, N=100):
    '''
    Evaluate <x^2> for the nth harmonic oscillator wavefunction, using Gauss-Hermite quadrature
    n: int
        Number of the wavefunction, with energy E(n) = hw(n + 1/2).
    N: int
        Number of sample points used for Gaussian quadrature.
    
    Returns:
    ex_x_sq: float
        Expected value of x^2 for the wavefunction.
    rms_x: float
        Square root of the expected value of x^2 for the wavefunction.
    '''
    from scipy.special import roots_hermite

    #Get the roots and associated weights to use for Gauss-Hermite quadrature
    xp, weights = roots_hermite(N)

    #Define the expression we want to integrate (square of the nth Hermite * x^2)
    def int_formula(x):
        return np.square(x * H(n, x))

    #Compute the integral (from -infinity to infinity) by dotting the function values with the weights
    gauss_integral = (int_formula(xp) * weights).sum()

    #Multiply by the prefactor to get the negative range, and return
    rms_x = (1/np.sqrt(np.power(2., n) * factorial(n) * np.sqrt(np.pi))) * np.sqrt(gauss_integral)
    return rms_x

# print(get_ex_x_sq_gh(n=5, N=100))
# >> <x^2> = 5.499999999999995, 
# >> x-rms = 2.3452078799117135

#Get errors
print((5.500000000000023 - 5.5)/5.5)
print((5.5-5.499999999999995)/5.5)

print(np.polynomial.legendre.leggauss(10)[1])