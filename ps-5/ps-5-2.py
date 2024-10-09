"""
Exercise 5.17 in Newman. (Pg. 205)
"""
import numpy as np
import matplotlib.pyplot as plt

#Define the curves we want to plot
def gamma_integrand(x, a):
    """
    Evaluates the integrand of the Gamma function for a given array/float x and value a.
    x: float | numpy.ndarray(float)
    a: float

    Returns: float | numpy.ndarray(float)
    """
    return np.power(x, a-1) * np.exp(-x)

#-----Part A-----#

def part_a():
    #Plot gamma integrand for a = 2, 3, 4
    x = np.linspace(0, 5, 500)
    colors = ["black", "gray", "lightgrey"]
    for a in np.arange(2, 5):
        plt.plot(x, gamma_integrand(x, a), label=f"a = {a}", color=colors[a-2])
        plt.vlines(x= a-1, ymin=0, ymax=gamma_integrand(a-1, a), color='red')

    plt.legend()
    plt.title("Gamma function Integrand for Varying $a$")
    plt.xlabel("x")
    plt.ylabel(r"g(x) = $x^{a-1}e^{-x}$")
    plt.savefig("gamma.eps", format="eps")
    plt.show()

part_a()

#-----Part B-----#
"""
Integrand given by f(x) = x^(a-1)e^(-x).
Derivative given by f'(x) = (a-1)x^(a-2)e^(-x) - x^(a-1)e^(-x) = [(a-1)x^(a-2) - x^(a-1)]e^(-x)
    This derivative is 0 when (a-1)x^(a-2) = x^(a-1) => a-1 = x

Second derivative is given by f''(x) = [(a-1)(a-2)x^(a-3) - (a-1)x^(a-2)]e^(-x) - [(a-1)x^(a-2) - x^(a-1)]e^(-x)

    Then f''(a-1) = [(a-1)(a-2)*(a-1)^(a-3) - (a-1)*(a-1)^(a-2) - (a-1)*(a-1)^(a-2) + (a-1)^(a-1)]e^(-x)
    But [(a-1)(a-2)*(a-1)^(a-3) - (a-1)*(a-1)^(a-2) - (a-1)*(a-1)^(a-2) + (a-1)^(a-1)]
        = (a-1)(a-2)*(a-1)^(a-3) - (a-1)*(a-1)^(a-2) = (a-2)*(a-1)^(a-2) - (a-1)*(a-1)^(a-2)
        = [a - 2 - a + 1]*(a-1)^(a-2) = -(a-1)^(a-2).

        Assuming that a > 1:
            -(a-1)^(a-2) < 0 always
        If a = 1:
            f'(x) = -e^(-x), doesn't have a local maximum/minimum
        If a < 1:
            By inspection, f(x) will be monotically decreasing
"""
#-----Part C-----#
"""
If we're making the substitution z = x / (c + x), which translates the range [0, infinity] to [0, 1],
then z = 1/2 when x = c. We want the midpoint z = 1/2 to occur at around the maximum of our integrand
(as we hope this will be the rough midpoint of the integral), so we choose c = a - 1, as per Part B.

Then zc + zx = x => zc = x(1 - z) => x = z * c / (1 - z) = (z/(1-z)) * (a-1)
=> dx = c[(1-z) + z]dz / (1 - z)^2 = [c / (1 - z)^2]dz = [(a - 1) / (1 - z)^2]dz
"""
#-----Part D-----#
"""
To make the calculation of the gamma function easier on the computer (i.e. avoid integer over-/underflow errors),
we should rewrite f(x) = x^(a-1)e^(-x) in a form such that the two terms being multiplied don't grow as far apart in magnitude.

We can do this by replacing x^(a-1) with e^(a-1)ln(x). This results in the following form:
f(x) = e^(a-1)ln(x) * e^(-x) = e^[(a-1)ln(x) - x].
This form of f(x) only has one exponential term e^g(x). It should behave better with the computer, because
g(x) is a subtraction between terms that aren't generally far apart, meaning catastrophic subtraction won't generally happen.
Additionally, g(x) tends to -infinity for both x = 0 and x = 1, without becoming a very small number over long portions of that range.
The chance for integer overflow is certainly still there, but it won't matter as much because for all large values of g(x), 
the resulting f(x) will be close to 0.
"""
#-----Part E-----#

def func_rescale(func, z, c=1):
    """
    Returns values at xp of a rescaled integrand function for new limits of 0 to 1, where the function is even and has 
    initial range [0, infinity]

    func : function(float)
        Function that we want to rescale. Should have the initial range [0, infinity].

    xp : float | numpy.ndarray(float)
        Input parameter (in new limits' coordinates)

    c: float
        Arbitrary parameter. Should be set to the expected midpoint of the integral for maximum integration accuracy.

    Returns: float | numpy.ndarray(float)
        Output of rescaled function
"""
    x = c * z / (1 - z)
    return (c / np.square(1 - z)) * func(x)

def gamma(a, N=100, z=0, weights=None):
    """
    Returns the approximate value of the gamma function at float a.
    a: float
        Argument of Gamma function
    N: int
        Number of sample points for Gauss-Legendre quadrature
    z: np.ndarray[float]
        Gauss-Legendre sample points. If you don't provide them, the function will do it for you.
    weights: np.ndarray[float]
        Gauss-Legendre weights corresponding to each provided z. Don't bother with these if you aren't providing z.
    """
    if a == 1.: return 1. #Trivial case. All a > 1. are covered by what follows

    #Define Gamma integrand for our chosen value of a
    def gamma_integrand_alt(x):
        """
        Evaluates the integrand of the Gamma function, with the revised form f(x) = e^[(a-1)ln(x) - x].
        x: float | numpy.ndarray(float)
        a: float

        Returns: float | numpy.ndarray(float)
        """
        return np.exp((a-1) * np.log(x) - x)
    
    #Get roots and weights for Gauss-Legendre
    if isinstance(z, int):
        z, weights = np.polynomial.legendre.leggauss(N)
        #Only use roots in the range [0, 1]
        start_index = int(N/2) if N%2 == 0 else int((N-1)/2)
        z = z[start_index:]
        weights = weights[start_index:]

    #Debugging
    # print(z)
    # print((a-1) * z / (1 - z))
    # print(func_rescale(gamma_integrand_alt, z, c=a-1))

    #Evaluate integral and return
    gauss_integral = (func_rescale(gamma_integrand_alt, z, c=a-1) * weights).sum()
    return gauss_integral

def part_e():
    print(gamma(1.5)) #Should be ~0.8862269254527579
    # >> 0.8863478841664341
    print(np.abs(gamma(1.5) - np.sqrt(np.pi)/2) / (np.sqrt(np.pi)/2))
    # >> 0.00013648729259080775
part_e()

#-----Part F-----#

def part_f():
    from math import factorial
    print(gamma(3)) #Should be 2! = 2
    # >> 1.9999998885682069
    print(gamma(6)) #Should be 5! = 120
    # >> 120.00000005608909
    print(gamma(10)) #Should be 9! = 362880
    # >> 362880.00000002544
    for i in [3, 6, 10]:
        print(np.abs(gamma(i) - factorial(i-1)) / factorial(i))
        # >> 1.857196552101925e-08
        # >> 7.79015173356533e-11
        # >> 7.009685796733151e-15
part_f()

#-----For my own curiosity's sake-----#
def plot_resids(low=1, high=10, step = 1, N=100):
    #Plot difference between Gamma function and actual factorial
    from math import factorial
    fig, axs = plt.subplots(2, 1)

    #Get roots of Legendre function and associated weights
    z, weights = np.polynomial.legendre.leggauss(N)
    #Only use roots (and their corresponding weights) if roots are in the range [0, 1]
    start_index = int(N/2) if N%2 == 0 else int((N-1)/2)
    z = z[start_index:]
    weights = weights[start_index:]

    a = np.arange(low, high, step)
    gam = np.zeros_like(a, dtype=np.float64)
    fac = np.zeros_like(a, dtype=np.float64)

    i = 0
    while i < a.size:
        gam[i] = gamma(a[i], z=z, weights = weights)
        fac[i] = factorial(int(a[i])-1)
        print(a[i], fac[i], gam[i])
        i += 1

    axs[0].plot(a, gam, label="Gamma function")
    axs[0].scatter(a, fac, label="Factorial", color='red')
    axs[0].set_ylabel("Function value")
    axs[0].set_title(r"$\Gamma(a)$ approximation vs. a")
    axs[0].legend()

    axs[1].plot(a, gam - fac)
    axs[1].set_ylabel(r"Residual $\Gamma(a) - a!$")
    axs[1].set_xlabel("a")

    plt.show()

# plot_resids(high=15)