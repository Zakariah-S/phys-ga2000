"""
Exercise 5.17 in Newman. (Pg. 205)
"""
import numpy as np
import matplotlib.pyplot as plt

#-----Part A-----#

def part_a():
    #Define the curves we want to plot
    def gamma_integrand(x, a):
        """
        Evaluates the integrand of the Gamma function for a given array/float x and value a.
        x: float | numpy.ndarray(float)
        a: float

        Returns: numpy.ndarray(float)
        """
        return np.power(x, a-1) * np.exp(-x)
    
    #Plot each curve for a = 2, 3, 4
    x = np.linspace(0, 5, 500)
    colors = ["black", "gray", "lightgrey"]
    for a in np.arange(2, 5):
        plt.plot(x, gamma_integrand(x, a), label=f"a = {a}", color=colors[a-2])

    plt.legend()
    plt.title("Gamma function Integrand for Varying $a$")
    plt.xlabel("x")
    plt.ylabel(r"$x^{a-1}e^{-x}$")
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
