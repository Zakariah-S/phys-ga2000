"""
Implement Brent's 1D minimization method. Test it on this function: y = (x − 0.3)^2 * exp(x). 
Compare to the scipy.optimize.brent implementation results.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brent as scibrent

def brent(f, start, end, tolerance=1e-8):
    """
    Use Brent's 1-D minimization method to find the minimum of a function of a single variable 
    along the interval [start, end].
    f: func
        Function for which the minimum should be found.
    start: float
        Beginning of the initial bracket.
    end: float
        End of the initial bracket.
    tolerance: float
        For successive guesses x0 and x1, x1 will be considered a minimum point of the function if |x1 - x0| <= tolerance

    Return: float
        Minimum of the function, within the given tolerance
    """

    def golden_section(a, c):
        """
        Use Golden Section Search to cut down the bracketing interval.
        Return the boundaries of the new bracket, followed by the 'golden point' x of the new bracket.
        """
        w = (3. - np.sqrt(5))/2. #About 0.382
        b = a + w * (c - a)
        x = b + (1 - 2.*w) * (c - a)

        if f(x) <= f(b):
            return b, c, x
        else:
            return a, b, x

    def parab_min(a, c, b=None):
        """
        Use parabolic interpolation to guess the minimum of the function. 
        Based on the guess, return the boundaries of a new bracketing interval, followed by the guess for the minimum.
        """
        #Initial guess using parabolic approximation
        a = min(a, c)
        if not b: b = (a + c) / 2
        c = max(a, c)
        x = b - 0.5 * (np.square(b-a) * (f(b) - f(c)) - np.square(b-c) * (f(b) - f(a))) / ((b-a) * (f(b) - f(c)) - (b-c)*(f(b) - f(a)))

        #Update bracket based on relative values of f(a), f(b), f(x), f(c)
        if f(x) > f(b) or x < a or x > c:
            #Parabolic minimization did not work if f(x) is greater than f(b) or if x is outside [a, b]
            #If it doesn't work, revert to Golden Section Search (see above function)
            return golden_section(a, c)
        
        #Determine the next bracket appropriately, setting the new guess for the minimum at the midpoint (arbitrary)
        if x <= b:
            return a, b, (a+b)/2
        else:
            return b, c, (b+c)/2
    
    a, c, new_guess = parab_min(start, end)

    #Initialise array of guesses that I can plot later
    guesses = np.array([new_guess])
    # print(guesses)
    
    while True:
        a, c, new_guess = parab_min(a, c, b=new_guess)
        guesses = np.append(guesses, new_guess)
        # print(guesses)
        if np.abs(new_guess - guesses[-2]) <= tolerance:
            return guesses
    
if __name__ == "__main__":
    def g(x):
        return np.square(x-0.3) * np.exp(x)
    
    #Find the minimum of g(x) using my Brent routine
    guesses = brent(g, 0., 1., tolerance=1e-8)
    print(f"Minimum at x = {guesses[-1]}")
    # >> Minimum at x = 0.30000000447034836
    print(f"Relative Error: {(guesses[-1] - 0.3)/0.3}")
    # >> Relative Error: 1.490116123085509e-08

    #See what the minimum is using SciPy's Brent routine
    min = scibrent(g, brack=(0., 1.), tol=1e-8)
    print(f"From SciPy Brent: minimum at x = {min}")
    # >> From SciPy Brent: minimum at x = 0.300000000023735
    print(f"Relative Error: {(min - 0.3)/0.3}")
    # >> Relative Error: 7.91167131808379e-11

    #Make a nice figure
    x = np.linspace(0.1, 0.5, 300)
    plt.plot(x, g(x), label=r"$f(x) = (x-0.3)^2e^x$")
    plt.scatter(guesses[0], g(guesses[0]), color='red', marker='.', zorder=3, label="Starting guess")
    plt.plot(guesses, g(guesses), color='orange', zorder=2, label="Guesses", lw=1)
    plt.scatter(guesses[-1], g(guesses[-1]), color='green', zorder=3, label="Final guess")
    plt.vlines(x=0.3, ymin=plt.ylim()[0], ymax=plt.ylim()[-1], color='lightgray', label="Actual minimum")

    plt.xlabel("x")
    plt.ylabel(r"$f(x)$")
    plt.title("Convergence of Brent's Method to a 1-D Minimum")
    plt.legend(loc='upper right')

    plt.savefig("brentmin.eps", format="eps")
    plt.show()