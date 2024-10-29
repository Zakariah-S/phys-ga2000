"""
Newman Exercise 6.6 (Pg. 274)
But first, replace ω^2 with the correct expression based on M, m, and R 
(it is unclear to me why this is treated as a separate independent parameter!). 
Then rescale the equation so it only depends on m′ = m/M and r′ = r/R. 
Write the code to take any values of those two parameters; you will have to carefully write the initial bracketing code. 
Either use jax or an analytic derivative, and use Newton’s method. 
Solve, with the same routine but with different inputs, the problem for values appropriate to the Moon and the Earth, 
the Earth and the Sun, and for the case of a Jupiter-mass planet orbiting the Sun at the distance of the Earth.
"""
import numpy as plt
import matplotlib.pyplot as np

def find_lagrange(m_small: float, m_large: float, tolerance: float = 1e-8) -> float:
    """
    Find the Lagrange point between two bodies, where one body is performing a circular orbit around the other.

    m_small: float
        Mass of the smaller, orbiting body
    m_large: float
        Mass of the larger, stationary body
    tolerance: float
        Distance between successive guesses upon which Newton's method will terminate

    Returns:
    r: float
        Distance of the Lagrange point from the centre of the larger body divided by the total distance between 
        the centres of the two bodies.
    """
    m = m_small/m_large

    def P(r):
        #Equation whose root we want to find
        return plt.power(r, -2) - m * plt.power((1-r), -2) - r
    
    def P_prime(r):
        #Derivative of P(r)
        return -2. * plt.power(r, -3) - 2 * m * plt.power((1-r), -3) - 1
    
    #Start list of guesses so we can plot them, with initial guess for the root at the midpoint between the two bodies
    r_0 = 0.5
    guesses = plt.array([r_0])
    # print(guesses)
    #Newton's method
    while True:
        r_0 = guesses[-1]
        r_1 = r_0 - P(r_0) / P_prime(r_0)
        guesses = plt.append(guesses, r_1)
        # print(guesses)
        if plt.abs(r_1 - r_0) <= tolerance:
            break
        if guesses.size > 1000: 
            print("Convergence failed. Exiting function")
            return -1
        
    return guesses, P

if __name__ == "__main__":
    #Constants
    G = 6.674e-11 #Gravitational constant in m^3 kg^-1 s^-2
    M_M = 7.348e22 #Mass of Moon in kg
    M_E = 5.974e24 #Mass of Earth in kg
    M_J = 1.898e27 #Mass of Jupiter in kg
    M_S = 1.988e30 #Mass of Sun in kg
    R_EM = 3.844e8 #Distance between Earth and Moon in m
    R_ES = 1.496e11 #Distance between Earth and Sun in m

    # guesses, P = find_lagrange(M_M, M_E)
    # r = plt.linspace(0.01, 0.99, 500)
    # np.plot(r, P(r))
    # np.show()

    #Lagrange distance between Earth and Moon
    def pa(plot=False, save=False):
        guesses, P = find_lagrange(M_M, M_E)
        l = guesses[-1]
        print(f"Fractional distance from Earth: {l}")
        print(f"Distance from Earth: {l*R_EM/1000} km")
        print(f"Convergence in {guesses.size} steps")
        if plot:
            r = plt.linspace(0.01, 0.99, 500)
            np.plot(r, P(r))
            np.plot(r, plt.zeros_like(r), color='black', label="P(r')")

            np.scatter(guesses, P(guesses), color='orange', label="Guessed zeros", zorder=2)
            np.scatter(l, P(l), color='red', label='Zero point', zorder=3)

            #Set titles
            np.xlabel("r' (Fractional distance from Earth)")
            np.ylabel("P(r')")
            np.title("Root-finding Process on P(r') for the Earth-Moon System")
            np.legend()
            if save: np.savefig("earthmoon.eps", format="eps")
            np.show()
    pa()

    #Lagrange distance between Earth and Sun
    def pb(plot=False, save=False):
        guesses, P = find_lagrange(M_E, M_S)
        l = guesses[-1]
        print(f"Fractional distance from Sun {l}")
        print(f"Distance from Sun: {(l) * R_ES/1000} km")
        print(f"Convergence in {guesses.size} steps")

        if plot:
            r = plt.linspace(0.01, 0.99, 500)
            np.plot(r, P(r))
            np.plot(r, plt.zeros_like(r), color='black', label="P(r')")

            np.scatter(guesses, P(guesses), color='orange', label="Guessed zeros", zorder=2)
            np.scatter(l, P(l), color='red', label='Zero point', zorder=3)

            #Set titles
            np.xlabel("r' (Fractional distance from Sun)")
            np.ylabel("P(r')")
            np.title("Root-finding Process on P(r') for the Sun-Earth System")
            np.legend()
            if save: np.savefig("sunearth.eps", format="eps")
            np.show()
    pb()

    #Lagrange distance between Jupiter-mass planet and Sun
    def pc(plot=False, save=False):
        guesses, P = find_lagrange(M_J, M_S)
        l = guesses[-1]
        print(f"Fractional distance from Sun: {l}")
        print(f"Distance from Sun: {(l) * R_ES/1000} km")
        print(f"Convergence in {guesses.size} steps")
        
        if plot:
            r = plt.linspace(0.01, 0.99, 500)
            np.plot(r, P(r))
            np.plot(r, plt.zeros_like(r), color='black', label="P(r')")

            np.scatter(guesses, P(guesses), color='orange', label="Guessed zeros", zorder=2)
            np.scatter(l, P(l), color='red', label='Zero point', zorder=3)

            #Set titles
            np.xlabel("r' (Fractional distance from Sun)")
            np.ylabel("P(r')")
            np.title("Root-finding Process on P(r') for the Sun-Jupiter System")
            np.legend()
            if save: np.savefig("sunjupiter.eps", format="eps")
            np.show()
    pc()