'''
Exercise 5.9 in Newman (Page 172)
Then: (c) Test the convergence by evaluating the choices N = 10, 20, 30, 40, 50, 60, 70.
'''
import numpy as np
import matplotlib.pyplot as plt

#-----Part A-----#
def func_rescale(func, xp=None, range=None):
    """
    From Blanton (2024) Computational Physics Jupyter Notebook on Integration:

    Returns values at xp of a rescaled function for new limits of -1 to 1
    
    Parameters
    ----------
    xp : np.float32 or np.float64
        input parameter (in new limits' coordinates)

    range : list or np.array
        [2] low and high limits of range to map to -1 to 1

    Returns
    -------
    fn : np.float32 or np.float64
        output of rescaled function
"""
    weight = (range[1] - range[0]) * 0.5
    x = range[0] + 0.5 * (range[1] - range[0]) * (xp + 1.)
    return(weight * func(x))

def cv(T, N=50, DT=428, num=6.022e25):
    """
    Return Debye's constant-volume heat capacity for a substance, with N points used for the involved integration.
    This value is given by the formula
    c_v = 9 * N * k_B * (T/DT)^3 * int(from 0 to (DT/T)) [(x^4 * e^x) / (e^x - 1)^2]dx.

    T: float | numpy.ndarray[float]
        Temperature(s) of the substance
    N: int
        Number of sample points used for the integral.
    DT: float
        DeBye temperature of the substance.
    num: (large) float
        Number of atoms
    """

    #Get T into the form of an array
    if isinstance(T, float) or isinstance(T, int):
        T = np.array([T], dtype=np.float64)
    else:
        T = np.array(T, dtype=np.float64)

    #Set up our output array
    cv = np.zeros_like(T, dtype=np.float64)

    #Get the Legendre roots and associated weights to use for Gaussian quadrature
    xp, weights = np.polynomial.legendre.leggauss(N)
    
    #Loop through the T array and compute cv(T) for each
    i = 0
    while i < T.size:
        #Set up range so we can feed it into func_rescale
        range = np.array([0, DT/T[i]], dtype=np.float64)

        #Define the expression we want to integrate
        def int_formula(x):
            return np.power(x, 4) * np.exp(x) / np.power((np.exp(x) - 1), 2)

        #Compute the integral by dotting the rescaled function values with the weights
        gauss_integral = (func_rescale(int_formula, xp, range) * weights).sum()
        cv[i] = gauss_integral

        i += 1

    return T, cv

#-----Part B-----#

def part_b():
    Ts, cvs = cv(np.arange(5, 500))
    plt.plot(Ts, cvs)
    plt.show()

# part_b()

#-----Part C-----#

def part_c(Ns):
    with open("cv_data.txt", 'w') as f:
        header_string = "temps"
        temps = np.arange(5, 500)
        data_arr = np.zeros((len(Ns)+1, temps.size))

        data_arr[0] = temps
        i = 1
        for N in Ns:
            header_string += f"\t{N}"
            Ts, cvs = cv(temps, N)
            plt.plot(Ts, cvs, label=f"{N} sample points")
            data_arr[i] = cvs
            i += 1

        f.write(header_string + "\n")
        data_arr = data_arr.T
        for row in data_arr:
            new_string = f"{row[0]}"
            for i in range(len(row) - 1):
                new_string += f"\t{row[i+1]}"
            f.write(new_string + "\n")

    plt.legend()
    plt.title("$C_V$ vs. Temperature")
    plt.xlabel("Temperature (K)")
    plt.ylabel("$C_V$")

    plt.savefig("cv_curves.eps", format="eps")
    plt.show()

part_c(Ns=[10, 20, 30, 40, 50, 60, 70])