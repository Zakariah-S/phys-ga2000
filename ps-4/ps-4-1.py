'''
Exercise 5.9 in Newman (Page 172)
Then: (c) Test the convergence by evaluating the choices N = 10, 20, 30, 40, 50, 60, 70.
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#-----Part A-----#
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
    
    #Define the expression we want to rescale and integrate
    def int_formula(x):
        return np.power(x, 4) * np.exp(x) / np.power((np.exp(x) - 1), 2)
    
    #Loop through the T array and compute cv(T) for each
    i = 0
    while i < T.size:
        #Set up range so we can feed it into func_rescale
        range = np.array([0, DT/T[i]], dtype=np.float64)

        #Compute the integral by dotting the rescaled function values with the weights
        gauss_integral = (func_rescale(int_formula, xp, range) * weights).sum()
        cv[i] = gauss_integral

        i += 1

    return T, 9 * num * 1.380649e-23 * np.power(T/DT, 3) * cv

#-----Part B-----#

def part_b():
    #Plot the c_V(T) curve
    Ts, cvs = cv(np.arange(5, 501))
    plt.plot(Ts, cvs)
    plt.title("Heat Capacity $C_V$ vs. Temperature")
    plt.xlabel("Temperature (K)")
    plt.ylabel("$C_V$ (J/K)")
    plt.savefig("cv_curve.eps", format="eps")
    plt.show()

part_b()

#-----Part C-----#
#Plot C_V vs. T for each N 
#(deprecated because I decided to instead plot the error in the Gaussian quadrature integral vs. N below)
def part_c(Ns):
    with open("cv_data.txt", 'w') as f:
        header_string = "temps"
        temps = np.arange(5, 501)
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
    plt.ylabel("$C_V$ (J/K)")

    plt.savefig("cv_curves.eps", format="eps")
    plt.show()

part_c(Ns=[10, 20, 30, 40, 50, 60, 70])

#-----Part C-----#

def part_c_again(Ns, temps = np.arange(5, 501)):
    Ns = np.array(Ns, dtype=np.uint32)
    data_arr = np.zeros((len(Ns), temps.size))

    i = 0
    while i < Ns.size:
        data_arr[i] = cv(temps, Ns[i])[1]
        i += 1
    
    data_arr = data_arr.T

    cmap = plt.get_cmap('coolwarm')
    colors = [cmap(i) for i in np.linspace(0, 1, temps.size)]

    #Set up figure with a small window on the side for a colorbar
    fig, axs = plt.subplots(1, 2, width_ratios=[40, 1])
    i = 0
    while i < temps.size:
        best_val = data_arr[i][-1]
        axs[0].plot(Ns, ((data_arr[i] - best_val) / best_val), color=colors[i])
        i += 1
    
    axs[0].set_title("Error in $C_V$ vs. number of sample points N")
    axs[0].set_xlabel("N")
    axs[0].set_ylabel(r"$\frac{c_v(N) - c_v}{c_v}$")

    #Set up colorbar with the cmap we used for each curve
    norm = mpl.colors.Normalize(vmin=temps[0], vmax=temps[-1])
    cb1 = mpl.colorbar.ColorbarBase(axs[1], cmap=cmap, norm=norm)
    cb1.set_label('Temperature (K)')

    plt.savefig("cv_errors.eps", format="eps")
    plt.show()
    print(data_arr.shape)

part_c_again(Ns=[10, 20, 30, 40, 50, 60, 70])