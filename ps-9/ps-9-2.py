"""
Exercise 8.7 in Newman. (Pg. 352)
Before approaching the problem numerically, 
show that set of possible values of R, ρ, C, m, and g maps to a one-parameter family of solutions, 
which in terms of a typical timescale T is controlled by the combination R^2ρCgT^2/m. 
That is, show that if you rescale variables from t to t′/T and from x to an appropriate x′, 
you get a unitless set of equations with one free (unitless) parameter.
"""
import numpy as np
import matplotlib.pyplot as plt

#-----Part A-----#

def rk4(f, T, dT, X):
    """
    Evaluate one 4th-order Runge-Kutta step of a set of entangled ordinary differential equations (same as from ps-9-1.py).
    f: func
        Function that takes in X and T variables, 
        returning an array of variables y_i that satisfies dX_i/dt = f_i(X, T) = y_i for each X_i.
    T: float
        Beginning time of time step
    dT: float
        Length of the time step
    X: numpy.ndarray[float]
        Values of each dependent variable at time T; x_i(T)
    """
    x_new = np.zeros_like(X)

    k1 = dT * f(*X, T)
    k2 = dT * f(*(X + 0.5*k1), T + 0.5*dT)
    k3 = dT * f(*(X + 0.5*k2), T + 0.5*dT)
    k4 = dT * f(*(X + k3), T + dT)

    x_new = X + (1/6)*k1 + (1/3)*k2 + (1/3)*k3 + (1/6)*k4
    
    # print("***")
    # print(X)
    # print(x_new)
    return x_new

def cannonball(v0, theta0, dt= 0.01, R = 0.08, m = 1., rho = 1.22, C = 0.47, g = 9.81, verbose = False) -> None:
    """
    Find the trajectory of a spherical cannonball that experiences drag force.
    v0: float
        Initial speed of the cannonball, in m/s.
    theta0: float
        Angle between the cannonball's initial direction of motion and the ground. 
        Should in radians and between 0 and pi/2.
    dt: float
        Length of a time step, in s.
    R: float
        Radius of the cannonball, in m.
    m: float
        Mass of the ball, in kg.
    rho: float
        Density of whatever the ball is travelling through, in kg/m^3.
        (Generally the density of air is 1.22 kg/m^3).
    C: float
        Coefficient of drag for the cannonball.
        For a sphere this is equal to 0.47, so don't change it 
        if you're interested in working with a spherical cannonball!
    g: float
        Acceleration due to gravity (gravitational field assumed constant), in m/s^2.
        On Earth, this value is 9.81 m/s^2.
    """
    #Define our characteristic time scale T and our one actual parameter alpha.
    T = 2 * v0 * np.sin(theta0) / g #Time it would take the ball to hit the ground without drag
    a = (np.pi/2) * np.square(R) * rho * C * g * np.square(T) / m

    #The time that it takes the ball to fall to Earth should not exceed T by a lot, 
    # so I aim to integrate over the interval 2T.

    steps = int(2 * T / dt)

    #Scaled, unitless t array
    dt_rescale = dt / T
    t = np.arange(0, steps) * dt_rescale

    #x, y, vx, vy arrays
    r = np.zeros((4, t.size))

    #Set up initial values, rescaling appropriately
    #Initial x
    r[0][0] = 0.
    #Initial y
    r[1][0] = 0.
    #Initial vx
    r[2][0] = v0 * np.cos(theta0) / (g * T)
    #Initial vy
    r[3][0] = v0 * np.sin(theta0) / (g * T)

    def f(x, y, vx, vy, t):
        v = np.hypot(vx, vy)
        return np.array([vx, vy, -a * vx * v, -a * vy * v - 1])
    
    i = 0
    while i < t.size-1:
        r[:,i+1] = rk4(f, t[i], dt_rescale, r[:,i])
        i += 1
        if r[1][i] < 0:
            #If y < 0, break because the cannonball has hit the ground
            break

    #Remove all indices after the point where the ball hit the ground
    r = r[:, :i+1]
    t = t[:i+1]

    #Give everything their units back
    t *= T
    r[:2] *= g * np.square(T)
    r[2:] *= g * T

    if verbose:
        #Print some results
        print(f"Time of flight:\t{t[-1]}.")
        print(f"Distance travelled along ground:\t{r[0][-1]}.")
        print(f"Height reached:\t{np.max(r[1])}.")

    return *r, t

def parta():
    #Plot a single trajectory
    x, y, vx, vy, t = cannonball(v0 = 100., theta0 = 30. * np.pi/180., verbose=True)
    plt.plot(x, y)

    plt.xlim(left=0.)
    plt.ylim(bottom=0.)

    plt.xlabel("$x(t)$, Distance Along Ground to Starting Point (m)")
    plt.ylabel("$y(t)$, Height Above Ground (m)")
    plt.title("Trajectory of a Cannonball with Drag")

    plt.savefig("trajectory.eps", format="eps")
    plt.show()
parta()

#-----Part B-----#
def partb(masses, v0=100., theta0=30., option: str = 'a', g=9.81):
    """
    Investigate dependece of distance travelled on mass.
    masses: list[float]
        Series of masses for which we want to find the distance travelled.
    v0: float
        Initial speed of each simulated cannonball, in m/s.
    theta0: float
        Angle between the cannonball's initial velocity and the plane of the ground, in rad.
    option: string
        If set to 'a': Plot trajectories for a series of masses given by masses.
        If set to 'b': Plot distance travelled vs. mass for a series of masses given by masses.
    """
    if option == 'a':
        #Plot trajectories for a series of masses
        for m in masses:
            x, y, vx, vy, t = cannonball(v0 = v0, theta0 = theta0 * np.pi/180., m=m)
            plt.plot(x, y, label=f"$m = {round(m, 2)}~kg$")
        x, y, vx, vy, t = cannonball(v0 = 100., theta0 = 30. * np.pi/180., m=m, C = 0.)
        plt.plot(x, y, label="No drag", c='black', ls='dashed')

        plt.xlim(left=0.)
        plt.ylim(bottom=0.)

        plt.xlabel("$x(t)$, Distance Along Ground to Starting Point (m)")
        plt.ylabel("$y(t)$, Height Above Ground (m)")
        plt.title("Trajectory of a Cannonball with Drag")
        plt.legend(loc='upper right')

        plt.savefig("trajectories.eps", format="eps")
        plt.show()

    if option == 'b':
        #Find distance for a large number of masses, and plot vs. mass
        masses = np.asarray(masses)
        x_max = np.zeros_like(masses)

        i = 0
        while i < masses.size:
            x, y, vx, vy, t = cannonball(v0 = v0, theta0 = theta0 * np.pi/180., m=masses[i])
            x_max[i] = x[-1]
            i += 1

        plt.plot(masses, x_max, label="With air resistance")

        #Also plot the distance travelled by a cannonball without drag
        x_nodrag = np.square(v0) * np.sin(2. * theta0 * np.pi/180.) / g
        plt.plot(masses, np.ones_like(x_max) * x_nodrag, ls='dashed', c='black', label="Without air resistance")

        plt.title("Landing Distance vs. Cannonball Mass")
        plt.xlabel("Mass (kg)")
        plt.ylabel("Distance travelled (m)")
        plt.legend()

        plt.savefig("distancemass.eps", format='eps')
        plt.show()

partb(masses = np.geomspace(1., 1000., 7), option='a')
partb(masses = np.arange(10., 1001., 10.), option='b')