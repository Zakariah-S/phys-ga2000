"""
Exercise 8.6 in Newman. (Pg. 351)
"""
import numpy as np
import matplotlib.pyplot as plt

#-----Part A-----#
def rk4(f, T, dT, X):
    """
    Evaluate one 4th-order Runge-Kutta step of a set of entangled ordinary differential equations.
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

def sho(steps: int = 1001, start=0., end=50., w: float = 1., x0: float = 1., v0: float = 0.):
    #Decompose ODE: d^2x/dt^2 = -w^2 x
    #Resulting equations:
    #   dx/dt = v = fx(x, v, t)
    #   dv/dt = -w^2 x = fv(x, v, t)

    def f(x, v, t):
        return np.array([v, -np.square(w) * x])

    t = np.linspace(start, end, steps)
    dt = t[1]-t[0]

    x = np.zeros_like(t)
    v = np.copy(x)

    #Initial conditions
    x[0] = x0
    v[0] = v0

    i = 0
    while i < t.size-1:
        x[i+1], v[i+1] = rk4(f, t[i], dt, np.array([x[i], v[i]]))
        i += 1

    return t, x, v

def parta(x0 = 1., w = 1.):
    x0 = 1.
    w = 1.

    t, x, v = sho(x0=x0, w=w)
    plt.plot(t, x, label="Approximated Position")
    plt.plot(t, x0 * np.cos(w*t), label="Analytic Position", ls='dashed')

    #Residuals of position
    # plt.plot(t, x - np.cos(w*t), label="Residuals")

    plt.xlabel("$t$ (units not given)")
    plt.ylabel("$x(t)$ (units not given)")
    plt.title(r"Simple Harmonic Oscillator for $x(0) = " + f"{x0}" + r"$")
    plt.legend()

    plt.savefig("ho.eps", format="eps")
    plt.show()
# parta()

#-----Part B-----#
def partb():
    x0 = 2.
    w = 1.

    t, x, v = sho(x0=x0, w=w)
    plt.plot(t, x, label="Approximated Position")
    plt.plot(t, x0 * np.cos(w*t), label="Analytic Position", ls='dashed')

    #Residuals of position
    # plt.plot(t, x - np.cos(w*t), label="Residuals")

    plt.xlabel("$t$ (units not given)")
    plt.ylabel("$x(t)$ (units not given)")
    plt.title(r"Simple Harmonic Oscillator for $x(0) = " + f"{x0}" + r"$")
    plt.legend()

    plt.savefig("ho2.eps", format="eps")
    plt.show()
# partb()

def a_b_redo():
    t, x, v = sho(x0=1.)
    plt.plot(t, x, label=f"$x(0) = {1.}$")

    t, x, v = sho(x0=2.)
    plt.plot(t, x, label=f"$x(0) = {2.}$")

    plt.legend()
    plt.xlabel("$t$ (Units not given)")
    plt.ylabel("$x(t)$ (Units not given)")
    plt.title("Harmonic Oscillator for Different Starting Positions")

    plt.savefig("bothab.eps", format='eps')
    plt.show()
a_b_redo()

#-----Part C-----#
def anharmonic(steps: int = 1001, start=0., end=50., w: float = 1., x0: float = 1., v0: float = 0.):
    #Decompose ODE: d^2x/dt^2 = -w^2 x
    #Resulting equations:
    #   dx/dt = v = fx(x, v, t)
    #   dv/dt = -w^2 x = fv(x, v, t)

    def f(x, v, t):
        return np.array([v, -np.square(w) * np.power(x, 3)])

    t = np.linspace(start, end, steps)
    dt = t[1]-t[0]

    x = np.zeros_like(t)
    v = np.copy(x)

    #Initial conditions
    x[0] = x0
    v[0] = v0

    i = 0
    while i < t.size-1:
        x[i+1], v[i+1] = rk4(f, t[i], dt, np.array([x[i], v[i]]))
        i += 1

    return t, x, v

def partc():
    fig, axs = plt.subplots(2, 1, sharex=True)
    x0 = [1., 2.]
    for i in range(len(x0)):
        ax = axs[i]
        t, x, v = anharmonic(x0=x0[i])

        ax.plot(t, x)

        # if i == 1: ax.set_xlabel("t (units not given)")
        # ax.set_ylabel("x(t) (units not given)")
        ax.set_title(r"$x(0) = " + f"{x0[i]}" + r"$")
    ax.set_xlabel("$t$ (units not given)")
    fig.supylabel("$x(t)$ (units not given)", fontsize='medium')
    plt.suptitle("Anharmonic Oscillator for Varying Start Positions", weight='bold')
    plt.tight_layout()
    plt.savefig("anhar.eps", format="eps")
    plt.show()
partc()

#-----Part D-----#
def partd():
    x0s = [1., 2., 3.]
    fig, axs = plt.subplots(1, 2, sharey=True)
    names = ["Harmonic", "Anharmonic"]
    oscs = [sho, anharmonic]
    for i in range(2):
        for x0 in x0s:
            t, x, v = oscs[i](x0=x0)
            axs[i].plot(x, v, label=r"$x(0) = " + f"{x0}" + r"$")
        axs[i].legend()
        axs[i].set_title(names[i])
        # axs[i].set_xlabel("$x(t)$ (units not given)")
        if i == 0: axs[i].set_ylabel("$v(t)$ (units not given)")
        i += 1

    fig.supxlabel("$x(t)$ (units not given)", fontsize='medium')
    fig.suptitle("Phase Space Plots for Oscillators", weight='bold')
    plt.savefig("phase.eps", format="eps")
    plt.show()
partd()

#-----Part E-----#
def vanderpol(steps: int = 1001, start=0., end=20., w=1., u = 1., x0 = 1., v0 = 0.):
    #Decompose ODE: d^2x/dt^2 - u(1-x^2)(dx/dt) + w^2 x = 0
    #Resulting equations:
    #   dx/dt = v = fx(x, v, t)
    #   dv/dt = u(1-x^2)v - w^2 x

    def f(x, v, t):
        return np.array([v, u * (1 - np.square(x)) * v - np.square(w) * x])

    t = np.linspace(start, end, steps)
    dt = t[1]-t[0]

    x = np.zeros_like(t)
    v = np.copy(x)

    #Initial conditions
    x[0] = x0
    v[0] = v0

    i = 0
    while i < t.size-1:
        x[i+1], v[i+1] = rk4(f, t[i], dt, np.array([x[i], v[i]]))
        i += 1

    return t, x, v

def parte():
    for u in [1., 2., 4.]:
        t, x, v = vanderpol(u=u)
        plt.plot(x, v, label=r"$\mu = " + str(u) + r"$")
    plt.legend()
    plt.title("Phase Space Plot for Van der Pol Oscillator")
    plt.xlabel("$x(t)$ (units not given)")
    plt.ylabel("$v(t)$ (units not given)")

    plt.savefig("vdp.eps", format='eps')
    plt.show()
parte()