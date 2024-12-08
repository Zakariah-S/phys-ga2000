"""
Exercise 9.9 of Newman. (Pg. 442)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as mani
from dcst import dst, idst

#Constants
hbar = 1.054e-34 #Reduced Planck constant, in J * Hz
m = 9.109e-31 #Mass of electron, in kg
L = 1e-8 #Length of box, in m

#Wavefunction at t = 0
def psi_init(x):
    return np.exp(-(1/2) * np.square((x - L/2) / 1e-10)) * np.exp(1j * 5e10 * x)

def test_calc(N: float, initial_state_func):
    """
    As per Part B, plot the simulated wavefunction at time 1e-16s.
    """
    x = np.linspace(0, L, N)
    init_state = initial_state_func(x) #Initial value of psi at each point

    #Inverse sine transform of real part of initial state
    alpha = dst(np.real(init_state))
    
    #Inverse sine transform of imaginary part of initial state
    eta = dst(np.imag(init_state))

    t = 1e-16 #Arbitrary test time

    k = np.arange(N)

    psi_t_sine_transform = alpha * np.cos(hbar * np.square(np.pi * k) / (2 * m * np.square(L)) * t) + eta * np.sin(hbar * np.square(np.pi * k) / (2 * m * np.square(L)) * t)
    psi_t = idst(psi_t_sine_transform)

    fig, ax = plt.subplots()
    ax.set_title('Particle in a 1-D Infinite Square Well')
    ax.set_xlabel('$x$ ($m$)')
    ax.set_ylabel(r'$\psi(x, t)$ ($m^{-1}$)')
    ax.set_xlim(x[0], x[-1])

    ax.plot(x, psi_t)
    ax.annotate('$t = $\t$1.00e-16~s$', xy=(0., 1.), xycoords='axes fraction', xytext=(1.2, -2.0), textcoords='offset fontsize')

    plt.savefig('comparisonplot2.png', format='png')
    plt.savefig('comparisonplot2.eps', format='eps')
    plt.show()

def simulate_well(N: float, initial_state_func, steps: int = 1000, dt: float = 1e-18):
    x = np.linspace(0, L, N)
    init_state = initial_state_func(x) #Initial value of psi at each point
    init_state[0] = 0.
    init_state[-1] = 0.

    #Inverse sine transform of real part of initial state
    alpha = dst(np.real(init_state[:-1]))
    
    #Inverse sine transform of imaginary part of initial state
    eta = dst(np.imag(init_state[:-1]))

    k = np.arange(N-1)
    psi = np.zeros((steps + 1, init_state.size), dtype=np.float64)
    psi[0] = np.real(init_state)

    for i in np.arange(1, steps + 1):
        print(f'{i} / {steps}')
        t = dt * i
        psi_t_sine_transform = alpha * np.cos(hbar * np.square(np.pi * k) * t / (2 * m * np.square(L))) + eta * np.sin(hbar * np.square(np.pi * k) * t / (2 * m * np.square(L)))
        psi[i][:-1] = idst(psi_t_sine_transform)

    return x, psi, dt

def save_data(X, Psi, dT, savename='welldata2'):
    # np.savetxt(savename + '.txt', psi, header=f'dt={dt},X_low={x[0]},X_high={x[-1]},X_count={x.size}')
    dT = np.array([dT])
    np.savez_compressed(savename, dt=dT, x=X, psi=Psi)

def animate_well(x, psi, dt):
    fig, ax = plt.subplots()
    ax.set_title('Particle in a 1-D Infinite Square Well')
    ax.set_xlabel('$x$ ($m$)')
    ax.set_ylabel('$\psi(x, t)$ ($m^{-1}$)')
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(bottom=1.2 * np.min(psi), top=1.2 * np.max(psi))

    psi_plot, = ax.plot(x, psi[0])
    time = ax.annotate('$t = $\t$0~s$', xy=(0., 1.), xycoords='axes fraction', xytext=(1.2, -2.0), textcoords='offset fontsize')

    def update(frame):
        psi_plot.set_data(x, psi[frame+1])
        time.set(text=f'$t = $\t${np.format_float_scientific(dt * frame, precision=2, trim="k", pad_left=2, min_digits=2)}~s$')
        return psi_plot

    anim = mani.FuncAnimation(fig, update, psi.shape[0] - 1, repeat=True, interval=1)
    plt.show()

def load_data(infile, stop_index=None):
    loaded = np.load(infile, mmap_mode='r')
    dt = loaded['dt'][0]
    x = loaded['x']
    
    if stop_index: psi = loaded['psi'][:stop_index]
    else: psi = loaded['psi']

    return x, psi, dt

# test_calc(1000+1, psi_init)

if __name__ == '__main__':
    #-----To generate data set and run animation
    # x, psi, dt = simulate_well(1000+1, psi_init, steps=1500)
    # save_data(x, psi, dt)
    # animate_well(x, psi, dt)

    #-----To run animation from saved data
    x, psi, dt = load_data('welldata2.npz')
    animate_well(x, psi, dt)

    #-----Time simulation
    # import timeit
    # time = timeit.timeit('simulate_well(N=1000+1, initial_state_func=lambda x: np.exp(-(1/2) * np.square((x - 1e-8/2) / 1e-10)) * np.exp(1j * 5e10 * x), steps=1500)', 
    #                      setup='import numpy as np\nfrom banded import banded\nfrom __main__ import simulate_well', number=1)
    # print(time)