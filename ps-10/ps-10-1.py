"""
Exercise 9.8 of Newman. (Pg. 439)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as mani
from banded import banded

#Constants
hbar = 1.054e-34 #Reduced Planck constant, in J * Hz
m = 9.109e-31 #Mass of electron, in kg
L = 1e-8 #Length of box, in m

#Wavefunction at t = 0
def psi_init(x):
    return np.exp(-(1/2) * np.square((x - L/2) / 1e-10)) * np.exp(1j * 5e10 * x)

def simulate_well(N: float, initial_state_func, steps: int = 1000, dt: float = 1e-18):
    x = np.linspace(0, L, N)
    init_state = initial_state_func(x) #Initial value of psi at each point

    a = x[1] - x[0] #Distance between sampling points

    #Get tridiagonal matrix A in the form we need it in for the banded function
    a1 = 1. + dt * 1.j*hbar / (2. * m * np.square(a))
    a2 = -dt * 1.j*hbar / (4. * m * np.square(a))

    A = np.zeros((3, N), dtype=np.complex128)
    A[0][1:] = a2 #upper diagonal
    A[1][:] = a1 #main diagonal
    A[2][:-1] = a2 #lower diagonal

    #Set up for calculations involving the symmetric tridiagonal matrix B
    b1 = np.conj(a1) #value along main diagonal
    b2 = np.conj(a2) #value along upper and lower diagonals

    #Initialise an array to hold states for each timestep
    psi = np.zeros((steps + 1, init_state.size), dtype=np.complex128)
    psi[0] = init_state

    for i in np.arange(psi.shape[0] - 1):
        print(f'{i+1} / {steps}')
        #Calculate v = B * psi(t)
        v = np.zeros_like(psi[i])
        v[0] = b1 * psi[i, 0] + b2 * psi[i, 1]
        v[1:-1] = b1 * psi[i, 1:-1] + b2 * (psi[i, 0:-2] + psi[i, 2:])
        v[-1] = b1 * psi[i, -1] + b2 * psi[i, -2]

        #Solve matrix equation to find psi(t + dt)
        psi[i+1] = banded(A, v, 1, 1)

    return x, psi, dt

def save_data(X, Psi, dT, savename='welldata'):
    # np.savetxt(savename + '.txt', psi, header=f'dt={dt},X_low={x[0]},X_high={x[-1]},X_count={x.size}')
    dT = np.array([dT])
    np.savez_compressed(savename, dt=dT, x=X, psi=np.real(Psi))

def animate_well(x, psi, dt):
    fig, ax = plt.subplots()
    ax.set_title('Particle in a 1-D Infinite Square Well')
    ax.set_xlabel('$x$ ($m$)')
    ax.set_ylabel('$\psi(x, t)$ ($m^{-1}$)')
    ax.set_xlim(x[0], x[-1])

    psi_plot, = ax.plot(x, psi[0])
    time = ax.annotate('$t = $\t$0~s$', xy=(0., 1.), xycoords='axes fraction', xytext=(1.2, -2.0), textcoords='offset fontsize')

    def update(frame):
        psi_plot.set_data(x, psi[frame+1])
        time.set(text=f'$t = $\t${np.format_float_scientific(dt * frame, precision=2, trim="k", pad_left=2, min_digits=2)}~s$')
        #Small snippet that saves plot at t = 1e-16
        # if np.abs(dt * frame - 1e-16) < 1e-18 and dt * frame > 1e-16 - 1e-18:
        #     plt.savefig('comparisonplot1.eps', format='eps')
        #     plt.savefig('comparisonplot1.png', format='png')
        #     print('plot saved')
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

x, psi, dt = simulate_well(N=1000+1, initial_state_func=psi_init, steps=1500)
save_data(x, psi, dt)

# x, psi, dt = load_data('welldata.npz')
# animate_well(x, psi, dt)