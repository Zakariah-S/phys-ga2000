import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as mani
import importlib
ps = importlib.import_module('ps-10-1')

#-----Compare these results with ps-10-1 results-----#
def resid_animation():
    x, psi1, dt = ps.load_data('welldata.npz')
    x, psi2, dt = ps.load_data('welldata2.npz')

    ps.animate_well(x, np.abs(psi2 - psi1), dt)
# resid_animation()

#-----Get snapshot of error at t=1e-16-----#
def error_snapshot():
    x, psi1, dt = ps.load_data('welldata.npz')
    x, psi2, dt = ps.load_data('welldata2.npz')

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.suptitle(r'Calculation of $\Psi(x, t = 1 \times 10^{-16}~s )$ using Different Integration Techniques')
    ax1.set_ylabel(r'$\psi(x, t = 1 \times 10^{-16}~s)$ ($m^{-1}$)')
    ax2.set_ylabel('Residuals ($m^{-1}$)')
    ax2.set_xlabel('$x$ ($m$)')

    ax1.plot(x, psi1[100], label='Crank-Nicholson')
    ax1.plot(x, psi2[100], label='Spectral')
    ax1.legend()
    ax2.plot(x, (psi2 - psi1)[100])
    plt.savefig('single_snapshot.eps', format='eps')
    plt.show()
# error_snapshot()

#-----Plot several snapshots of both wavefunctions-----#
def snapshots():
    x, psi1, dt = ps.load_data('welldata.npz')
    x, psi2, dt = ps.load_data('welldata2.npz')

    fig, axs = plt.subplots(4, 2, sharex = 'col', sharey=True, figsize=[6, 7.5])

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel('$x$ ($m$)')
    plt.ylabel('$\psi(x,~t)$ ($m^{-1}$)')
    plt.title('Integration of $\psi(x,~t)$ Using Different PDE Methods', y=1.08, fontstyle='italic')

    # fig.supxlabel('$x$ ($m$)')
    # fig.supylabel('$\psi(x,~t)$ ($m^{-1}$)')
    # fig.suptitle('Integration of $\psi(x,~t)$ Using Different PDE Methods')

    axs[0][0].set_title('Crank-Nicolson')
    axs[0][1].set_title('Spectral')

    frames = np.arange(0, 1501, 500)
    frames[0] = 100 #Changed because I need to plot the wavefunction at t = 1e-16 s

    i = 0
    for row in axs:
        row[0].plot(x / 1e-8, psi1[frames[i]], lw=0.7)
        row[1].plot(x, psi2[frames[i]], lw=0.7)
        timer_text = f'$t = $\t${np.format_float_scientific(dt * frames[i], precision=2, trim="k", pad_left=2, min_digits=2)}~s$'
        for j in [0, 1]:
            row[j].annotate(timer_text, 
                        xy=(0., 1.), xycoords='axes fraction', xytext=(1.2, -2.0), textcoords='offset fontsize', fontsize='x-small')
        i += 1

    plt.tight_layout()
    plt.savefig('comparison.eps', format='eps')
    plt.show()
snapshots()