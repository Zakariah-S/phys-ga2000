import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

with fits.open("/content/drive/MyDrive/Grad/Computational Physics/specgrid.fits") as hdu_list:
    logwave = hdu_list["LOGWAVE"].data
    wave = np.power(10, logwave)
    flux = hdu_list["FLUX"].data

def part_a():
    for i in np.arange(5):
        # plt.plot(logwave, flux[i], label=f"Galaxy {i}")
        plt.plot(wave, flux[i], label=f"Galaxy {i}")
    plt.xlabel("Wavelength ($\AA$)")
    plt.ylabel("Flux ($10^{−17}~erg~s^{−1}~cm^{−2}~\AA^{-1}$)")
    plt.title("Flux vs Wavelength")
    plt.legend()
    plt.savefig("part_a.eps", format="eps")
    plt.show()
part_a()

def part_b():
    #Set up array to hold normalisation constants
    norms = np.zeros(flux.shape[0])

    #Set up weights for trapezoid method, factoring in "distance" between log(wavelength)-measures
    global trap_weights
    trap_weights = np.zeros(flux.shape[1], dtype=np.float64)
    trap_weights[1:-1] = wave[2:] - wave[1:-1]
    trap_weights[0] = 0.5 * (wave[1] - wave[0])
    trap_weights[-1] = 0.5 * (wave[-1] - wave[-2])
    # print(trap_weights)

    #Calculate multiplicative normalisation constants
    for i in np.arange(flux.shape[0]):
        norms[i] = 1/((flux[i] * trap_weights).sum())
    
    #Plot new spectra
    for i in np.arange(5):
        # plt.plot(logwave, norms[i] * flux[i], label=f"Galaxy {i}")
        plt.plot(wave, norms[i] * flux[i], label=f"Galaxy {i}")
        print((norms[i] * trap_weights * flux[i]).sum())
    plt.xlabel("Wavelength ($\AA$)")
    plt.ylabel("Normalised Flux ($\AA^{-1}$)")
    plt.title("Normalised Flux vs Wavelength")
    plt.legend()
    plt.tight_layout()
    plt.savefig("part_b.eps", format="eps")
    plt.show()

    return norms
norms = part_b()

def part_c():
    #Set up arrays to hold the means of each spectrum, and the residual spectra f - f_avg
    mean_spectra = np.zeros(flux.shape[0])
    resid_spectra = np.zeros_like(flux)

    #Actually calculate f_avg and f - f_avg
    for i in np.arange(flux.shape[0]):
        mean_spectra[i] = np.mean(flux[i])
        resid_spectra[i] = norms[i] * (flux[i] - mean_spectra[i])
    
    #Plot new spectra
    for i in np.arange(5):
        # plt.plot(logwave, resid_spectra[i], label=f"Galaxy {i}")
        plt.plot(wave, resid_spectra[i], label=f"Galaxy {i}")
        print(np.mean(resid_spectra[i]))
    plt.xlabel("Wavelength ($\AA$)")
    plt.ylabel("Normalised Flux ($\AA^{-1}$)")
    plt.title("Normalised, Mean-Centered Flux vs Wavelength")
    plt.legend()
    plt.tight_layout()
    plt.savefig("part_c.eps", format="eps")
    plt.show()

    return mean_spectra, resid_spectra

mean_spectra, resid_spectra = part_c()

#Get eigenvectors using eig
#Runtime was 1:28 for all galaxies
#59s for 500 galaxies
def part_d():
    #Get covariance matrix
    cov = (resid_spectra.T).dot(resid_spectra)
    # print(cov.shape)

    #Get eigenvectors
    print("Getting eigenvectors")
    evals, evecs = np.linalg.eig(cov)
    # print(evals.shape)
    # print(evecs.shape)

    #Plot first five eigenvectors
    for i in np.arange(5):
        # plt.plot(logwave, evecs[:, i], label=f"Eigenvector {i}")
        plt.plot(wave, evecs[:, i], label=f"Eigenvector {i}")
    plt.xlabel("Wavelength ($\AA$)")
    plt.ylabel("Eigenvector ($\AA^{-2}$)")
    plt.title("Eigenvectors of the Covariance Matrix")
    plt.legend()
    plt.savefig("part_d.eps", format="eps")
    plt.show()

    return cov, evals, evecs
cov, evals, evecs = part_d()

#Sort eigenvectors in order of decreasing eigenvalue (should have done this before to be honest)
isort = np.argsort(evals)[::-1]
evalsort = evals[isort]
evecsort = evecs[:, isort]

#Get eigenvectors using SVD
#Ran in 1:43 for all galaxies
#5s for 500 galaxies
def part_e():
    u, w, vT = np.linalg.svd(resid_spectra, full_matrices=False)

    #Plot first five SVD eigenvectors
    for i in np.arange(5):
        plt.plot(wave, v[:, i], label=f"Eigenvector {i}")
    plt.xlabel("Wavelength ($\AA$)")
    plt.ylabel("Eigenvector ($\AA^{-2}$)")
    plt.title("Eigenvectors using SVD")
    plt.legend()
    plt.savefig("part_e.eps", format="eps")
    plt.show()

    return w, vT.T
w, v = part_e()

#Compare condition numbers of the two approaches
def part_f():
    print(w[0], w[-1])
    print(np.sqrt(evalsort[0]), np.sqrt(evalsort[-1]))
    print(f"For svd, condition number = {w[0]/w[-1]}")
    print(f"For eig, condition number = {np.sqrt(evalsort[0]/evalsort[-1])}")
part_f()

def part_g(): #I originally had a version of this using evecsort, but removed it to keep code a bit cleaner
    # eigenbasis_spectra = v.T.dot(resid_spectra.T).T
    eigenbasis_spectra = resid_spectra.dot(v)
    # print(eigenbasis_spectra.shape)

    #Approximate spectrum using 5 of these eigenspectra
    truncated_eigen = eigenbasis_spectra.copy()
    truncated_eigen[:,6:] = 0
    print(truncated_eigen.shape)

    for i in np.arange(5):
        plt.plot(wave, ((v.dot(truncated_eigen[i].T))/norms[i] + mean_spectra[i] - flux[i])/flux[i], label=f"Galaxy {i}")
        # plt.scatter(wave, ((v.dot(truncated_eigen[i].T))/norms[i] + mean_spectra[i] - flux[i])/flux[i], label=f"Galaxy {i}")
    plt.legend()
    plt.title("Residual Spectra for a 5-Eigenspectra Approximation")
    plt.xlabel("Wavelength ($\AA$)")
    plt.ylabel(r"$\frac{Reproduced~Spectrum~-~Original~Spectrum}{Original~Spectrum}$")
    plt.ylim(-10, 10)
    plt.tight_layout()
    plt.savefig("part_g.eps", format="eps")
    plt.show()

    return eigenbasis_spectra
eigenbasis_spectra = part_g()

#Reproduce the second galaxy spectrum using all eigenspectra
# plt.plot(wave, (v.dot(eigenbasis_spectra[1].T))/norms[1] + mean_spectra[1], label="Reproduced")
# plt.plot(wave, flux[1], label="Original", lw=0.5)
# plt.legend()
# plt.title("Reproduced Spectrum")
# plt.xlabel("Wavelength ($\AA$)")
# plt.ylabel("Flux ($10^{−17}~erg~s^{−1}~cm^{−2}~\AA^{-1}$)")
# plt.show()

#Get residuals of reproduced spectrum
# plt.plot(wave, ((v.dot(eigenbasis_spectra[1].T))/norms[1] + mean_spectra[1] - flux[1])/flux[1])
# plt.title("Full Reproduced Spectrum Residuals")
# plt.xlabel("Wavelength ($\AA$)")
# plt.ylabel("Reproduced Spectrum - Original Spectrum")
# plt.savefig("resids.eps", format='eps')
# plt.show()

#plot c_0, c_1, c_2 as functions of each other
def part_h():
    fig, axs = plt.subplots(1, 2)

    #c_0 vs c_1
    ax = axs[0]
    ax.scatter(eigenbasis_spectra[:,0], eigenbasis_spectra[:,1], s=0.5)
    ax.set_xlabel("$c_0$")
    ax.set_ylabel("$c_1$")
    ax.set_title("$c_1$ vs $c_0$")

    #c_0 vs c_2
    ax = axs[1]
    ax.scatter(eigenbasis_spectra[:,0], eigenbasis_spectra[:,2], s=0.5)
    ax.set_xlabel("$c_0$")
    ax.set_ylabel("$c_2$")
    ax.set_title("$c_2$ vs $c_0$")

    plt.tight_layout()
    plt.savefig("part_h.eps", format="eps")
    plt.show()
part_h()

#Plot RMS-residual curve for the first five galaxy spectra
def part_i(Ns, spectra):
    #Ns should be a list/array of integers, the integers being the differing numbers of eigenspectra to use for approximating
    #Spectra should be a list/array of integers, with each integer being the index of a galaxy spectrum you want to approximate
    for i in spectra:
        rms_resids = np.zeros_like(Ns, dtype=np.float64)
        j = 0
        while j < Ns.size:
            #For each N given, approximate the spectrum using N eigenspectra and find the RMS of residuals
            truncated_eigen = eigenbasis_spectra.copy()
            truncated_eigen[:,int(Ns[j]+1):] = 0
            resid = ((v.dot(truncated_eigen[i].T))/norms[i] + mean_spectra[i] - flux[i])/flux[i]
            rms_resids[j] = np.sqrt(np.mean(np.square(resid)))
            j += 1
        plt.plot(Ns, rms_resids, label=f"Galaxy {i}")
        print(rms_resids[-1])
    plt.legend()
    plt.title("RMS Residual vs. Number of Eigenspectra Used")
    plt.xlabel("Number of Eigenspectra")
    plt.ylabel("Root-Mean-Squared Residual")
    plt.savefig("part_i.eps", format="eps")
    plt.show()
part_i(np.arange(21, dtype=np.uint64), np.arange(5))