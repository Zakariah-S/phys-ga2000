"""
This problem demonstrates an application of linear algebra to signal analysis. 
Download a “signal” as a function of time from this file. 
Assume that all the measurements have the same uncertainty, with a standard deviation of 2.0 in the signal units.

(a) Plot the data.

(b) Use the SVD technique to find the best third-order polynomial fit in time to the signal. 
    Pay attention to the scaling of the independent variable (time).

(c) Calculate the residuals of the data with respect to your model. 
    Argue that this is not a good explanation of the data given what you know about the measurement uncertainties.

(d) Try a much higher order polynomial. 
    Is there any reasonable polynomial you can fit that is a good explanation of the data? 
    Define “reasonable polynomial” as whether the design matrix has a viable condition number.

(e) Try fitting a set of sin and cos functions plus a zero-point offset. 
    As a Fourier series does, use a harmonic sequence with increasing frequency, 
    starting with a period equal to half of the time span covered. Does this model do a “good job” explaining the data? 
    Are you able to determine a typical periodicity in the data? You may have noticed a periodicity from the first plot.

This last fit is a version of the Lomb-Scargle technique for detecting variability in unevenly sampled data, 
which is designed to be a very close approximation to fitting with a set of Fourier modes. 
Implementing the method the way we do here (explicitly decomposing the design matrix) is not the usual method, 
since it is slower than other techniques, but it is the simplest to implement.
"""
import numpy as np
import matplotlib.pyplot as plt

#Unpack data
with open("signal.dat", "r") as f:
    #Get relevant lines of file
    lines = f.readlines()[1:]

    #Set up arrays to receive data
    time = np.zeros(len(lines))
    signal = np.zeros_like(time)

    #Add data to arrays
    for i in range(len(lines)):
        pair = lines[i].split("|")[1:-1]
        time[i] = pair[0].strip()
        signal[i] = pair[1].strip()

#Sort data in order of increasing time
sort_index = np.argsort(time)
time = time[sort_index]
signal = signal[sort_index]

#-----Part A-----#

#Plot data
def part_a():
    plt.plot(time, signal)
    plt.xlabel("Time")
    plt.ylabel("Signal")
    plt.title("Signal vs. Time")
    plt.savefig("signalplot.eps", format="eps")
    plt.show()

    # #Get max time
    # max = np.argmax(time)
    # print(time[max], signal[max])
# part_a()

#-----Part B-----#
""" 
Use the SVD technique to find the best third-order polynomial fit in time to the signal. 
Pay attention to the scaling of the independent variable (time).
"""
def part_b():

    #Set up matrix A for third-degree polynomial regression

    #Rescale t to ~order unity
    t_rescale = (time - np.mean(time)) / np.std(time)

    A = np.zeros((len(t_rescale), 4))
    A[:, 0] = 1.
    A[:, 1] = t_rescale
    A[:, 2] = t_rescale**2
    A[:, 3] = t_rescale**3

    print(A)
    #Decompose A using the SVD method
    (u, w, vt) = np.linalg.svd(A, full_matrices=False)
    # print(u)
    # print(w)
    # print(vt)

    #Replace all ~0 elements on the diagonal of w with infinity. This ensures that the corresponding elements of w^(-1) will be 0.
    print(w)
    # >> [68.00312122 49.58519311 17.73756426 11.48710147]
    w[w < 1e-15 * np.max(w)] = np.inf

    #Get the "inverse" of A and derive the model's predicted signal = A * (A^(-1)) * [real signal]
    Ainv = vt.transpose().dot(np.diag(1./w)).dot(u.transpose())
    signal_predict = A.dot(Ainv).dot(signal)

    #Plot the signal and the predicted signal
    plt.plot(time, signal, label="Measured Signal")
    plt.plot(time, signal_predict, label="Model")
    plt.xlabel("Time")
    plt.ylabel("Signal")
    plt.title("Third-degree Polynomial Model")

    plt.legend()
    plt.savefig("thirddegmodel.eps", format="eps")
    plt.show()
# part_b()

#-----Part C-----#
""" 
Calculate the residuals of the data with respect to your model. 
Argue that this is not a good explanation of the data given what you know about the measurement uncertainties.
"""
def part_c():

    #Repeat the derivation of the model from Part B, and plot residuals

    #Rescale t to ~order unity
    t_rescale = (time - np.mean(time)) / np.std(time)
    A = np.zeros((len(t_rescale), 4))
    A[:, 0] = 1.
    A[:, 1] = t_rescale
    A[:, 2] = t_rescale**2
    A[:, 3] = t_rescale**3

    (u, w, vt) = np.linalg.svd(A, full_matrices=False)
    print(w)
    # >> [68.00312122 49.58519311 17.73756426 11.48710147]
    w[w < 1e-15 * np.max(w)] = np.inf

    Ainv = vt.transpose().dot(np.diag(1./w)).dot(u.transpose())
    signal_predict = A.dot(Ainv).dot(signal)

    plt.hlines(y=[-2., 2.], xmin=0., xmax=np.max(time), color='gray', ls='-', label="signal errors")
    plt.plot(time, (signal_predict - signal))
    # plt.ylabel(r"Residuals $\left( \frac{predicted~signal - signal}{signal} \right)$")
    plt.ylabel(r"Residuals $(predicted~signal - true~signal)$")
    plt.xlabel("Time")
    plt.title("Residuals of Third-degree Polynomial Model")
    plt.legend()

    plt.savefig("thirdresids.eps", format="eps")
    plt.show()
# part_c()
"""
This model is really bad! The residual regularly goes over 2.0-- the error we've assigned to the signal.
"""

#-----Part D-----#
"""
Try a much higher order polynomial.
Is there any reasonable polynomial you can fit that is a good explanation of the data? 
Define “reasonable polynomial” as whether the design matrix has a viable condition number.
"""
def fit_poly(degree):
    """
    Fit a polynomial of any given degree to the signal.
    degree: int
        The order of the polynomial to be fit.
    Returns:
        w: The singular value matrix, with all values less than 1e-15 * the max value replaced with infinity
    """
    t_rescale = (time - np.mean(time)) / np.std(time)
    A = np.zeros((len(t_rescale), degree + 1))
    for i in np.arange(0, A.shape[1]):
        A[:, i] = np.power(t_rescale, i)

    (u, w, vt) = np.linalg.svd(A, full_matrices=False)
    w_fit = w.copy()
    w_fit[w < 1e-15 * np.max(w)] = np.inf

    Ainv = vt.transpose().dot(np.diag(1./w_fit)).dot(u.transpose())
    signal_predict = A.dot(Ainv).dot(signal)

    return w, signal_predict

def part_d(span, title = "", save=False, savename="test.eps", legend=True):
    fig, axs = plt.subplots(2, 1, sharex=True)

    #for when i was making just one the signal vs model plot
    # fig, ax = plt.subplots()
    # axs = [ax]

    axs[0].plot(time, signal, label="Data")
    axs[1].hlines(y=[-2., 2.], xmin=0., xmax=np.max(time), color='gray', ls='-', label="signal errors")
    for i in span:
        w, signal_predict = fit_poly(i)
        print(np.max(w) / np.min(w) / 1e15)
        axs[0].plot(time, signal_predict, label=f"Degree-{i} fit")
        axs[1].plot(time, signal_predict - signal, label=f"Degree-{i} fit")

    axs[0].set_ylabel("Signal")
    axs[1].set_ylabel(r"Residuals" + "\n" + r"$(predicted~signal - true~signal)$")
    axs[1].set_xlabel("Time")
    # axs[0].set_xlabel("Time")
    axs[0].set_title(title)
    if legend==True:
        axs[0].legend()
        axs[1].legend()
    if save: plt.savefig(savename, format="eps")
    plt.show()

# part_d(span=np.arange(5, 31, 5, dtype=np.uint32))

# part_d(span=[32], title="32-degree Polynomial fit", save=True, savename="bestpoly.eps", legend=False)
# >> condition number = 0.07e15

# part_d(span=[34], title = "34-degree Polynomial fit", save=True, savename="bestpoly.eps", legend=False)
# >> condition number = 0.60e15

# part_d(span=[35])
# >> condition number = 1.8e15

# part_d(span=[15, 20, 25], title="Best-fit polynomials", save=True, savename="polyfits.eps")

#-----Part E-----#
"""
Try fitting a set of sin and cos functions plus a zero-point offset. 
As a Fourier series does, use a harmonic sequence with increasing frequency, 
starting with a period equal to half of the time span covered. Does this model do a “good job” explaining the data? 
Are you able to determine a typical periodicity in the data? You may have noticed a periodicity from the first plot.
"""

def fit_trigs(iterations):
    """
    Fit a series of cosine and sines of linearly increasing frequency to the data.
    The fitted model corresponds to 1 + sin(pt) + cos(pt) + .... + sin([iterations]*pt) + cos([iterations]*pt)
    iterations: int
        The number of sines (and the number of cosines) fit to the data
    Returns:
        w: numpy.ndarray
            The singular value matrix, with all values less than 1e-15 * the max value replaced with infinity
        signal_predict: numpy.ndarray
            The predicted signal.
    """
    A = np.zeros((len(time), 2 * iterations + 1))
    A[:, 0] = np.ones_like(time)

    #Start with frequency corresponding to a period of ~half the max time
    freq = 2 * np.pi / (1e9 * 0.5)
    added_freq = freq
    for i in np.arange(1, iterations + 1):
        A[:, i] = np.sin(freq * time)
        A[:, i + iterations] = np.cos(freq * time)
        #up frequency for next iteration of sines and cosines
        freq += added_freq

    (u, w, vt) = np.linalg.svd(A, full_matrices=False)
    w_fit = w.copy()
    w_fit[w < 1e-15 * np.max(w)] = np.inf
    Ainv = vt.transpose().dot(np.diag(1./w_fit)).dot(u.transpose())
    x = Ainv.dot(signal)
    signal_predict = A.dot(Ainv).dot(signal)

    return w, signal_predict, x

def part_e():

    # Find the highest possible number of sines and cosines with condition number < 10^(14): 448
    # num = 420
    # condition_num = 0.
    # while condition_num < 0.1:
    #     print(num)
    #     num += 1
    #     w, signal_predict, x = fit_trigs(num)
    #     condition_num = np.max(w) / np.min(w) / 1e15
    # num -= 1

    #Now that we know the number after running the while loop once, I'll just insert it here and proceed as normal lol
    num = 448
    num = 50
    print(num)
    w, signal_predict, x = fit_trigs(num)
    condition_num = np.max(w) / np.min(w) / 1e15
    print(condition_num)

    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(time, signal, label="Data")
    axs[0].plot(time, signal_predict, label="Model")
    axs[0].set_ylabel("Signal")
    axs[0].set_title("Sine and Cosine Series Fit to Signal Data")
    axs[0].legend()

    axs[1].plot(time, signal_predict - signal)
    axs[1].hlines(y=[-2., 2.], xmin=0., xmax=np.max(time), color='gray', ls='-', label="signal errors")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel(r"Residuals" + "\n" + r"$(predicted~signal - true~signal)$")

    plt.savefig("fourierfit.eps", format="eps")
    plt.show()

    #Plot coefficients
    plt.clf()

    sine_coeffs = x[1:num+1]
    cos_coeffs = x[num+1:]

    freq = 2 * np.pi / (1e9 * 0.5)
    freqs = freq * np.arange(1., num + 1.)
    period = 2 * np.pi/freqs

    # plt.plot(period, np.abs(sine_coeffs), label="Sine coefficients")
    # plt.plot(period, np.abs(cos_coeffs), label="Cosine coefficients")
    plt.plot(period, np.abs(sine_coeffs) + np.abs(cos_coeffs))
    plt.legend()
    plt.xlabel("Period")
    plt.ylabel("Coefficient Sum")
    plt.title("Coefficient Sums for each Period")
    plt.savefig("coeeffs.eps", format="eps")
    plt.show()
part_e()