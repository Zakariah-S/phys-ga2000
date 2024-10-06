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
    plt.title("Signal over Time")
    plt.savefig("signalplot.eps", format="eps")
    plt.show()
part_a()

#-----Part B-----#
