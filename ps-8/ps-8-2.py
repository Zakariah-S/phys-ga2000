"""
Exercise 7.3 of Newman (n.b. you may use numpy or scipy’s FFT routines), Pg. 317
"""
import numpy as np
import matplotlib.pyplot as plt

def get_freqs(infile: str, show1: bool = True, format='eps') -> None: #Function for Part A
    """
    Take a .txt file with time-series data and apply a time-based FFT, plotting the resulting frequencies
    infile: string
        Should point to a file with a float on each line.
    show1: bool
        If True, the plot of the signal is shown (the plot is saved regardless).
    format: string
        File extension to save plots with.
    """
    savename1 = infile[:-4] + 'data'
    savename2 = infile[:-4] + 'freq'

    with open(infile, 'r') as f:
        lines = f.readlines()
        signal = np.zeros(len(lines))
        for i in range(signal.size):
            signal[i] = float(lines[i].strip())

    #Calculate signal times, given a sample rate of 44100 Hz
    times = np.arange(signal.size) / 44100.

    #Plot + save signal data
    plt.xlabel("Time (s)")
    plt.ylabel("Signal (units unknown)")
    plt.title(f"{infile[:-4].capitalize()} Signal Over Time")
    plt.plot(times, signal)
    plt.tight_layout()
    plt.savefig(savename1 + '.' + format, format=format)
    if show1: plt.show()

    #Fourier transform data
    fdata = np.fft.rfft(signal)

    #Calculate frequencies, given the sample rate of 44100 Hz.
    """ 
    FFT algorithm returns the frequency as [number of data points within a period]/[total data points]
    To convert this to Hz, we use the fact that the time range of the total data set is T = 100000 / (44100 samples s^(-1)).
    Then k=1 is the case where f = 1/T = 44100 / 100000 = 0.441, 
    and the rest of the ks are proportional to the corresponding fs in the same way.
    """ 
    freqs = np.arange(fdata.size) * 0.441

    plt.clf()
    plt.plot(freqs[:10001], np.absolute(fdata[:10001]))
    plt.title(f"Frequencies of {infile[:-4].capitalize()}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude of frequency component")
    plt.savefig(savename2  + '.' + format, format=format)
    plt.show()

    return freqs, fdata

if __name__ == "__main__":
    freqs, fdata = get_freqs('piano.txt', format='eps', show1=False)
    print(f"Piano: {freqs[np.argmax(fdata)]} Hz")
    # >> The piano is playing 525.231 Hz = C5, which is an octave higher than middle C (which is C4).

    freqs, fdata = get_freqs('trumpet.txt', format='eps', show1=False)
    print(f"Trumpet: {freqs[np.argmax(fdata)]} Hz")
    # >> The trumpet is playing 1043.847 = C6, two octaves higher than middle C.