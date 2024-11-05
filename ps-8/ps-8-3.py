"""
Exercise 7.4 of Newman (n.b. you may use numpy or scipy’s FFT routines), Pg. 317
"""
import numpy as np
import matplotlib.pyplot as plt

#-----Part A-----#
"""Load in data from infile and plot it. Return a numpy.ndarray populated with the data."""
with open('dow.txt', 'r') as f:
    lines = f.readlines()
    data = np.zeros(len(lines))
    for i in range(data.size):
        data[i] = float(lines[i].strip())

plt.plot(data)
plt.xlabel("Day (with Day 0 in late 2006)")
plt.ylabel("Dow Jones Closing Value (USD)")
plt.title("Value of the Dow Jones Industrial Average Over Time")
plt.savefig("dowplot.eps", format="eps")

plt.show()

#-----Part B-----#
"""Fourier transform data"""
fdata = np.fft.rfft(data)

#-----Part C-----#
"""Set all but the first [clip_fraction * 100]% of elements to 0"""
clip_fraction = 0.10
clip_index = int(clip_fraction * fdata.size)

clipped_fdata = fdata.copy()
clipped_fdata[clip_index:] = 0

#-----Part D-----#
"""Inverse Fourier transform clipped frequency data to get an approximation to the real data"""
approx_data = np.fft.irfft(clipped_fdata)

fig, axs = plt.subplots(2, 1, sharex=True)

ax = axs[0]
ax.plot(data, label="Original Data")
ax.plot(approx_data, label=f"Approximation")
ax.set_ylabel("Dow Jones Closing Value\n(USD)")
ax.set_title(f"Approximating the Dow Jones with the First {clip_fraction * 100}% of the Fourier Components")
ax.legend()

ax=axs[1]
ax.plot((approx_data-data)/data)
ax.set_xlabel("Day (with Day 0 in late 2006)")
ax.set_ylabel("Fractional Residuals\n(Approximation - Data)/(Data)")

plt.tight_layout()
plt.savefig("djapprox.eps", format="eps")
plt.show()

#-----Part E-----#
"""Repeat parts C and D, now using only the first 2% of Fourier components"""
clip_fraction = 0.02
clip_index = int(clip_fraction * fdata.size)

clipped_fdata = fdata.copy()
clipped_fdata[clip_index:] = 0

approx_data = np.fft.irfft(clipped_fdata)

fig, axs = plt.subplots(2, 1, sharex=True)

ax = axs[0]
ax.plot(data, label="Original Data")
ax.plot(approx_data, label=f"Approximation")
ax.set_ylabel("Dow Jones Closing Value\n(USD)")
ax.set_title(f"Approximating the Dow Jones with the First {clip_fraction * 100}% of the Fourier Components")
ax.legend()

ax=axs[1]
ax.plot((approx_data-data)/data)
ax.set_xlabel("Day (with Day 0 in late 2006)")
ax.set_ylabel("Fractional Residuals\n(Approximation - Data)/(Data)")

plt.tight_layout()
plt.savefig("djapprox2.eps", format="eps")
plt.show()