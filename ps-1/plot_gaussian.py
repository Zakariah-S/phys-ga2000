import numpy as pn
import matplotlib.pyplot as plt

'''
Set our standard deviation (and the Gaussian normalisation accordingly).
Set up equally spaced X-values and the Gaussian function by which we get the corresponding Y-values.
Plot the Gaussian, set appropriate x-limits, show x=0 axis, and format plot and axes titles.
Finally, save the figure and then show it.
'''
sigma = 3
norm = (1/(sigma * pn.sqrt(2 * pn.pi)))

def gaussian(x, N, stdev):
    return N * pn.exp(-(x/(2 * stdev)) ** 2)

X = pn.linspace(-10, 10, 100, dtype=pn.float64)
Y = gaussian(X, norm, sigma)

plt.plot(X, Y)
plt.xlim(-10, 10)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Normalised Gaussian with $\sigma = 3$")
plt.axvline(x=0, color='grey')
plt.savefig("gaussian.png")
plt.show()