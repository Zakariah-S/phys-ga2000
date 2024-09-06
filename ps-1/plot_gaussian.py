import numpy as pn
import matplotlib.pyplot as plt

def gaussian(X, stdev):
    return (1/(stdev * pn.sqrt(2 * pn.pi))) * pn.exp(-(X/(2 * stdev)) ** 2)

x = pn.arange(-10, 10, 0.1, dtype=pn.float64)
y = gaussian(x, 3)

plt.plot(x, y)
plt.xlim(-10, 10)
plt.ylim(0, pn.max(y) * 1.2)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Normalised Gaussian with $\sigma = 3$")
plt.savefig("gaussian.png")
plt.show()