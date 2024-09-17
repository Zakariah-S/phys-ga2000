import numpy as np
import matplotlib.pyplot as plt

def mandelbrot(N: int, iterations: int = 100):
    #Plot an image of the Mandelbrot set at a specified resolution

    #Set up a N x N grid spanning the (x, y) coords from (-2, -2) to (2, 2)
    x = y = np.linspace(-2, 2, N, dtype=np.float64)
    x, y = np.meshgrid(x, y)

    #Define the complex number c at each point of the coordinate system
    c = x + y * 1j
    
    #Set up z_0
    z = np.zeros_like(c)

    #Set up a marker that initially includes every point in the grid, but switches to excluding once |z(c)| passes 2
    marker = np.ones_like(x) #points marked 1 are included in the set
    for i in np.arange(iterations):
        z = z ** 2 + c
        marker[np.absolute(z) > 2] = 0 #points marked 0 are excluded from the set
    
    plt.imshow(marker, origin='lower', extent=(-2, 2, -2, 2))
    plt.show()

# mandelbrot(1000, 20)

def colored_mandelbrot(N: int, iterations: int = 100) -> np.ndarray:
    ''' -
    Plot an image of the Mandelbrot set at a specified resolution, 
    colouring based on how many iterations pass before the point drops out of the set.
    Code is the same as the function above, except the marker now takes values other than 1 or 0
    '''
    x = y = np.linspace(-2, 2, N, dtype=np.float64)
    x, y = np.meshgrid(x, y)
    c = x + y * 1j
    
    z = np.zeros_like(c)
    marker = np.full(c.shape, -1, dtype=np.int32)
    for i in np.arange(iterations):
        z = z ** 2 + c
        marker[np.logical_and(np.absolute(z) > 2, marker == -1)] = i
    marker[marker == -1] = iterations
    
    plt.imshow(marker, origin='lower', extent=(-2, 2, -2, 2))
    plt.colorbar()
    plt.show()

colored_mandelbrot(1000, 20)