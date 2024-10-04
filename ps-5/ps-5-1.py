"""
Exercise 5.15 in Newman. Additionally, use jax to perform the autodiff version of the deriva-
tive to verify it works as advertised.
(Pg. 194)
"""
import numpy as np
import matplotlib.pyplot as plt

#Function we want to differentiate
def f(x):
    return 1 + 0.5 * np.tanh(2. * x)

#Actual derivative of the function
def f_prime(x):
    return 1/np.square(np.cosh(2.*x))

#-----Part A-----#

def part_a(): #Plot function, derivative, and approximated derivative

    #Central difference derivative
    def get_derivatives(func, x, h=1.e-5):
        return ((func(x + 0.5 * h) - func(x - 0.5 * h)) / h)

    #Set up plots, x values to evaluate function + derivatives
    fig, axs = plt.subplots(2, 1, sharex=True)
    x = np.linspace(-2, 2, 1000, dtype=np.float64)

    #Plot function, derivative, approximated derivative
    axs[0].plot(x, f(x), label='Function')
    axs[0].plot(x, f_prime(x), label="Derivative")
    axs[0].plot(x, get_derivatives(f, x), label='Approximate derivative', ls="dotted")
    axs[0].legend()
    axs[0].set_title(r"Derivative of $1 + \frac{1}{2}tanh(2x)$")
    axs[0].set_ylabel("Evaluation at x")

    #Plot residuals
    axs[1].plot(x, get_derivatives(f, x) - f_prime(x))
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("Residuals")

    plt.show()

part_a()

#-----Part B-----#

def part_b():
    import jax
    import jax.numpy as jnp

    #Get function using Jax's numpy
    def f_jax(x):
        return 1. + 0.5 * jnp.tanh(2. * x)

    #Get the derivative using jax
    def jax_derivatives(func, x):
        jax_deriv = jax.grad(func)
        return jax.vmap(jax_deriv)(x)
    
    #Set up plots, x values to evaluate function + derivatives
    fig, axs = plt.subplots(2, 1, sharex=True, dpi=300)
    x = np.linspace(-2, 2, 1000, dtype=np.float64)

    #Plot function, derivative, Jax-evaluated derivative
    axs[0].plot(x, f(x), label='Function')
    axs[0].plot(x, f_prime(x), label="Derivative")
    axs[0].plot(x, jax_derivatives(f_jax, x), label='Jax derivative', ls="dotted")
    axs[0].legend()
    axs[0].set_title(r"Derivative of $1 + \frac{1}{2}tanh(2x)$")
    axs[0].set_ylabel("Evaluation at x")

    #Plot residuals
    axs[1].plot(x, jax_derivatives(f_jax, x) - f_prime(x))
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("Residuals")

    plt.savefig("jax_deriv.eps", format="eps")
    plt.show()

# part_b()