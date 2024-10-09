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

    #Set up x values to evaluate function + derivatives
    x = np.linspace(-2, 2, 1000, dtype=np.float64)

    #Plot function, derivative, approximated derivative
    plt.plot(x, f(x), label=r"$f(x) = 1 + \frac{1}{2}tanh(2x)$")
    plt.plot(x, f_prime(x), label=r"$f'(x) = sech^2(2x)$")
    plt.plot(x, get_derivatives(f, x), label=r"Approximation of $f'(x)$", ls="dotted")
    plt.legend()
    plt.title(r"Derivative of $1 + \frac{1}{2}tanh(2x)$")
    plt.xlabel("x")
    plt.ylabel("Evaluation at x")

    plt.savefig("cdd.eps", format='eps')
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
    x = np.linspace(-2, 2, 1000, dtype=np.float64)

    #Plot function, derivative, Jax-evaluated derivative
    plt.plot(x, f(x), label=r"$f(x) = 1 + \frac{1}{2}tanh(2x)$")
    plt.plot(x, f_prime(x), label=r"$f'(x) = sech^2(2x)$")
    plt.plot(x, jax_derivatives(f_jax, x), label=r"$f'(x)$ using Jax", ls="dotted")
    plt.legend()
    plt.title(r"Derivative of $1 + \frac{1}{2}tanh(2x)$")
    plt.xlabel("x")
    plt.ylabel("Evaluation at x")

    plt.savefig("jax_deriv.eps", format="eps")
    plt.show()

part_b()