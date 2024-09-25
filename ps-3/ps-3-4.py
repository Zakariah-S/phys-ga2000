'''
Demonstrate that the central limit theorem works. Do so by generating random variate y = (1/N) * Sum(i=0)(N)(x_i),
where x_i is a random variate distributed as exp(−x) (you can use the np.random library to generate the exponentially-distributed 
variates). 
A) First, calculate analytically how you expect the mean (that one should be easy) and variance of y to vary with N. 
    (Done in overleaf)
B) Show visually that for large N the distribution of y tends towards Gaussian.
C) Show as a function of N how the mean, variance, skewness, and kurtosis of the distribution change. 
D) Estimate at which N the skewness and kurtosis have reached about 1% of their value for N = 1.
'''
import numpy as np
import matplotlib.pyplot as plt

#-----Part A: Calculate how mean and variance change with N-----#
'''
<y> = 1, regardless of N.
Var(y) = 1/N

When plugging these into the formula for a normalised Gaussian, we get:
G(y) = sqrt(N/2pi) * exp((-N/2) * (y-1)^2)

I have used G(y) instead of P(y) because the actual probability distribution P(y) only approaches a Gaussian, 
but might not equal it until N  = infinity.
'''
#-----Part B: Show visually that for large N the distribution of y tends towards Gaussian.-----#

def get_random_variates(x_num, y_num):
    '''
    Return an array of random variables y, with each y_i generated by 
    taking the mean of an array x_i of exponentially-distributed random variables.
    x_num: int
        The number of elements in each array x_i.
    y_num: int
        The number of elements in the returned array y.
    '''
    x = np.random.exponential(size=(y_num, x_num)) #Exponential random variables x
    y = np.mean(x, axis=1)
    
    return y

def emerging_gaussian(x, N):
    return np.sqrt(N / (2*np.pi)) * np.exp(-(N/2) * np.square(x-1))

def dist_vs_N(Ns, y_num=10000, bins=np.linspace(0, 5, 1000)):
    '''
    Get and plot the distribution of the arrays y generated by the above function, 
    for varying N (= x_num = the numbers of elements in the arrays x_i used to generate each random number y_i.)
    Ns: numpy.ndarray[np.int]
        Array of integers, each of which will be subsituted as x_num in the function get_random_variates once.
    y_num: int
        The number of random variates y_i generated for each call of get_random_variates.
    '''
    #For integration
    trapezoid_array = np.full(y_num, 1, dtype=np.float64)
    trapezoid_array[0] = 0.5
    trapezoid_array[-1] = 0.5
    width = bins[1] - bins[0]

    for N in Ns:
        #Get our variables y, bin them, normalise and plot the resulting histogram
        y = get_random_variates(x_num=N, y_num=y_num)
        hist, edges = np.histogram(y, bins)
        bin_centers = (edges[1:] + edges[:-1])/2

        integral = np.sum(y * width * trapezoid_array)
        normalised_hist = hist / integral

        # normalised_hist = hist / (y_num * (edges[1]-edges[0]))
        # print(np.sum(normalised_hist * (edges[1]-edges[0]))) #Debug
        plt.plot(bin_centers, normalised_hist, label=f"N = {N}")

        #Also plot a Gaussian with the same mean and variance as the current distribution
        if N != 1: plt.plot(bin_centers, emerging_gaussian(bin_centers, N), label=f"Gaussian for {N}", ls='dotted')

    #Plot the exponential curve that fits the histogram for N = 1
    plt.plot(bin_centers, np.exp(-bin_centers), ls='dotted')
    plt.legend()
    plt.title("Variation of P(y) with N")
    plt.xlabel('y')
    plt.ylabel('P(y)')
    plt.show()

# dist_vs_N(np.power(10, np.arange(0, 4+1)), y_num=10000, bins=np.linspace(0, 2, 50))
# dist_vs_N(np.array([1, 5, 10, 50, 100]), y_num=10000, bins=np.linspace(0, 2, 50))

#-----Part C: Show how the mean, variance, unbiased skewness, and unbiased kurtosis of Y change-----#

def m(i, x):
    '''
    General formula for the biased ith central moment of an array x. 
    I'm only using this to simplify the formulas for skew and kurtosis.
    '''
    mean = np.mean(x)
    return np.mean(np.power(x-mean, i))
    
def skew(x):
    '''
    Calculate the skewness of an array x.
    '''
    n = x.size
    biased_skew = m(3, x) / (np.power(m(2, x), 3/2))
    correction_factor = (np.sqrt(n * (n-1)) / (n-2))
    return correction_factor * biased_skew

def kurtosis(x):
    '''
    Calculate the kurtosis of an array x.
    '''
    n = x.size
    return (n-1)/(n-2)/(n-3) * ((n+1) * (m(4, x) / np.square(m(2, x))) - 3 * (n-1))

def stats_vs_N(Ns, y_num = 100000):
    Ns = np.asarray(Ns)
    means = np.zeros_like(Ns, dtype=np.float64)
    vars = np.zeros_like(Ns, dtype=np.float64)
    skews = np.zeros_like(Ns, dtype=np.float64)
    kurtoses = np.zeros_like(Ns, dtype=np.float64)

    i = 0
    for N in Ns:
        y = get_random_variates(x_num=N, y_num=y_num)
        means[i] = np.mean(y)
        vars[i] = np.var(y, ddof=1)
        skews[i] = skew(y)
        kurtoses[i] = kurtosis(y)
        i += 1
    
    plt.plot(Ns, means, label="Mean values", color='blue')
    plt.plot(Ns, np.ones_like(Ns), label="Expected mean", ls='dotted')

    plt.plot(Ns, vars, label="Variances")
    plt.plot(Ns, 1/Ns, label="Expected variances", ls='dotted')

    plt.plot(Ns, skews, label="Skews")

    plt.plot(Ns, kurtoses, label="Kurtoses")

    plt.legend()
    plt.title("Variation of y's Statistics with N")
    plt.xlabel('N (number of exponential terms summed to get y)')
    plt.ylabel('Statistic value')
    plt.savefig('ystats.eps', format='eps')
    plt.show()

# stats_vs_N(np.arange(1, 51, 1))

#-----Part D: Estimate at which N the skewness and kurtosis have reached about 1% of their value for N = 1-----#

def find_threshold_N(stat_func, threshold=0.01, y_num = 100000, iterations=1):
    '''
    For a given statistic-generating function that acts on the array of random variates y,
    find the threshold N after which stat_func(y) < (threshold) * (stat_func(x) for the distribution P(x) = e^(-x)).
    stat_func: func
        The statistic-generating function. Examples include np.mean, np.var, and the skew and kurtosis functions defined above.
    threshold: float
        Actually a factor multipled by the exponential distribution's statistic in order to get the threshold.
    y_num: int
        The number of random variates y in each statistical ensemble.
    '''
    N_vals = np.zeros(iterations, dtype=np.int128)

    def get_one_N():
        x = get_random_variates(1, y_num)
        exp_stat = stat_func(x)

        next_stat = exp_stat
        print(next_stat, 1)

        #Keep increasing the power of ten until we find the order of magnitude where our border N is
        x_num: int  = 1
        while next_stat > threshold * exp_stat:
            curr_stat = next_stat
            x_num *= 10
            next_stat = stat_func(get_random_variates(x_num, y_num))
            print(next_stat, x_num)

        #Adopt an additive step-size = current x_num and keep iterating over Ns. This lets us search the range [10^(x_num), 10^(x_num+1).
        #Once we once again reach the threshold skew, we go back to the previous x_num, divide our step size by 10, and proceed forward. We do this until the step size is 1.
        x_num = int(x_num / 10)
        step = x_num
        print(f"Step-size: {step}")
        while step != 0:
            next_stat = curr_stat #Set next_skew back to the skew value that was before our cutoff
            while next_stat > threshold * exp_stat:
                curr_stat = next_stat
                x_num += step
                next_stat = stat_func(get_random_variates(x_num, y_num))
                print(next_stat, x_num)
            x_num -= step
            step = int(step / 10)
            print(f"New step-size: {step}")
        return 0

# find_threshold_N(skew)
print(int(0.1))