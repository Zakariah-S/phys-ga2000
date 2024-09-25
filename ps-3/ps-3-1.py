'''
Read Example 4.3 (Pg. 137) in Newman. Using successively larger matrices (10×10, 30×30, etc.) find empirically and plot how the matrix 
multiplication computation rises with matrix size. Does it rise as N^3 as predicted? Use both an explicit function 
(i.e. the one in the example) and use the dot() method. How do they differ?
'''

import numpy as np
import timeit
import matplotlib.pyplot as plt

#Multiply two N x N matrices where elements are chosen randomly, according to the method shown in Example 4.3
def matrix_mult_textbook(A, B) -> np.ndarray:
    C = np.zeros_like(A) #product matrix
    N = A.shape[0]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i,j] += A[i, k] * B[j, k]
    
    return C

#Multiply two N x N matrices where elements are chosen randomly, using np.ndarray.dot()
def matrix_mult_numpy(A, B) -> np.ndarray:
    #A and B should be square matrices of the same size
    # A = np.random.random((N, N))
    # B = np.random.random((N, N))
    return A.dot(A)

def find_mult_times(func, N_low: int, N_high: int, N_step: int = np.uint(1), sample_size = 100):
    '''
    Iterate through the given range of Ns. 
    For each N: 
        Time the execution of one of the above matrix multiplication functions multiple times.
        Calculate the average and standard deviation of the times taken for each repetition.
    
    func: function
        The chosen matrix multiplication function. 
        This should be either matrix_mult_textbook or matrix_mult_numpy.
    N_low: int
        Minimum of the range of Ns to be tested.
    N_high: int
        Max of the range of Ns to be tested.
    N_step: int
        Distance between consecutive Ns in the iteration (default is 1).
    sample_size: int
        Number of repetitions of the function for each N. For a larger dataset (and tighter errors), this can be increased

    Returns:
    Ns, times, time_errors: nd.arrays
        Arrays containing the numbers N, the average time taken for each N, and the errors on these times.
    '''
    Ns = np.arange(N_low, N_high, N_step)
    times = np.zeros_like(Ns, dtype=np.float64)
    time_errors = np.zeros_like(Ns, dtype=np.float64)

    i = 0
    for N in Ns:
        time_list = timeit.repeat(stmt=f"{func.__name__}(A, B)", 
                                  setup=f"from __main__ import {func.__name__}\nimport numpy as np\nA=np.random.random(({N}, {N}))\nB=np.random.random(({N}, {N}))", 
                                         repeat= sample_size, number=1)
        time_list = np.array(time_list, dtype=np.float64)
        print(np.mean(time_list))
        times[i] = np.mean(time_list)
        time_errors[i] = np.std(time_list)
        i += 1

    return Ns, times, time_errors

if __name__ == "__main__":
    # Ns, times, errors = find_mult_times(matrix_mult_textbook, 1, 51, 1)
    Ns, times, errors = find_mult_times(matrix_mult_numpy, 1, 202, 10, sample_size=1)

    # plt.errorbar(np.power(Ns, 3), times, errors)
    plt.plot(np.power(Ns,3), times)
    # plt.xscale('log')
    # plt.yscale('log')

    plt.xlabel("Number of columns/rows N of matrix")
    plt.ylabel("Avg time taken for multiplication (s)")
    plt.title("Matrix multiplication time vs. N")
    plt.savefig("multiplytime.eps", format='eps')
    plt.show()