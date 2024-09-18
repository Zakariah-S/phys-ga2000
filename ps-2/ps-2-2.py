import numpy as np

''' 
Assignment A: find the smallest floats (in both 32- and 64-bit) that can be added to 1 in order to get a value != 1.
'''

def smallest_significant_float(float_func = np.float32, positive = True):
    addend = float_func(1)
    while float_func(1) + addend != float_func(1):
        positive_result = addend
        addend /= float_func(2)
    if positive: return positive_result
    else: #Look at the smallest number you can subtract from 1. If it smaller than the positive result, return that instead.
        addend = float_func(1)
        while float_func(1) - addend != float_func(1):
            negative_result = addend
            addend /= float_func(2)
        if negative_result < positive_result: return negative_result
        else: return positive_result

def problem_2_2a_solver():
    #---When only looking at adding numbers to 1:
    result32 = smallest_significant_float(np.float32)
    print(result32)
    # >> 1.1920929e-07
    print(f"log_2 of the above: {np.log2(result32)}")
    # >> -23

    result64 = smallest_significant_float(np.float64)
    print(result64)
    # >> 2.220446049250313e-16
    print(f"log_2 of the above: {np.log2(result64)}")
    # >> -52

    #---When we also consider the smallest number we can subtract from 1:
    result32 = smallest_significant_float(np.float32, positive=False)
    print(result32)
    # >> 5.9604645e-08
    print(f"log_2 of the above: {np.log2(result32)}")
    # >> -24

    result64 = smallest_significant_float(np.float64, positive=False)
    print(result64)
    # >> 1.1102230246251565e-16
    print(f"log_2 of the above: {np.log2(result64)}")
    # >> -53 

problem_2_2a_solver() #So I only have to comment out one line when I'm working on the other parts of the problem

''' 
Assignment B: Find (approximately) the minimum and maximum positive numbers in 32-bit and 64-bit representation.
'''

def find_extrema(float_func = np.float32):
    #Find smallest positive number
    start = float_func(1)
    curr = start
    while True:
        temp = curr / float_func(2)
        if temp == float_func(0): 
            smallest = curr
            break
        else: curr = temp
    #Find largest representable power of two
    curr = start
    while True:
        temp = curr * float_func(2)
        if temp == np.inf:
            almost_largest = curr
            break
        else: 
            curr = temp
    #Keep adding smaller powers of 2 to almost_largest until integer overflow occurs
    power = float_func(1)
    largest = almost_largest
    while True:
        temp = largest + almost_largest / (2 ** power)
        power += 1
        # print(temp, largest)
        if temp==np.inf or temp==largest:
            break
        else:
            largest = temp
    return smallest, largest

def problem_2_2b_solver():
    small32, large32 = find_extrema(np.float32)
    print(small32, large32)
    # >> 1e-45, 3.402823669209385e+38
    print(np.log2(small32))
    # >> -149

    small64, large64 = find_extrema(np.float64)
    print(small64, large64)
    # >> 5e-324, 1.7976931348623157e+308
    print(np.log2(small64))
    # >> -1074

problem_2_2b_solver() #So I only have to comment out one line when I'm working on the other parts of the problem

'''
We can actually use the class numpy.finfo to find the exact answers to both of the above problems.
'''

#Difference between 1. and the next representable float larger than 1.:
def solve_problem_using_numpy_builtins():
    print("***\nUsing np.finfo:\n***")
    print(f"The smallest float32 you can add to 1: {np.finfo(np.float32).eps}")
    print(f"The smallest float64 you can add to 1: {np.finfo(np.float64).eps}\n***")
    print(f"The smallest float32 you can subtract from 1: {np.finfo(np.float32).epsneg}")
    print(f"The smallest float64 you can subtract from 1: {np.finfo(np.float64).epsneg}\n***")
    print(f"The smallest positive float32 is: {np.finfo(np.float32).smallest_subnormal}.")
    print(f"The largest float32 is {np.finfo(np.float32).max}")
    print(f"The smallest positive float64 is: {np.finfo(np.float64).smallest_subnormal}.")
    print(f"The largest float64 is {np.finfo(np.float64).max}")

solve_problem_using_numpy_builtins() #So I only have to comment out one line when I'm working on the other parts of the problem
print(np.log2(np.finfo(np.float32).smallest_subnormal))