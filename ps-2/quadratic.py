import numpy as np

#Part A------#
def quadratic_a(a, b, c) -> np.ndarray:
    '''
    Find roots of a quadratic with form ax^2 + bx + c, where a, b, and c are real coefficients.
    This function uses the standard quadratic formula (-b +/- sqrt(b^2 - 4ac)) / 2a.
    '''
    a, b, c = [np.float64(n) for n in [a, b, c]]

    #If determinant b^2 - 4ac < 0, work using complex numbers
    determinant: np.float64 = np.square(b) - 4*a*c
    if determinant < 0:
        #Get a, b, and c as numpy complex floats
        a, b, c = [np.complex128(n) for n in [a, b, c]]
        solutions: np.ndarray = np.array([], dtype=np.complex128)
        #Change the determinant to a complex number
        determinant = np.complex128(determinant)
    else:
        solutions: np.ndarray = np.array([], dtype=np.float64)

    #Now actually use the quadratic formula
    for k in np.arange(1, -2, -2):
        solution = -b / (2*a) + k * np.sqrt(determinant) / (2*a)
        solutions = np.append(solutions, solution)

    print(f"Solutions: {solutions}")
    return solutions
# >> solutions: [-1.00000761e-06 -1.00000000e+06]

#Part B------#
def quadratic_b(a, b, c) -> np.ndarray:
    '''
    Find the roots of a quadratic with form ax^2 + bx + c, where a, b, and c are real coefficients.
    This function uses the alternative quadratic formula 2c / (-b -/+ sqrt(b^2 - 4ac)).
    '''
    a, b, c = [np.float64(n) for n in [a, b, c]]

    #If determinant b^2 - 4ac < 0, work using complex numbers
    determinant: np.float64 = np.square(b) - 4*a*c
    if determinant < 0:
        #Get a, b, and c as numpy complex floats
        a, b, c = [np.complex128(n) for n in [a, b, c]]
        solutions: np.ndarray = np.array([], dtype=np.complex128)
        #Change the determinant to a complex float
        determinant = np.complex128(determinant)
    else:
        solutions: np.ndarray = np.array([], dtype=np.float64)

    #Actually use the formula now
    for k in np.arange(-1, 2, 2):
        solution = 2 * c / (-b + k * np.sqrt(determinant))
        solutions = np.append(solutions, solution)

    print(f"Solutions: {solutions}")
    return solutions
# >> solutions: [-1.00000000e-06 -1.00001058e+06]

#Part C-----#
'''
The functions from parts A and B above encounter errors because the the determinant (b^2 - 4ac) is very close to b^2.
Because of this, the calculation problem -b +/- sqrt(determinant) requires the computer to take the difference of two
almost-equivalent numbers, which causes large errors. 
'''

def quadratic(a, b, c) -> np.ndarray:
    a, b, c = [np.float64(n) for n in [a, b, c]]
    #If determinant b^2 - 4ac < 0, work using complex numbers
    determinant: np.float64 = np.square(b) - 4*a*c
    if determinant < 0:
        #Get a, b, and c as numpy complex floats
        a, b, c = [np.complex128(n) for n in [a, b, c]]
        solutions: np.ndarray = np.array([], dtype=np.complex128)
        #Change the determinant to a complex number
        determinant = np.complex128(determinant)
        solution_1 = 2*c/(-b - np.sqrt(determinant))
        solution_2 = (-b - np.sqrt(determinant)) / (2*a)
    else: #work using real numbers i.e. float64s
        solutions: np.ndarray = np.array([], dtype=np.float64)
    #Get the two solutions, but using the equation forms from a and b that don't cause huge subtraction errors
    if b >= np.float64(0):
        solution_1 = 2*c/(-b - np.sqrt(determinant))
        solution_2 = (-b - np.sqrt(determinant)) / (2*a)
    else: 
        solution_1 = (-b + np.sqrt(determinant)) / (2*a)
        solution_2 = 2*c/(-b + np.sqrt(determinant))
    solutions = np.append(solutions, np.array([solution_1, solution_2]))

    print(f"Solutions: {solutions}")
    return solutions

#Function that evaluates the quadratic ax^2 + bx + c for an array of given values x.
def plug_into_quadratic(x, a, b, c):
    a, b, c = [np.float64(n) for n in [a, b, c]]
    result = a * np.square(x) + b * x + c
    print(f"Plugging these into the quadratic gives: {result}")
    return result

if __name__ == "__main__":
    solutions_a = quadratic_a(0.001, 1000, 0.001)
    # >> [-1.00000761e-06 -1.00000000e+06]
    plug_into_quadratic(solutions_a, 0.001, 1000, 0.001)
    # >> [-7.61449237e-09  7.24792480e-08]

    solutions_b = quadratic_b(0.001, 1000, 0.001)
    # >> [-1.00000000e-06 -1.00001058e+06]
    plug_into_quadratic(solutions_b, 0.001, 1000, 0.001)
    # >> [0. 10575.62534721]

    solutions = quadratic(0.001, 1000, 0.001)
    # >> [-1.e-06 -1.e+06]
    plug_into_quadratic(solutions, 0.001, 1000, 0.001)
    # >> [0.0000000e+00 7.2479248e-08]
