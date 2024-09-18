import numpy as np

#Goal: to get the Madelung constant = sum_over_all_ijk_from_(-L)_to_(L)((i^2 + j^2 + k^2) ** (-1/2))

def get_madelung_loop(L: int) -> np.float64:
    #Set up range through which we want to loop for each axis
    atom_index = np.arange(-L, L + 1)
    
    #Function that outputs 1 if the input integer is even and -1 if it is odd
    def is_even(num: np.int64) -> np.float64:
        return np.float64(1) if num % 2 == 0 else np.float64(-1)

    madelung = np.float64(0)
    for i in atom_index:
        for j in atom_index:
            for k in atom_index:
                if i == 0 and j == 0 and k == 0:
                    continue
                madelung += is_even(i + j + k) * np.power(i**2 + j**2 + k**2, -1/2)
    # print(madelung)
    return madelung

get_madelung_loop(10)

def get_madelung_noloop(L: int) -> np.float64:
    #Create 3-D numpy arrays that hold the x, y, and z coordinates for each vertex in the L x L x L lattice
    x = y = z = np.arange(-L, L + 1, dtype=np.float64)
    i, j, k = np.meshgrid(x, y, z)
    # print(i)
    # print(j)
    # print(k)

    #create table of 1s or -1s depending on whether i + j + k (sum of indices) is odd or even
    signs = (i + j + k) % 2
    signs[signs == 1] = -1
    signs[signs == 0] = 1

    #Set distance = np.inf at the centre, so that potential term for that point is zeroed out
    i[L, L, L] = j[L, L, L] = k[L, L, L] = np.inf

    #tabulate potential contributions for each charge in the grid
    potential_table = signs * np.power(i**2 + j**2 + k**2, -1/2)
    # print(np.sum(potential_table))
    return np.sum(potential_table)

get_madelung_noloop(100)

if __name__ == "__main__":
    #-----Madellung calculations:
    M_true = -1.7476 #true Madellung constant
    M_1 = get_madelung_loop(100)
    print(M_1)
    # >> -1.7418198158396148
    abs_error = np.abs(M_true - M_1)
    print(abs_error)
    # >> 0.005780184160385282
    frac_error = np.abs(abs_error/M_true)
    print(frac_error)
    # >> 0.0033074983751346316

    M_2 = get_madelung_noloop(100)
    print(M_2)
    # >> -1.7418198158361038
    abs_error = np.abs(M_true - M_2)
    print(abs_error)
    # >> 0.0057801841638962514
    frac_error = np.abs(abs_error/M_true)
    print(frac_error)
    # >> 0.0033074983771436547

    #-----Timing the calculation
    import timeit
    time1 = timeit.timeit("get_madelung_loop(100)", setup="from __main__ import get_madelung_loop", number=1)
    print(time1)
    # >> 19.478233301 s for a lattice side length of 2(100) + 1
    time2 = timeit.timeit("get_madelung_noloop(100)", setup="from __main__ import get_madelung_noloop", number=1)
    print(time2)
    # >> 0.6514724299999983 s for a lattice side length of 2(100) + 1
    print(time1/time2)
    # >> Method 2 runs ~29.8987837459216x faster than method 1