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
    #Madellung calculations:
    # print(get_madelung_loop(100))
    # >> -1.7418198158396148
    # print(get_madelung_noloop(100))
    # >> -1.7418198158361038

    #Timing the calculation
    import timeit
    print(timeit.timeit("get_madelung_loop(40)", setup="from __main__ import get_madelung_loop", number=100))
    # >> 131.251118484 s for a lattice side length of 2(40) + 1
    print(timeit.timeit("get_madelung_noloop(40)", setup="from __main__ import get_madelung_noloop", number=100))
    # >> 2.931880102000008 s for a lattice side length of 2(40) + 1