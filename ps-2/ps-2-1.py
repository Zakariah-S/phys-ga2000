import numpy as np

#-----Using NumPy's frexp function and working exclusively in floats-----#
print("*****Attempt 1*****")
num: float = 100.98763
# print(f'Number: {num}')

#---Using float32s (singles):
decimal_mantissa32, exp32 = np.frexp(np.float32(num)) #frexp returns the mantissa and exponent in decimal format
# print(decimal_mantissa32)
print(f'Number when converted to float32 and back: {decimal_mantissa32 * 2 ** exp32}')
# >> 100.98763275146484

#---Using float64s (doubles):
decimal_mantissa64, exp64 = np.frexp(np.float64(num))
# print(decimal_mantissa64)
print(f'Number when converted to float64 and back: {decimal_mantissa64 * 2 ** exp64}')
# >> 100.98763

#---Difference between the two
difference = (np.float64(decimal_mantissa32 * 2 ** exp32) - decimal_mantissa64 * 2 ** exp64)
print(f'Difference between the two results: {difference}')
# >> 2.7514648479609605e-06

#-----Using Professor Blanton's get_bits function to reproduce the represented number as an exact fraction-----#
print("\n*****Attempt 2*****")
#The following function is from cell 13 of Professor Blanton's 'Intro' Jupyter noteook.
def get_bits(num) -> list:
    bytes = num.tobytes()
    # print(bytes)
    bits = []
    for byte in bytes:
        bits = bits + np.flip(np.unpackbits(np.uint8(byte)), np.uint8(0)).tolist()
    return list(reversed(bits))

def bit_to_decimal(bits, starting_power: int) -> np.float64:
        '''
        Convert a list of bits (read in order of most- to least- significant) from binary to decimal.
        bits = list or array of bits (integers that are either 0 or 1)
        starting_power = the power of 2 represented by the most significant (leftmost) bit.
        '''
        number = np.float64(0)
        power = np.float64(starting_power)
        for bit in bits:
            number += np.float64(bit) * (np.float64(2)) ** power
            power -= 1
        # print(number)
        return number

def reconstruct_decimal(bit_list):
    #Reconstruct a 32-bit number from its list of bits, as a decimal. 
    sign = np.float64(-1) ** bit_list[0]
    exponent_bits = bit_list[1:9]
    mantissa_bits = bit_list[9:]
    # print(exponent_bits)
    # print(mantissa_bits)
    
    exponent = bit_to_decimal(exponent_bits, 7) - np.float64(127)
    # print(exponent)
    mantissa = 1 + bit_to_decimal(mantissa_bits, -1)
    # print(mantissa)

    # print(sign * mantissa * np.float64(2) ** exponent)

def reconstruct_fraction(bit_list):
    #Reconstruct a 32-bit number from its list of bits, as a fraction. Returns the numerator and denominator of the number.
    sign = np.float64(-1) ** bit_list[0]
    exponent_bits = bit_list[1:9]
    mantissa_bits = bit_list[9:]

    #Get exponent
    exponent = bit_to_decimal(exponent_bits, 7) - np.float64(127)

    #Reconstruct mantissa as fraction
    numerator = 0
    denominator = np.float64(1)
    for bit in mantissa_bits:
        numerator = numerator * np.float64(2) + bit
        denominator *= np.float64(2)
    # print(numerator, denominator)
    # print(numerator/denominator)
    numerator = sign * (numerator + denominator) * np.float64(2) ** exponent

    #Divide numerator and denominator by their gcf, which we get using euclid's algorithm
    a, b = numerator, denominator
    while b != 0:
         a, b = b, a%b
    gcf = np.float64(a)
    # print(gcf)
    numerator /= a
    denominator /= a

    print(f"The 32-bit representation of this number is equal to {numerator}/{denominator}")
    print(f"In decimal-point format, this is {numerator/denominator}")
    return numerator, denominator

bits = get_bits(np.float32(num))
reconstruct_decimal(bits)
numerator, denominator = reconstruct_fraction(bits)
# >> 13236651.0/131072.0 = 100.98763275146484
difference = np.abs(numerator/denominator - np.float64(num))
print(f"The difference between the real value and the represented one is approximately {difference}")
# >> 2.7514648479609605e-06