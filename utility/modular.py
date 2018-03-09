'''
Created on 17.12.2017

@author: Mikhail Aristov
'''
import numpy as np
import timeit as ti

# Calculates how many bits are necessary to store an integer number (sans sign bit)
def BitSize(n):
    n = abs(int(n))
    result = 0
    while(n > 0):
        n = n >> 1
        result += 1
    return result

# Extended Euclidean algorithm
def ExtendedIntegerEuclidean(a, b):
    a, b = int(round(a)), int(round(b))
    if a == 0:
        return (b, 0, 1)
    else:
        g, x, y = ExtendedIntegerEuclidean(b % a, a)
        return (g, y - (b // a) * x, x)

# Greatest common denominator
def gcd(a, b):
    result, _, _ = ExtendedIntegerEuclidean(a, b)
    return result

# Lowest common multiple
def lcm(a, b):
    return (a * b) // gcd(a, b)

# Returns the multiplicative inverse of a modulo m
def ModularIntegerInverse(a, m):
    a, m = int(round(a)), int(round(m))
    g, x, _ = ExtendedIntegerEuclidean(a, m)
    if g != 1:
        #print(g, a, m)
        raise ArithmeticError('modular inverse does not exist', g, a, m)
    else:
        return x % m

# Gaussian elimination within the ring
def ModularMatrixInverse(Matrix, Modulus):
    # Assert that matrix is square
    assert(len(Matrix.shape) == 2)
    assert(Matrix.shape[0] == Matrix.shape[1])
    matrixSize = Matrix.shape[0]
    # Assert that matrix is invertible
    if(int(round(np.linalg.det(Matrix))) == 0):
        raise ArithmeticError('the matrix is not invertible, because its determinant is zero!')
    # Initialize the result as identity matrix
    result = np.zeros((matrixSize, matrixSize),dtype=np.int64)
    for i in range(0, matrixSize):
        result[i, i] = 1
    # Invert the matrix
    startTime = ti.default_timer()
    for i in range(0, matrixSize):
        # Reduce current row so its leading coefficient is 1
        factor = np.int64(ModularIntegerInverse(Matrix[i][i], Modulus))
        Matrix[i] = [MultModulo(v, factor, Modulus) for v in Matrix[i]]
        result[i] = [MultModulo(v, factor, Modulus) for v in result[i]] # Dito for the result matrix...
        # Subtract current row from every other row
        for j in range(0, matrixSize):
            if(i != j):
                factor = Matrix[j][i]
                Matrix[j] = [(Matrix[j][k] - MultModulo(Matrix[i][k], factor, Modulus)) % Modulus for k in range(matrixSize)]
                result[j] = [(result[j][k] - MultModulo(result[i][k], factor, Modulus)) % Modulus for k in range(matrixSize)]
    print("matrix of size", matrixSize, "inverted in", round(ti.default_timer() - startTime, 5), "seconds")
    return result

# This ensures that the product of factors does not cause integer overflows
def MultModulo(Factor1, Factor2, Modulus):
    return np.int64(int(Factor1) * int(Factor2) % int(Modulus))

# Uses bitwise operators to accelerate multiplication modulo a power of 2
def MultModuloPowerOf2(Factor1, Factor2, ModulusPower):
    # Check the modulus and prepare the bitmask
    assert(ModulusPower <= 63)
    bitMask = 2 ** ModulusPower - 1
    Factor1, Factor2 = Factor1 & bitMask, Factor2 & bitMask
    # Loop through bits of the first factor and
    result = 0
    while Factor1 > 0:
        # Add the second factor, properly bit-shifted, to the result
        if Factor1 & 1:
            result = (result + Factor2) & bitMask
        # Bit-shift the factors (within the bitmask)
        Factor2 = (Factor2 & bitMask) << 1
        Factor1 = Factor1 >> 1
    return result