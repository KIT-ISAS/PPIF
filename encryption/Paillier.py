'''
Created on 16.01.2018

@author: Mikhail Aristov
'''
from random import randint
from utility import next_prime, gcd, lcm, ModularIntegerInverse

class PaillierCryptosystem(object):
    '''
    This is the core implementation of the Paillier homomorphic cryptosystem.
    '''
    
    @staticmethod
    def GetRandomPrime(PrimeRange):
        searchStart = PrimeRange - randint(PrimeRange // 2, PrimeRange)
        return next_prime(searchStart)
    
    @staticmethod
    def L(x, n):
        return (x - 1) // n
    
    @staticmethod
    def KeyGen(KeyLength):
        # Pick two primes randomly
        PrimeRange = 2 ** (KeyLength // 2) - 1
        p, q = PaillierCryptosystem.GetRandomPrime(PrimeRange), PaillierCryptosystem.GetRandomPrime(PrimeRange)
        # Ensure the that gcd(pq, (p - 1)(q - 1)) == 1
        while gcd(p * q, (p - 1) * (q - 1)) != 1:
            q = PaillierCryptosystem.GetRandomPrime(PrimeRange)
        return PaillierCryptosystem.KeyGenFromPrimes(p, q)
    
    @staticmethod
    def KeyGenFromPrimes(p, q):
        # Ensure the that gcd(pq, (p - 1)(q - 1)) == 1
        assert(gcd(p * q, (p - 1) * (q - 1)) == 1)
        # Compute additional values
        n = p * q
        nSquared = n * n
        l = lcm(p - 1, q - 1)
        # Get a random integer generator
        g = randint(0, nSquared)
        # Ensure n divides the order of g
        tmp = PaillierCryptosystem.L(pow(g, l, nSquared), n)
        mu = ModularIntegerInverse(tmp, n)
        # Return pk, sk
        return (n, g), (l, mu, n)
    
    @staticmethod
    def Encrypt(pk, m):
        # Check the message for size
        n, g, nSquared = pk[0], pk[1], pk[0] * pk[0]
        # Handle negative plaintexts
        assert(abs(m) < (n // 2))
        if m < 0:
            m += n
        # Pick a random noise factor
        r = randint(0, n)
        # Compute ciphertext
        tmp1 = pow(g, m, nSquared)
        tmp2 = pow(r, n, nSquared)
        # Return 
        return tmp1 * tmp2 % nSquared
    
    @staticmethod
    def EncryptZeros(pk, Count = 1):
        assert(Count > 0)
        return [PaillierCryptosystem.Encrypt(pk, 0) for _ in range(Count)]
    
    @staticmethod
    def Decrypt(sk, c):
        # Check the ciphertext for size
        l, mu, n, nSquared = sk[0], sk[1], sk[2], sk[2] * sk[2]
        assert(c >= 0 and c < nSquared)
        # Compute plaintext message
        tmp = PaillierCryptosystem.L(pow(c, l, nSquared), n)
        result = tmp * mu % n
        # Handle negative plaintexts
        if result > (n // 2):
            result -= n
        return result
    
    @staticmethod
    def Add(pk, c1, c2):
        return (c1 * c2) % (pk[0] * pk[0])
    
    @staticmethod
    def Sub(pk, c1, c2):
        subtrahend = ModularIntegerInverse(c2, pk[0] * pk[0])
        return PaillierCryptosystem.Add(pk, c1, subtrahend)
    
    @staticmethod
    def Mult(pk, EncryptedFactor, PlaintextFactor):
        if PlaintextFactor == 0:
            # Anything multiplied by zero is zero, so return a fresh zero encryption
            return PaillierCryptosystem.Encrypt(pk, 0)
        else:
            nSquared = pk[0] * pk[0]
            result = pow(EncryptedFactor, abs(PlaintextFactor), nSquared)
            # Invert result if the plaintext factor was negative
            if PlaintextFactor < 0:
                result = ModularIntegerInverse(result, nSquared)
            return result