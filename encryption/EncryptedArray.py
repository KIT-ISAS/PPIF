'''
Created on 25.02.2018

@author: Mikhail Aristov
'''
from numpy import ndarray, array, int64
from encryption import PaillierCryptosystem as Crypto

class EncryptedArray(object):
    '''
    A wrapper class for encrypting and storing entire arrays.
    '''

    def __init__(self, size, pk, plaintextBuffer = None):
        '''
        Constructor
        '''
        # Store the public key and initialize simple matrix operations
        self.PublicKey = pk

        # Store matrix dimensions
        self.Height = size[0]
        self.Width = size[1]
        totalElements = self.Height * self.Width
        
        # Check the buffer or fill data with zeros if empty
        content = [0] * totalElements
        if plaintextBuffer is not None:
            plaintextBuffer = array(plaintextBuffer, dtype=int64).flatten()
            for i in range(min(totalElements, len(plaintextBuffer))):
                content[i] = plaintextBuffer[i]
        self.DATA = [Crypto.Encrypt(self.PublicKey, int(m)) for m in content] 
        
    def Add(self, otherMatrix):
        if otherMatrix.Width != self.Width or otherMatrix.Height != self.Height:
            raise RuntimeError("matrix dimensions do not match!\n", self.DATA, "\n", otherMatrix.DATA)
        if otherMatrix.PublicKey[0] != self.PublicKey[0]:
            raise RuntimeError("other matrix is encrypted under different key!\n", self.DATA, "\n", otherMatrix.DATA)
        self.DATA = [Crypto.Add(self.PublicKey, self.DATA[i], otherMatrix.DATA[i]) for i in range(len(self.DATA))]

    def Decrypt(self, SecretKey):
        plaintexts = [Crypto.Decrypt(SecretKey, int(ct)) for ct in self.DATA]
        return ndarray((self.Height, self.Width), buffer = array(plaintexts, dtype=int64), dtype=int64)