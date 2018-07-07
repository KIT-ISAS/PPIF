'''
Created on 09.03.2018

@author: Mikhail Aristov
'''
import numpy as np
from numpy.random import normal as Gauss
from encryption import PaillierCryptosystem as Crypto
from simulation import Parameters as param

class SimAgent(object):
    '''
    This class simulates a mobile agent moving through the simulated terrain.
    '''

    def __init__(self, ID, CentralSensorHub):
        '''
        Constructor
        '''
        self.ID = ID
        self.Name = "Agent" + str(self.ID)
        
        # Generate a public-private key pair
        # self.pk, self.sk = Crypto.KeyGen(param.CRYPTO_KEY_LENGTH)
        
        # From the security standpoint, each agent should have its own key pair, but for this simulation,
        # we hard-wire a simple 64-bit key (not secure in any form or shape!) for performance reasons
        self.pk, self.sk = Crypto.KeyGenFromPrimes(5915587277, 5754853343)
        
        # Initialize position and velocity
        self.MyPos = self.SampleUniformPositionOnTheSquareEdge(param.AREA_SIDE_LENGTH)
        self.MyVelocity = self.SampleGauissanVelocityVectorPointingInwards(self.MyPos, param.AREA_SIDE_LENGTH, param.AGENT_VELOCITY_SIGMA)
        
        # Initialize system model
        self.SystemMatrix = np.identity(2, dtype=float)
        self.ProcessNoise = np.identity(2, dtype=float) * param.AGENT_VELOCITY_SIGMA * param.AGENT_VELOCITY_SIGMA
    
        # Initialize state estimate and its covariance matrix for encrypted filter (unnormalized and normalized)
        self.xEstEnc,  self.CxEnc  = np.ndarray((2,1), buffer = np.ones((2), dtype=float) * (param.AREA_SIDE_LENGTH / 2)), np.identity((2), dtype=float) * param.QUANTIZATION_FACTOR_24
        self.xEstEncN, self.CxEncN = np.copy(self.xEstEnc), np.copy(self.CxEnc)
        # Same for unencrypted in different quantization levels
        self.xEstF,  self.CxF  = np.copy(self.xEstEnc), np.copy(self.CxEnc)
        self.xEst8,  self.Cx8  = np.copy(self.xEstEnc), np.copy(self.CxEnc)
        self.xEst16, self.Cx16 = np.copy(self.xEstEnc), np.copy(self.CxEnc)
        self.xEst24, self.Cx24 = np.copy(self.xEstEnc), np.copy(self.CxEnc)
        # Same for unencrypted AND normalized
        self.xEstFN,  self.CxFN  = np.copy(self.xEstEnc), np.copy(self.CxEnc)
        self.xEst8N,  self.Cx8N  = np.copy(self.xEstEnc), np.copy(self.CxEnc)
        self.xEst16N, self.Cx16N = np.copy(self.xEstEnc), np.copy(self.CxEnc)
        self.xEst24N, self.Cx24N = np.copy(self.xEstEnc), np.copy(self.CxEnc)

        # Set sensor hub
        self.MySensor = CentralSensorHub
    
    def Update(self):
        '''
        Updates the agent's position after one time step, as well as its estimates of it (including both prediction and filtering).
        
        Return False if the agent would leave the simulated area with this update, or True otherwise.
        If False is returned, no estimates are computed.
        '''
        # Update my real position (and quit if this makes me leave the simulation space)
        self.MyPos += self.MyVelocity
        if self.MyPos[0] < 0 or self.MyPos[0] > param.AREA_SIDE_LENGTH or self.MyPos[1] < 0 or self.MyPos[1] > param.AREA_SIDE_LENGTH:
            return False
        
        # Prediction step on encrypted measurements
        if not param.DO_NOT_ENCRYPT:
            self.xEstEnc,  self.CxEnc,  yVecEnc,  yMatEnc  = self.PredictionStep(self.xEstEnc,  self.CxEnc,  fast=True)
            self.xEstEncN, self.CxEncN, yVecEncN, yMatEncN = self.PredictionStep(self.xEstEncN, self.CxEncN, fast=True)
        
        # Prediction step on regular controls
        self.xEstF,  self.CxF,  yVecF,  yMatF  = self.PredictionStep(self.xEstF,  self.CxF,  fast=True)
        self.xEst8,  self.Cx8,  yVec8,  yMat8  = self.PredictionStep(self.xEst8,  self.Cx8,  fast=True)
        self.xEst16, self.Cx16, yVec16, yMat16 = self.PredictionStep(self.xEst16, self.Cx16, fast=True)
        self.xEst24, self.Cx24, yVec24, yMat24 = self.PredictionStep(self.xEst24, self.Cx24, fast=True)

        # Prediction step on normalized controls
        self.xEstFN,  self.CxFN,  yVecFN,  yMatFN  = self.PredictionStep(self.xEstFN,  self.CxFN,  fast=True)
        self.xEst8N,  self.Cx8N,  yVec8N,  yMat8N  = self.PredictionStep(self.xEst8N,  self.Cx8N,  fast=True)
        self.xEst16N, self.Cx16N, yVec16N, yMat16N = self.PredictionStep(self.xEst16N, self.Cx16N, fast=True)
        self.xEst24N, self.Cx24N, yVec24N, yMat24N = self.PredictionStep(self.xEst24N, self.Cx24N, fast=True)
        
        # Obtain encrypted measurements from the sensor grid
        encVec, encMat, encVecN, encMatN, iVecF, iMatF, iVec8, iMat8, iVec16, iMat16, iVec24, iMat24, iVecFN, iMatFN, iVec8N, iMat8N, iVec16N, iMat16N, iVec24N, iMat24N = self.MySensor.GetAggregatedMeasurements(self.MyPos, self.pk, fast=True)
 
        # Decrypt the measurements
        if not param.DO_NOT_ENCRYPT:
            iVecEnc,  iMatEnc  = self.DecryptMeasurementResults(encVec,  encMat,  iVec16)
            iVecEncN, iMatEncN = self.DecryptMeasurementResults(encVecN, encMatN, iVec16N)
 
        # Unquantize the measurements
        iVec8,   iMat8   = self.Unquantize(iVec8,   iMat8,   param.QUANTIZATION_FACTOR_8)
        iVec16,  iMat16  = self.Unquantize(iVec16,  iMat16,  param.QUANTIZATION_FACTOR_16)
        iVec24,  iMat24  = self.Unquantize(iVec24,  iMat24,  param.QUANTIZATION_FACTOR_24)
        iVec8N,  iMat8N  = self.Unquantize(iVec8N,  iMat8N,  param.QUANTIZATION_FACTOR_8)
        iVec16N, iMat16N = self.Unquantize(iVec16N, iMat16N, param.QUANTIZATION_FACTOR_16)
        iVec24N, iMat24N = self.Unquantize(iVec24N, iMat24N, param.QUANTIZATION_FACTOR_24)
        
        # Apply the information filter to each instance
        self.xEstF,   self.CxF   = self.InformationFilterStep(yVecF,   yMatF,   iVecF,   iMatF)
        self.xEst8,   self.Cx8   = self.InformationFilterStep(yVec8,   yMat8,   iVec8,   iMat8)
        self.xEst16,  self.Cx16  = self.InformationFilterStep(yVec16,  yMat16,  iVec16,  iMat16)
        self.xEst24,  self.Cx24  = self.InformationFilterStep(yVec24,  yMat24,  iVec24,  iMat24)
        self.xEstFN,  self.CxFN  = self.InformationFilterStep(yVecFN,  yMatFN,  iVecFN,  iMatFN)
        self.xEst8N,  self.Cx8N  = self.InformationFilterStep(yVec8N,  yMat8N,  iVec8N,  iMat8N)
        self.xEst16N, self.Cx16N = self.InformationFilterStep(yVec16N, yMat16N, iVec16N, iMat16N)
        self.xEst24N, self.Cx24N = self.InformationFilterStep(yVec24N, yMat24N, iVec24N, iMat24N)
        if not param.DO_NOT_ENCRYPT:
            self.xEstEnc,  self.CxEnc  = self.InformationFilterStep(yVecEnc,  yMatEnc,  iVecEnc,  iMatEnc)
            self.xEstEncN, self.CxEncN = self.InformationFilterStep(yVecEncN, yMatEncN, iVecEncN, iMatEncN)
        
        return True
    
    def SampleUniformPositionOnTheSquareEdge(self, SquareSize):
        result = np.random.random_sample(2) * SquareSize
        # Pick a side to project onto
        side = np.random.randint(0, 4)
        if side == 0:
            result[0] = 0
        elif side == 1:
            result[0] = SquareSize
        elif side == 2:
            result[1] = 0
        elif side == 3:
            result[1] = SquareSize
        return np.ndarray((2,1), buffer = result)
    
    def SampleGauissanVelocityVectorPointingInwards(self, InitialPos, FieldSize, VelocitySigma):
        result = np.ndarray((2,1), buffer = Gauss(0, VelocitySigma, 2))
        # Flip the velocity if it would cause the plane to leave the field within three time steps
        projection = InitialPos + result * 3
        if projection[0] < 0 or projection[0] > FieldSize:
            result[0] *= -1
        if projection[1] < 0 or projection[1] > FieldSize:
            result[1] *= -1
        return result
    
    def Unquantize(self, InformationVector, InformationMatrix, QuantizationFactor):
        return InformationVector.astype(float) / QuantizationFactor, InformationMatrix.astype(float) / QuantizationFactor
    
    def PredictionStep(self, estimate, covariance, fast = False):
        predictedCov = covariance + self.ProcessNoise if fast else np.dot(self.SystemMatrix, np.dot(covariance, self.SystemMatrix)) + self.ProcessNoise
        predictedCovInfo = np.linalg.inv(predictedCov)
        predictedState = np.copy(estimate) if fast else np.dot(self.SystemMatrix, estimate)
        predictedStateInfo = np.dot(predictedCovInfo, predictedState)
        return predictedState, predictedCov, predictedStateInfo, predictedCovInfo
    
    def InformationFilterStep(self, predictedInfoVector, predictedInfoMatrix, measurementInfoVector, measurementInfoMatrix):
        filteredCov = np.linalg.inv(predictedInfoMatrix + measurementInfoMatrix)
        filteredState = np.dot(filteredCov, predictedInfoVector + measurementInfoVector)
        return filteredState, filteredCov
        
    def DecryptMeasurementResults(self, encInfoVector, encInfoMatrix, validationVector):
        assert(not param.DO_NOT_ENCRYPT)
        # The try-catch is here for debugging overflow errors
        try:
            iVecDecQuantized = encInfoVector.Decrypt(self.sk)
            iVecDec, iMatDec = self.Unquantize(iVecDecQuantized, encInfoMatrix.Decrypt(self.sk), param.QUANTIZATION_FACTOR_16)
        except OverflowError as e:
            print("info vector", encInfoVector.DATA)
            print("info matrix", encInfoMatrix.DATA)
            raise e
        
        # Check for major discrepancies between decrypted and unencrypted measurements, which is indicative of an encryption overflow
        if iVecDecQuantized != validationVector:
            print("encryption overflow error!")
            print(encInfoVector.DATA)
            print(iVecDecQuantized.flatten())
            print(iVecDec.flatten())
            print(validationVector.flatten())
        
        return iVecDec, iMatDec