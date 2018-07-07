'''
Created on 24.02.2018

@author: Mikhail Aristov
'''
import numpy as np
from numpy.random import normal as Gauss
from encryption import EncryptedArray
from simulation import Parameters as param

class SimSensor(object):
    '''
    Simulated sensor measuring the position of the tracked object via angle and distance to it.
    Dynamically estimates the measurement covariance around the measured position.
    '''
    def __init__(self, ID, PosX, PosY):
        '''
        Constructor
        '''
        self.ID = ID
        self.Name = "Sensor" + str(self.ID)
        self.MyPos = np.ndarray((2,1), buffer = np.array([PosX, PosY], dtype = float))
        self.MyNeighbours = []
        
        self.MeasurementMatrix = np.identity(2, dtype = float)
        self.PolarCovarianceMatrix = np.ndarray((2,2), buffer = np.array([param.SENSOR_DISTANCE_SIGMA * param.SENSOR_DISTANCE_SIGMA, 0, 0, param.SENSOR_AZIMUTH_SIGMA * param.SENSOR_DISTANCE_SIGMA], dtype=float))
    
        self.LastMeasurementCount = 0

    def GlobalCartesianToLocalPolar(self, CartesianCoordinates):
        assert(CartesianCoordinates.shape == (2,1))
        localCoord = CartesianCoordinates - self.MyPos
        azimuth = np.arctan2(localCoord[1], localCoord[0])
        distance = np.linalg.norm(localCoord)
        return azimuth, distance
    
    def LocalPolarToGlobalCartesian(self, Azimuth, Distance):
        return np.ndarray((2,1), buffer = np.array([np.cos(Azimuth), np.sin(Azimuth)]), dtype = float) * Distance + self.MyPos
    
    def AddPolarNoise(self, realAzimuth, realDist):
        # Apply Gaussian noise to azimuth and distance
        noisyAzimuth = Gauss(realAzimuth, param.SENSOR_AZIMUTH_SIGMA)
        noisyDist = Gauss(realDist, param.SENSOR_DISTANCE_SIGMA)
        # Get corresponding noisy relative position and return
        noisyPos = self.LocalPolarToGlobalCartesian(noisyAzimuth, noisyDist)
        return noisyPos, noisyAzimuth, noisyDist
    
    def GetCovarianceAtMeasuredPos(self, noisyAzimuth, noisyDist):
        sinTheta, cosTheta = np.sin(noisyAzimuth), np.cos(noisyAzimuth)
        JacobianContent = np.array([cosTheta, -sinTheta * noisyDist, sinTheta, cosTheta * noisyDist])
        Jacobian = np.ndarray((2,2), buffer = JacobianContent)
        return np.dot(np.dot(Jacobian, self.PolarCovarianceMatrix), np.transpose(Jacobian))
    
    def CanSeeAgent(self, realPos):
        _, realDist = self.GlobalCartesianToLocalPolar(realPos)
        return (realDist <= param.SENSOR_DETECTION_RANGE)
    
    def GetNoisyMeasurement(self, realPos):
        # Simulate the azimuth and distance to the realPos from MyPos
        realAzimuth, realDist = self.GlobalCartesianToLocalPolar(realPos)
        if realDist > param.SENSOR_DETECTION_RANGE:
            raise ValueError("target out of detection range")
        #Add noise and return
        return self.AddPolarNoise(realAzimuth, realDist)
    
    def GetMeasurementAndCovariance(self, realPos):
        noisyPos, noisyAzimuth, noisyDist = self.GetNoisyMeasurement(realPos)
        covarianceMatrix = self.GetCovarianceAtMeasuredPos(noisyAzimuth, noisyDist)
        return noisyPos, covarianceMatrix

    def GetMeasurementInformationForm(self, realPos, fast = False):
        measurement, covariance = self.GetMeasurementAndCovariance(realPos)
        invCovariance = np.linalg.inv(covariance)
        # If the measurement matrix is an identity, we can fast-track the calculations
        if fast:
            return np.dot(invCovariance, measurement), invCovariance
        # Otherwise, we do the full information filter
        factor = np.dot(np.transpose(self.MeasurementMatrix), invCovariance)
        informationVector = np.dot(factor, measurement)
        informationMatrix = np.dot(factor, self.MeasurementMatrix)
        return informationVector, informationMatrix
    
    def GetMeasurementInfoFormAsInteger(self, realPos, scalingFactor, fast = False):
        iVecF, iMatF = self.GetMeasurementInformationForm(realPos, fast = fast)
        iVec8  = np.rint(iVecF * param.QUANTIZATION_FACTOR_8).astype(np.int64)
        iMat8  = np.rint(iMatF * param.QUANTIZATION_FACTOR_8).astype(np.int64)
        iVec16 = np.rint(iVecF * param.QUANTIZATION_FACTOR_16).astype(np.int64)
        iMat16 = np.rint(iMatF * param.QUANTIZATION_FACTOR_16).astype(np.int64)
        iVec24 = np.rint(iVecF * param.QUANTIZATION_FACTOR_24).astype(np.int64)
        iMat24 = np.rint(iMatF * param.QUANTIZATION_FACTOR_24).astype(np.int64)
        
        # Normalized
        iVecFN, iMatFN = iVecF * scalingFactor, iMatF * scalingFactor
        iVec8N  = np.rint(iVecFN * param.QUANTIZATION_FACTOR_8).astype(np.int64)
        iMat8N  = np.rint(iMatFN * param.QUANTIZATION_FACTOR_8).astype(np.int64)
        iVec16N = np.rint(iVecFN * param.QUANTIZATION_FACTOR_16).astype(np.int64)
        iMat16N = np.rint(iMatFN * param.QUANTIZATION_FACTOR_16).astype(np.int64)
        iVec24N = np.rint(iVecFN * param.QUANTIZATION_FACTOR_24).astype(np.int64)
        iMat24N = np.rint(iMatFN * param.QUANTIZATION_FACTOR_24).astype(np.int64)
        
        return iVecF, iMatF, iVec8, iMat8, iVec16, iMat16, iVec24, iMat24, iVecFN, iMatFN, iVec8N, iMat8N, iVec16N, iMat16N, iVec24N, iMat24N
    
    def GetAggregatedMeasurements(self, realPos, publicKey, fast = False, scalingFactor = None):
        # Obtain the scaling factor
        self.CountMeasurements(realPos)
        if scalingFactor is None:
            scalingFactor = float(param.RENORMALIZATION_FACTOR) / self.LastMeasurementCount if self.LastMeasurementCount > 0 else 0
    
        try:
            iVecF, iMatF, iVec8, iMat8, iVec16, iMat16, iVec24, iMat24, iVecFN, iMatFN, iVec8N, iMat8N, iVec16N, iMat16N, iVec24N, iMat24N = self.GetMeasurementInfoFormAsInteger(realPos, scalingFactor = scalingFactor, fast = fast)
        except ValueError:
            iVecF,  iMatF  = np.zeros((2,1), dtype=float), np.zeros((2,2), dtype=float)
            iVec8,  iMat8  = np.zeros((2,1), dtype=np.int64), np.zeros((2,2), dtype=np.int64)
            iVec16, iMat16, iVec24, iMat24 = np.copy(iVec8), np.copy(iMat8), np.copy(iVec8), np.copy(iMat8)
            iVecFN, iMatFN = np.copy(iVecF), np.copy(iMatF)
            iVec8N, iMat8N, iVec16N, iMat16N, iVec24N, iMat24N = np.copy(iVec8), np.copy(iMat8), np.copy(iVec8), np.copy(iMat8), np.copy(iVec8), np.copy(iMat8)
            
        # Encrypt 16 bit-quantized measurements
        if not param.DO_NOT_ENCRYPT:
            encVec  = EncryptedArray((2,1), publicKey, plaintextBuffer=iVec16)
            encMat  = EncryptedArray((2,2), publicKey, plaintextBuffer=iMat16)
            encVecN = EncryptedArray((2,1), publicKey, plaintextBuffer=iVec16N)
            encMatN = EncryptedArray((2,2), publicKey, plaintextBuffer=iMat16N)
        else:
            encVec, encMat, encVecN, encMatN = 0, 0, 0, 0
        
        for neighbor in self.MyNeighbours:
            nencVec, nencMat, nencVecN, nencMatN, niVecF, niMatF, niVec8, niMat8, niVec16, niMat16, niVec24, niMat24, niVecFN, niMatFN, niVec8N, niMat8N, niVec16N, niMat16N, niVec24N, niMat24N = neighbor.GetAggregatedMeasurements(realPos, publicKey, scalingFactor = scalingFactor, fast = fast)
            # Now aggregate the measurements
            iVecF += niVecF
            iMatF += niMatF
            iVec8 += niVec8
            iMat8 += niMat8
            iVec16 += niVec16
            iMat16 += niMat16
            iVec24 += niVec24
            iMat24 += niMat24
            # Normalized forms
            iVecFN += niVecFN
            iMatFN += niMatFN
            iVec8N += niVec8N
            iMat8N += niMat8N
            iVec16N += niVec16N
            iMat16N += niMat16N
            iVec24N += niVec24N
            iMat24N += niMat24N
            # Encrypted aggregation
            if not param.DO_NOT_ENCRYPT:
                encVec.Add(nencVec)
                encMat.Add(nencMat)
                encVecN.Add(nencVecN)
                encMatN.Add(nencMatN)
        
        return encVec, encMat, encVecN, encMatN, iVecF, iMatF, iVec8, iMat8, iVec16, iMat16, iVec24, iMat24, iVecFN, iMatFN, iVec8N, iMat8N, iVec16N, iMat16N, iVec24N, iMat24N
    
    def CountMeasurements(self, AgentPos):
        self.LastMeasurementCount = 1 if self.CanSeeAgent(AgentPos) else 0
        for neighbor in self.MyNeighbours:
            self.LastMeasurementCount += neighbor.CountMeasurements(AgentPos)
        return self.LastMeasurementCount
