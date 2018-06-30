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
    Qfactor8bit = 2**8
    Qfactor16bit = 2**16
    Qfactor24bit = 2**24

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
        infMeas, infCov = self.GetMeasurementInformationForm(realPos, fast = fast)
        outMeas8bit = np.rint(infMeas * self.Qfactor8bit).astype(np.int64)
        outCov8bit = np.rint(infCov * self.Qfactor8bit).astype(np.int64)
        outMeas16bit = np.rint(infMeas * self.Qfactor16bit).astype(np.int64)
        outCov16bit = np.rint(infCov * self.Qfactor16bit).astype(np.int64)
        outMeas24bit = np.rint(infMeas * self.Qfactor24bit).astype(np.int64)
        outCov24bit = np.rint(infCov * self.Qfactor24bit).astype(np.int64)
        
        # Normalized
        infMeasNorm, infCovNorm = infMeas * scalingFactor, infCov * scalingFactor
        outMeas8bitNorm = np.rint(infMeasNorm * self.Qfactor8bit).astype(np.int64)
        outCov8bitNorm = np.rint(infCovNorm * self.Qfactor8bit).astype(np.int64)
        outMeas16bitNorm = np.rint(infMeasNorm * self.Qfactor16bit).astype(np.int64)
        outCov16bitNorm = np.rint(infCovNorm * self.Qfactor16bit).astype(np.int64)
        outMeas24bitNorm = np.rint(infMeasNorm * self.Qfactor24bit).astype(np.int64)
        outCov24bitNorm = np.rint(infCovNorm * self.Qfactor24bit).astype(np.int64)
        
        return outMeas8bit, outCov8bit, infMeas, infCov, outMeas16bit, outCov16bit, outMeas24bit, outCov24bit, infMeasNorm, infCovNorm, outMeas8bitNorm, outCov8bitNorm, outMeas16bitNorm, outCov16bitNorm, outMeas24bitNorm, outCov24bitNorm
    
    def GetAggregatedMeasurementInfoFormAsInteger(self, realPos, fast = False, scalingFactor = None):
        # Obtain the scaling factor
        self.CountMeasurements(realPos)
        if scalingFactor is None:
            scalingFactor = float(param.RENORMALIZATION_FACTOR) / self.LastMeasurementCount if self.LastMeasurementCount > 0 else 0
    
        try:
            outMeas8bit, outCov8bit, infMeas, infCov, outMeas16bit, outCov16bit, outMeas24bit, outCov24bit, infMeasN, infCovN, outMeas8bitN, outCov8bitN, outMeas16bitN, outCov16bitN, outMeas24bitN, outCov24bitN = self.GetMeasurementInfoFormAsInteger(realPos, scalingFactor = scalingFactor, fast = fast)
        except ValueError:
            outMeas8bit, outCov8bit = np.zeros((2,1), dtype=np.int64), np.zeros((2,2), dtype=np.int64)
            outMeas16bit, outCov16bit = np.zeros((2,1), dtype=np.int64), np.zeros((2,2), dtype=np.int64)
            outMeas24bit, outCov24bit = np.zeros((2,1), dtype=np.int64), np.zeros((2,2), dtype=np.int64)
            infMeas, infCov = np.zeros((2,1), dtype=float), np.zeros((2,2), dtype=float)
            outMeas8bitN, outCov8bitN = np.zeros((2,1), dtype=np.int64), np.zeros((2,2), dtype=np.int64)
            outMeas16bitN, outCov16bitN = np.zeros((2,1), dtype=np.int64), np.zeros((2,2), dtype=np.int64)
            outMeas24bitN, outCov24bitN = np.zeros((2,1), dtype=np.int64), np.zeros((2,2), dtype=np.int64)
            infMeasN, infCovN = np.zeros((2,1), dtype=float), np.zeros((2,2), dtype=float)
        
        for neighbor in self.MyNeighbours:
            neighInt8Meas, neighInt8Cov, neighFloatMeas, neighFloatCov, neighInt16Meas, neighInt16Cov, neighInt24Meas, neighInt24Cov, neighFloatMeasN, neighFloatCovN, neighInt8MeasN, neighInt8CovN, neighInt16MeasN, neighInt16CovN, neighInt24MeasN, neighInt24CovN = neighbor.GetAggregatedMeasurementInfoFormAsInteger(realPos, scalingFactor = scalingFactor, fast=fast)
            # Now aggregate the measurements
            outMeas8bit += neighInt8Meas
            outCov8bit += neighInt8Cov
            outMeas16bit += neighInt16Meas
            outCov16bit += neighInt16Cov
            outMeas24bit += neighInt24Meas
            outCov24bit += neighInt24Cov
            infMeas += neighFloatMeas
            infCov += neighFloatCov
            outMeas8bitN += neighInt8MeasN
            outCov8bitN += neighInt8CovN
            outMeas16bitN += neighInt16MeasN
            outCov16bitN += neighInt16CovN
            outMeas24bitN += neighInt24MeasN
            outCov24bitN += neighInt24CovN
            infMeasN += neighFloatMeasN
            infCovN += neighFloatCovN
        
        return outMeas8bit, outCov8bit, infMeas, infCov, outMeas16bit, outCov16bit, outMeas24bit, outCov24bit, infMeasN, infCovN, outMeas8bitN, outCov8bitN, outMeas16bitN, outCov16bitN, outMeas24bitN, outCov24bitN
    
    def GetEncryptedMeasurement(self, realPos, requester, publicKey, networkLog = False, scalingFactor = None):  
        # Get plaintext measurement as integer, but if it throws an error because it cannot detect the agent at this range,
        # simply returning all zeroes in this case doesn't affect the end result
        try:
            intMeas, intCov, floatMeas, floatCov = self.GetMeasurementInfoFormAsInteger(realPos, scalingFactor, fast = True)
        except ValueError:
            intMeas, intCov = np.zeros((2,1), dtype=np.int64), np.zeros((2,2), dtype=np.int64)
            floatMeas, floatCov = np.zeros((2,1), dtype=float), np.zeros((2,2), dtype=float)
        # Encrypt every element with the supplied public key
        encMeas = EncryptedArray((2,1), publicKey, plaintextBuffer=intMeas)
        encCov = EncryptedArray((2,2), publicKey, plaintextBuffer=intCov)
        
        # Now query every neighbor for their measurements
        for neighbor in self.MyNeighbours:
            # Get the measurements from neighbors
            neighEncMeas, neighEncCov, neighIntMeas, neighIntCov, neighFloatMeas, neighFloatCov = neighbor.GetEncryptedMeasurement(realPos, self.Name, publicKey, scalingFactor = scalingFactor)
            # Now aggregate the measurements
            encMeas.Add(neighEncMeas)
            encCov.Add(neighEncCov)
            intMeas += neighIntMeas
            intCov += neighIntCov
            floatMeas += neighFloatMeas
            floatCov += neighFloatCov
        # Return the aggregated results to requester
        if networkLog:
            print("NETWORK LOG:", self.Name, "->", requester, ":", [encMeas.DATA, encCov.DATA])
        return encMeas, encCov, intMeas, intCov, floatMeas, floatCov
    
    def CountMeasurements(self, AgentPos):
        self.LastMeasurementCount = 1 if self.CanSeeAgent(AgentPos) else 0
        for neighbor in self.MyNeighbours:
            self.LastMeasurementCount += neighbor.CountMeasurements(AgentPos)
        return self.LastMeasurementCount