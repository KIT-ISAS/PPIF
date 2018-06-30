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
    
    def GetMeasurementInfoFormAsInteger(self, realPos, fast = False):
        infMeas, infCov = self.GetMeasurementInformationForm(realPos, fast = fast)
        outMeas8bit = np.rint(infMeas * self.Qfactor8bit).astype(np.int64)
        outCov8bit = np.rint(infCov * self.Qfactor8bit).astype(np.int64)
        outMeas16bit = np.rint(infMeas * self.Qfactor16bit).astype(np.int64)
        outCov16bit = np.rint(infCov * self.Qfactor16bit).astype(np.int64)
        outMeas24bit = np.rint(infMeas * self.Qfactor24bit).astype(np.int64)
        outCov24bit = np.rint(infCov * self.Qfactor24bit).astype(np.int64)
        return outMeas8bit, outCov8bit, infMeas, infCov, outMeas16bit, outCov16bit, outMeas24bit, outCov24bit
    
    def GetAggregatedMeasurementInfoFormAsInteger(self, realPos, fast = False):
        try:
            outMeas8bit, outCov8bit, infMeas, infCov, outMeas16bit, outCov16bit, outMeas24bit, outCov24bit = self.GetMeasurementInfoFormAsInteger(realPos, fast = fast)
            MeasCount = 1
        except ValueError:
            outMeas8bit, outCov8bit = np.zeros((2,1), dtype=np.int64), np.zeros((2,2), dtype=np.int64)
            outMeas16bit, outCov16bit = np.zeros((2,1), dtype=np.int64), np.zeros((2,2), dtype=np.int64)
            outMeas24bit, outCov24bit = np.zeros((2,1), dtype=np.int64), np.zeros((2,2), dtype=np.int64)
            infMeas, infCov = np.zeros((2,1), dtype=float), np.zeros((2,2), dtype=float)
            MeasCount = 0
        
        for neighbor in self.MyNeighbours:
            extraMeasCount, neighInt8Meas, neighInt8Cov, neighFloatMeas, neighFloatCov, neighInt16Meas, neighInt16Cov, neighInt24Meas, neighInt24Cov = neighbor.GetAggregatedMeasurementInfoFormAsInteger(realPos, fast=fast)
            # Now aggregate the measurements
            MeasCount += extraMeasCount
            outMeas8bit += neighInt8Meas
            outCov8bit += neighInt8Cov
            outMeas16bit += neighInt16Meas
            outCov16bit += neighInt16Cov
            outMeas24bit += neighInt24Meas
            outCov24bit += neighInt24Cov
            infMeas += neighFloatMeas
            infCov += neighFloatCov
        
        return MeasCount, outMeas8bit, outCov8bit, infMeas, infCov, outMeas16bit, outCov16bit, outMeas24bit, outCov24bit
    
    def GetEncryptedMeasurement(self, realPos, requester, publicKey, networkLog = False):        
        # Get plaintext measurement as integer, but if it throws an error because it cannot detect the agent at this range,
        # simply returning all zeroes in this case doesn't affect the end result
        try:
            intMeas, intCov, floatMeas, floatCov = self.GetMeasurementInfoFormAsInteger(realPos, fast = True)
        except ValueError:
            intMeas, intCov = np.zeros((2,1), dtype=np.int64), np.zeros((2,2), dtype=np.int64)
            floatMeas, floatCov = np.zeros((2,1), dtype=float), np.zeros((2,2), dtype=float)
        # Encrypt every element with the supplied public key
        encMeas = EncryptedArray((2,1), publicKey, plaintextBuffer=intMeas)
        encCov = EncryptedArray((2,2), publicKey, plaintextBuffer=intCov)
        
        # Now query every neighbor for their measurements
        for neighbor in self.MyNeighbours:
            # Get the measurements from neighbors
            neighEncMeas, neighEncCov, neighIntMeas, neighIntCov, neighFloatMeas, neighFloatCov = neighbor.GetEncryptedMeasurement(realPos, self.Name, publicKey)
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