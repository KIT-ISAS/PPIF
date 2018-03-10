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
        outMeas = np.rint(infMeas * param.QUANTIZATION_FACTOR).astype(np.int64)
        outCov = np.rint(infCov * param.QUANTIZATION_FACTOR).astype(np.int64)
        return outMeas, outCov, infMeas, infCov
    
    def GetAggregatedMeasurementInfoFormAsInteger(self, realPos, fast = False):
        try:
            outMeas, outCov, infMeas, infCov = self.GetMeasurementInfoFormAsInteger(realPos, fast = fast)
        except ValueError:
            outMeas, outCov = np.zeros((2,1), dtype=np.int64), np.zeros((2,2), dtype=np.int64)
            infMeas, infCov = np.zeros((2,1), dtype=float), np.zeros((2,2), dtype=float)
        
        for neighbor in self.MyNeighbours:
            neighIntMeas, neighIntCov, neighFloatMeas, neighFloatCov = neighbor.GetAggregatedMeasurementInfoFormAsInteger(realPos, fast=fast)
            # Now aggregate the measurements
            outMeas += neighIntMeas
            outCov += neighIntCov
            infMeas += neighFloatMeas
            infCov += neighFloatCov
        
        return outMeas, outCov, infMeas, infCov
    
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