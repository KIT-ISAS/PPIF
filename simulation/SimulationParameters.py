'''
Created on 09.03.2018

@author: Mikhail Aristov
'''
from numpy import pi

class SimParameters(object):
    '''
    A convenience class for bundling all simulation settings.
    '''
    
    # Whether parallel processing should be attempted
    TRY_MULTIPROCESSING = True
    
    # Whether to actually encrypt communications (or to just evaluate the quantization, for performance reasons)
    DO_NOT_ENCRYPT = False
    
    # How many agent runs in total the simulation should include
    TOTAL_RUNS = 10000
    
    # How many agent runs in total the simulation should include
    RENORMALIZATION_FACTOR = 10

    # The size of the square area where the simulation occurs (along one side)
    AREA_SIDE_LENGTH = 100.0 # m
    
    # Fractional precision of pre-encryption quantization
    QUANTIZATION_FACTOR_8  = 2**8
    QUANTIZATION_FACTOR_16 = 2**16
    QUANTIZATION_FACTOR_24 = 2**24
    
    # The length of the encryption key (for test purposes only, no real security guarantee!)
    CRYPTO_KEY_LENGTH = 64 # bits
    
    # The number of sensors per row of the (square) sensor grid
    SENSOR_GRID_ROW_COUNT = 5 # DO NOT OVERRIDE WITHOUT UPDATING RunSimulation.py!
    
    # The standard deviation of the sensors' angle measurement
    SENSOR_AZIMUTH_SIGMA = pi/36 # radians = 5 deg
    
    # The standard deviation of the sensors' distance measurement
    SENSOR_DISTANCE_SIGMA = 2.0 # m
    
    # Maximum distance at which sensors can detect the agent
    SENSOR_DETECTION_RANGE = 50.0 # m
    
    # The standard deviation for sampling agents' initial velocity (X and Y components are sampled independently)
    AGENT_VELOCITY_SIGMA = 5.0 # m/s