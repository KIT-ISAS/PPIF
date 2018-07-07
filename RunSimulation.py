'''
Created on 09.03.2018

@author: Mikhail Aristov
'''
import multiprocessing as mp
import numpy as np
from simulation import Parameters as param, Agent, Sensor

# Override default simulation parameters (see the class definition for a full list)
#param.DO_NOT_ENCRYPT = True

# Place 25 sensors across the area
sensorCount, placementStep = param.SENSOR_GRID_ROW_COUNT * param.SENSOR_GRID_ROW_COUNT, param.AREA_SIDE_LENGTH / (param.SENSOR_GRID_ROW_COUNT - 1)
SensorArray = []

# Place the sensors
for i in range(sensorCount):
    # Calculate the grid position
    row, col = i // param.SENSOR_GRID_ROW_COUNT, i % param.SENSOR_GRID_ROW_COUNT
    # Initialize a new sensor and add it to the grid
    newSensor = Sensor(i, placementStep * col, placementStep * row)
    SensorArray.append(newSensor)

# Link the sensors to each other in a three-tier hierarchy
CentralHub, MinorHub1, MinorHub2, MinorHub3, MinorHub4 = SensorArray[12], SensorArray[6], SensorArray[8], SensorArray[16], SensorArray[18]
CentralHub.MyNeighbours = [MinorHub1, MinorHub2, MinorHub3, MinorHub4]
MinorHub1.MyNeighbours = [SensorArray[0], SensorArray[1], SensorArray[5], SensorArray[7], SensorArray[10]]
MinorHub2.MyNeighbours = [SensorArray[2], SensorArray[3], SensorArray[4], SensorArray[9], SensorArray[13]]
MinorHub3.MyNeighbours = [SensorArray[11], SensorArray[15], SensorArray[20], SensorArray[21], SensorArray[22]]
MinorHub4.MyNeighbours = [SensorArray[14], SensorArray[17], SensorArray[19], SensorArray[23], SensorArray[24]]
# TODO: Make the above part generic...

# Runs a simulation a specified number of times
def Simulation(runs, PID = None):
    EstimateCount, MeasCount, errEnc, errF, err8, err16, err24, errEncN, errFN, err8N, err16N, err24N = 0,0,0,0,0,0,0,0,0,0,0,0
    for run in range(runs):
        if runs < 100 or run % (runs // 100) == 0:
            if PID is None:
                print("Running simulation #", run, "of", runs)
            else:
                print("Process #", PID, "is at", np.round(run/runs*100, 2), "%")
        
        # Spawn a new agent at the edge of the field with a constant velocity
        CurrentAgent = Agent(run, CentralHub)
        while CurrentAgent.Update():
            # Evaluate the estimates
            if not param.DO_NOT_ENCRYPT:
                errEnc  += np.linalg.norm(CurrentAgent.xEstEnc  - CurrentAgent.MyPos) ** 2
                errEncN += np.linalg.norm(CurrentAgent.xEstEncN - CurrentAgent.MyPos) ** 2
            
            errF   += np.linalg.norm(CurrentAgent.xEstF   - CurrentAgent.MyPos) ** 2
            err8   += np.linalg.norm(CurrentAgent.xEst8   - CurrentAgent.MyPos) ** 2
            err16  += np.linalg.norm(CurrentAgent.xEst16  - CurrentAgent.MyPos) ** 2
            err24  += np.linalg.norm(CurrentAgent.xEst24  - CurrentAgent.MyPos) ** 2
            errFN  += np.linalg.norm(CurrentAgent.xEstFN  - CurrentAgent.MyPos) ** 2
            err8N  += np.linalg.norm(CurrentAgent.xEst8N  - CurrentAgent.MyPos) ** 2
            err16N += np.linalg.norm(CurrentAgent.xEst16N - CurrentAgent.MyPos) ** 2
            err24N += np.linalg.norm(CurrentAgent.xEst24N - CurrentAgent.MyPos) ** 2
            
            EstimateCount += 1
            MeasCount += CentralHub.LastMeasurementCount
    return EstimateCount, MeasCount, errEnc, errF, err8, err16, err24, errEncN, errFN, err8N, err16N, err24N

# A wrapper for the simulation function that lets it run in parallel
def ParallelSimulaton(PID, runs, globalEstCount, globalMeasCount, globalSquaredError, globalSquaredErrorNormalized):
    # Run the simulation
    EstimateCount, MeasCount, errEnc, errF, err8, err16, err24, errEncN, errFN, err8N, err16N, err24N = Simulation(runs, PID=PID)
    # Synchronize the output
    with globalEstCount.get_lock():
        globalEstCount.value += EstimateCount
    with globalMeasCount.get_lock():
        globalMeasCount.value += MeasCount
    with globalSquaredError.get_lock():
        globalSquaredError[0] += errF
        globalSquaredError[1] += err8
        globalSquaredError[2] += err16
        globalSquaredError[3] += err24
        globalSquaredError[4] += errEnc
    with globalSquaredErrorNormalized.get_lock():
        globalSquaredErrorNormalized[0] += errFN
        globalSquaredErrorNormalized[1] += err8N
        globalSquaredErrorNormalized[2] += err16N
        globalSquaredErrorNormalized[3] += err24N
        globalSquaredErrorNormalized[4] += errEncN

if __name__ == '__main__':
    # Check for parallelization
    if param.TRY_MULTIPROCESSING and mp.cpu_count() >= 4:
        # Leave a couple cores for the system
        parallelProcessCount = mp.cpu_count() - 2
        # Split the total number of experiments equally between processes
        experimentsPerProcess = param.TOTAL_RUNS // parallelProcessCount
        # Prepare output structures for the processes
        syncEstimateCount, syncMeasurementCount, syncError, syncErrorNormalized = mp.Value("d", 0), mp.Value("d", 0), mp.Array("d", 5), mp.Array("d", 5)
        # Create the process objects
        processes = [mp.Process(target=ParallelSimulaton, args=(p, experimentsPerProcess, syncEstimateCount, syncMeasurementCount, syncError, syncErrorNormalized)) for p in range(parallelProcessCount)]
        
        # Run all processes in parallel
        [p.start() for p in processes]
        [p.join() for p in processes]
        
        # Finally, format the output
        EstimateCount = int(syncEstimateCount.value)
        AvgMeasCount = float(syncMeasurementCount.value) / EstimateCount
        
        errF    = np.sqrt(float(syncError[0]) / EstimateCount)
        err8    = np.sqrt(float(syncError[1]) / EstimateCount)
        err16   = np.sqrt(float(syncError[2]) / EstimateCount)
        err24   = np.sqrt(float(syncError[3]) / EstimateCount)
        errEnc  = np.sqrt(float(syncError[4]) / EstimateCount)
        
        errFN   = np.sqrt(float(syncErrorNormalized[0]) / EstimateCount)
        err8N   = np.sqrt(float(syncErrorNormalized[1]) / EstimateCount)
        err16N  = np.sqrt(float(syncErrorNormalized[2]) / EstimateCount)
        err24N  = np.sqrt(float(syncErrorNormalized[3]) / EstimateCount)
        errEncN = np.sqrt(float(syncErrorNormalized[4]) / EstimateCount)
    else:
        EstimateCount, MeasCount, errEnc, errF, err8, err16, err24, errEncN, errFN, err8N, err16N, err24N = Simulation(param.TOTAL_RUNS)
        AvgMeasCount = MeasCount / EstimateCount
        
        errEnc  = np.sqrt(errEnc  / EstimateCount)
        errF    = np.sqrt(errF    / EstimateCount)
        err8    = np.sqrt(err8    / EstimateCount)
        err16   = np.sqrt(err16   / EstimateCount)
        err24   = np.sqrt(err24   / EstimateCount)
        
        errEncN = np.sqrt(errEncN / EstimateCount)
        errFN   = np.sqrt(errFN   / EstimateCount)
        err8N   = np.sqrt(err8N   / EstimateCount)
        err16N  = np.sqrt(err16N  / EstimateCount)
        err24N  = np.sqrt(err24N  / EstimateCount)
    
    # Print the results
    print("total number of estimations made:", EstimateCount, ", per experiment:", EstimateCount / param.TOTAL_RUNS)
    print("average aggregated measurements per estimation:", AvgMeasCount)
    print("============= REGULAR ESTIMATION =============")
    print("RMSE     plaintext float:\t", errF)
    print("RMSE  8 bit quantization:\t", err8)
    print("RMSE 16 bit quantization:\t", err16)
    if not param.DO_NOT_ENCRYPT:
        print("RMSE 16 bit  (encrypted):\t", errEnc)
    print("RMSE 24 bit quantization:\t", err24)
    print("precision loss due to 16 bit quantization (negative means smaller MSE with encryption!):\n", err16 - errF)
    print("============ NORMALIZED ESTIMATION ===========")
    print("RMSE     plaintext float:\t", errFN)
    print("RMSE  8 bit quantization:\t", err8N)
    print("RMSE 16 bit quantization:\t", err16N)
    if not param.DO_NOT_ENCRYPT:
        print("RMSE 16 bit  (encrypted):\t", errEncN)
    print("RMSE 24 bit quantization:\t", err24N)