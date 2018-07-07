'''
Created on 09.03.2018

@author: Mikhail Aristov
'''
import multiprocessing as mp
import numpy as np
from simulation import Parameters as param, Agent, Sensor

# Override default simulation parameters (see the class definition for a full list)
param.TRY_MULTIPROCESSING = False
param.TOTAL_RUNS = 10
param.DO_NOT_ENCRYPT = True

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
    EstimateCount, MeasCount, seFloatNoRenorm, se8BitNoRenorm, se16BitNoRenorm, se24BitNoRenorm = 0,0,0,0,0,0
    seFloatNorm, se8BitNorm, se16BitNorm, se24BitNorm = 0,0,0,0
    for run in range(runs):
        if runs < 100 or run % (runs // 100) == 0:
            if PID is None:
                print("Running simulation #", run, "of", runs)
            else:
                print("Process #", PID, "is at", np.round(run/runs*100, 2), "%")
        
        # Spawn a new agent at the edge of the field with a constant velocity
        CurrentAgent = Agent(run, CentralHub)
        while CurrentAgent.Update():
            # Evaluate the estimate
            encryptionError, controlError = CurrentAgent.xEst8 - CurrentAgent.MyPos, CurrentAgent.xEstF - CurrentAgent.MyPos
            q16error, q24error = CurrentAgent.xEst16 - CurrentAgent.MyPos, CurrentAgent.xEst24 - CurrentAgent.MyPos
            
            controlErrorN, q8errorN = CurrentAgent.xEstFN - CurrentAgent.MyPos, CurrentAgent.xEst8N - CurrentAgent.MyPos
            q16errorN, q24errorN = CurrentAgent.xEst16N - CurrentAgent.MyPos, CurrentAgent.xEst24N - CurrentAgent.MyPos
            
            #floatErrNorm, q8ErrNorm, q16ErrNorm, q24ErrNorm
            
            seFloatNoRenorm += np.asscalar(controlError[0]*controlError[0] + controlError[1]*controlError[1])
            se8BitNoRenorm += np.asscalar(encryptionError[0]*encryptionError[0] + encryptionError[1]*encryptionError[1])
            se16BitNoRenorm += np.asscalar(q16error[0]*q16error[0] + q16error[1]*q16error[1])
            se24BitNoRenorm += np.asscalar(q24error[0]*q24error[0] + q24error[1]*q24error[1])
            
            seFloatNorm += np.asscalar(controlErrorN[0]*controlErrorN[0] + controlErrorN[1]*controlErrorN[1])
            se8BitNorm += np.asscalar(q8errorN[0]*q8errorN[0] + q8errorN[1]*q8errorN[1])
            se16BitNorm += np.asscalar(q16errorN[0]*q16errorN[0] + q16errorN[1]*q16errorN[1])
            se24BitNorm += np.asscalar(q24errorN[0]*q24errorN[0] + q24errorN[1]*q24errorN[1])
            
            EstimateCount += 1
            MeasCount += CentralHub.LastMeasurementCount
    return EstimateCount, MeasCount, seFloatNoRenorm, se8BitNoRenorm, se16BitNoRenorm, se24BitNoRenorm, seFloatNorm, se8BitNorm, se16BitNorm, se24BitNorm

# A wrapper for the simulation function that lets it run in parallel
def ParallelSimulaton(PID, runs, globalEstCount, globalMeasCount, globalMSENoNorm, globalMSENormalized):
    # Run the simulation
    EstimateCount, MeasCount, MSE_FloatNoRenorm, MSE_8BitNoRenorm, MSE_16BitNoRenorm, MSE_24BitNoRenorm, MSE_FloatNorm, MSE_8BitNorm, MSE_16BitNorm, MSE_24BitNorm = Simulation(runs, PID=PID)
    # Synchronize the output
    with globalEstCount.get_lock():
        globalEstCount.value += EstimateCount
    with globalMeasCount.get_lock():
        globalMeasCount.value += MeasCount
    with globalMSENoNorm.get_lock():
        globalMSENoNorm[0] += MSE_FloatNoRenorm
        globalMSENoNorm[1] += MSE_8BitNoRenorm
        globalMSENoNorm[2] += MSE_16BitNoRenorm
        globalMSENoNorm[3] += MSE_24BitNoRenorm
    with globalMSENormalized.get_lock():
        globalMSENormalized[0] += MSE_FloatNorm
        globalMSENormalized[1] += MSE_8BitNorm
        globalMSENormalized[2] += MSE_16BitNorm
        globalMSENormalized[3] += MSE_24BitNorm

if __name__ == '__main__':
    # Check for parallelization
    if param.TRY_MULTIPROCESSING and mp.cpu_count() >= 4:
        # Leave a couple cores for the system
        parallelProcessCount = mp.cpu_count() - 2
        # Split the total number of experiments equally between processes
        experimentsPerProcess = param.TOTAL_RUNS // parallelProcessCount
        # Prepare output structures for the processes
        syncEstimateCount, syncMeasurementCount, syncMSENoNorm, syncMSENormalized = mp.Value("d", 0), mp.Value("d", 0), mp.Array("d", 4), mp.Array("d", 4)
        # Create the process objects
        processes = [mp.Process(target=ParallelSimulaton, args=(p, experimentsPerProcess, syncEstimateCount, syncMeasurementCount, syncMSENoNorm, syncMSENormalized)) for p in range(parallelProcessCount)]
        
        # Run all processes in parallel
        [p.start() for p in processes]
        [p.join() for p in processes]
        
        # Finally, format the output
        EstimateCount = int(syncEstimateCount.value)
        AvgMeasCount = float(syncMeasurementCount.value) / EstimateCount
        MSE_FloatNoRenorm = float(syncMSENoNorm[0]) / EstimateCount
        MSE_8BitNoRenorm = float(syncMSENoNorm[1]) / EstimateCount
        MSE_16BitNoRenorm = float(syncMSENoNorm[2]) / EstimateCount
        MSE_24BitNoRenorm = float(syncMSENoNorm[3]) / EstimateCount
        MSE_FloatNorm = float(syncMSENormalized[0]) / EstimateCount
        MSE_8BitNorm = float(syncMSENormalized[1]) / EstimateCount
        MSE_16BitNorm = float(syncMSENormalized[2]) / EstimateCount
        MSE_24BitNorm = float(syncMSENormalized[3]) / EstimateCount
    else:
        EstimateCount, MeasCount, MSE_FloatNoRenorm, MSE_8BitNoRenorm, MSE_16BitNoRenorm, MSE_24BitNoRenorm, MSE_FloatNorm, MSE_8BitNorm, MSE_16BitNorm, MSE_24BitNorm = Simulation(param.TOTAL_RUNS)
        AvgMeasCount = MeasCount / EstimateCount
        MSE_FloatNoRenorm /= EstimateCount
        MSE_8BitNoRenorm /= EstimateCount
        MSE_16BitNoRenorm /= EstimateCount
        MSE_24BitNoRenorm /= EstimateCount
        MSE_FloatNorm /= EstimateCount
        MSE_8BitNorm /= EstimateCount
        MSE_16BitNorm /= EstimateCount
        MSE_24BitNorm /= EstimateCount
    
    # Print the results
    print("total number of estimations made:", EstimateCount, ", per experiment:", EstimateCount / param.TOTAL_RUNS)
    print("average aggregated measurements per estimation:", AvgMeasCount)
    print("============= REGULAR ESTIMATION =============")
    print("control (plaintext float) estimator MSE:", MSE_FloatNoRenorm)
    print("MSE  8bit:\t", MSE_8BitNoRenorm)
    print("MSE 16bit:\t", MSE_16BitNoRenorm)
    print("MSE 24bit:\t", MSE_24BitNoRenorm)
    print("precision loss due to 16 bit quantization (negative means smaller MSE with encryption!):\n", MSE_16BitNoRenorm - MSE_FloatNoRenorm)
    print("============ NORMALIZED ESTIMATION ===========")
    print("control (plaintext float) estimator MSE:", MSE_FloatNorm)
    print("MSE  8bit:\t", MSE_8BitNorm)
    print("MSE 16bit:\t", MSE_16BitNorm)
    print("MSE 24bit:\t", MSE_24BitNorm)