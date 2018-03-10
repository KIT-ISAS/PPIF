'''
Created on 09.03.2018

@author: Mikhail Aristov
'''
import multiprocessing as mp
import numpy as np
from simulation import Parameters as param, Agent, Sensor

# Override default simulation parameters (see the class definition for a full list)
param.TOTAL_RUNS = 100
param.TRY_MULTIPROCESSING = False

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
    EstimateCount, EncryptedEstimateCovariance, ControlEstimateCovariance = 0, np.zeros((2,2), dtype=float), np.zeros((2,2), dtype=float)
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
            encryptionError, controlError = CurrentAgent.stateEstimate - CurrentAgent.MyPos, CurrentAgent.controlEstimate - CurrentAgent.MyPos
            EncryptedEstimateCovariance += np.dot(encryptionError, np.transpose(encryptionError))
            ControlEstimateCovariance += np.dot(controlError, np.transpose(controlError))
            EstimateCount += 1
    return EstimateCount, EncryptedEstimateCovariance, ControlEstimateCovariance

# A wrapper for the simulation function that lets it run in parallel
def ParallelSimulaton(PID, runs, globalEstCount, globalEncCov, globalControlCov):
    # Run the simulation
    EstimateCount, EncryptedEstimateCovariance, ControlEstimateCovariance = Simulation(runs, PID=PID)
    # Synchronize the output
    with globalEstCount.get_lock():
        globalEstCount.value += EstimateCount
    with globalEncCov.get_lock():
        EncryptedEstimateCovariance = np.reshape(EncryptedEstimateCovariance, 4)
        for i in range(4):
            globalEncCov[i] += np.asscalar(EncryptedEstimateCovariance[i])
    with globalControlCov.get_lock():
        ControlEstimateCovariance = np.reshape(ControlEstimateCovariance, 4)
        for i in range(4):
            globalControlCov[i] += np.asscalar(ControlEstimateCovariance[i])

if __name__ == '__main__':
    # Check for parallelization
    if param.TRY_MULTIPROCESSING and mp.cpu_count() >= 4:
        # Leave a couple cores for the system
        parallelProcessCount = mp.cpu_count() - 2
        # Split the total number of experiments equally between processes
        experimentsPerProcess = param.TOTAL_RUNS // parallelProcessCount
        # Prepare output structures for the processes
        syncEstimateCount, syncEncryptedEstimateCovariance, syncControlEstimateCovariance = mp.Value("d", 0), mp.Array("d", 4), mp.Array("d", 4)
        # Create the process objects
        processes = [mp.Process(target=ParallelSimulaton, args=(p, experimentsPerProcess, syncEstimateCount, syncEncryptedEstimateCovariance, syncControlEstimateCovariance)) for p in range(parallelProcessCount)]
        
        # Run all processes in parallel
        [p.start() for p in processes]
        [p.join() for p in processes]
        
        # Finally, format the output
        EstimateCount = int(syncEstimateCount.value)
        EncryptedEstimateCovariance = np.ndarray((2,2), buffer=np.array(syncEncryptedEstimateCovariance[:], dtype=float)) / (EstimateCount - 1)
        ControlEstimateCovariance = np.ndarray((2,2), buffer=np.array(syncControlEstimateCovariance[:], dtype=float)) / (EstimateCount - 1)
    else:
        EstimateCount, EncryptedEstimateCovariance, ControlEstimateCovariance = Simulation(param.TOTAL_RUNS)
        EncryptedEstimateCovariance /= EstimateCount - 1
        ControlEstimateCovariance /= EstimateCount - 1
    
    # Print the results
    print("total number of estimations made:", EstimateCount, ", per experiment:", EstimateCount / param.TOTAL_RUNS)
    print("encrypted estimator covariance:\n", EncryptedEstimateCovariance)
    print("control (plaintext float) estimator covariance:\n", ControlEstimateCovariance)
    print("precision loss due to encryption (negative means smaller covariance with encryption!):\n", EncryptedEstimateCovariance - ControlEstimateCovariance)