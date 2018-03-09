'''
Created on 09.03.2018

@author: Mikhail Aristov
'''
from numpy import zeros, dot, transpose
from simulation import Parameters as param, Agent, Sensor

# Override default simulation parameters (see the class definition for a full list)
param.TOTAL_RUNS = 100

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
# TODO: Make generic...

# Start the simulation
EstimateCount, EncryptedEstimateCovariance, ControlEstimateCovariance = 0, zeros((2,2), dtype=float), zeros((2,2), dtype=float)
for run in range(param.TOTAL_RUNS):
    if param.TOTAL_RUNS < 100 or run % (param.TOTAL_RUNS // 100) == 0:
        print("Running simulation #", run, "of", param.TOTAL_RUNS)
    
    # Spawn a new agent at the edge of the field with a constant velocity
    CurrentAgent = Agent(run, CentralHub)
    while CurrentAgent.Update():
        # Evaluate the estimate
        encryptionError, controlError = CurrentAgent.stateEstimate - CurrentAgent.MyPos, CurrentAgent.controlEstimate - CurrentAgent.MyPos
        EncryptedEstimateCovariance += dot(encryptionError, transpose(encryptionError))
        ControlEstimateCovariance += dot(controlError, transpose(controlError))
        EstimateCount += 1

# Finally, normalize estimator covariance
EncryptedEstimateCovariance /= EstimateCount - 1
ControlEstimateCovariance /= EstimateCount - 1
print("total number of estimations made:", EstimateCount, ", per experiment:", EstimateCount / param.TOTAL_RUNS)
print("encrypted estimator covariance:\n", EncryptedEstimateCovariance)
print("control (plaintext float) estimator covariance:\n", ControlEstimateCovariance)
print("precision loss due to encryption (negative means smaller covariance with encryption!):\n", EncryptedEstimateCovariance - ControlEstimateCovariance)