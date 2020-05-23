


Installation:

1. Download the simulator from [here](https://github.com/udacity/self-driving-car-sim/releases)
2. Convert it into binary
   * chmod +x /location_of_unzip_file/term2_sim.app/Contents/MacOS/term2_sim_mac
3. 


## Dataset from the simulator:
The simulator is same as that of the Behaviour planner simulator. In this case, instead of using a deep-learning model to predict the steering angle we use the sensor data to get the steering angle.
 
  * CTE: Cross Track Error: How far is the vehicle from the reference trajectory.
  * Speed: Vehicle speed in the moving direction
  * Angle: Orientation of the vehicle 
  * Steer value: the value to steer 
