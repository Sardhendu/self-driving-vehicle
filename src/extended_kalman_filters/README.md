# Extended Kalman Filter:


# What do the Files do:

1. run.cpp: Dummy main function that replicates the functionality of main.cpp to run the kalman filter algorithm without the uWebSocket requirement. This is mainly useful to debug the code and ensure the code works as expected.
2. parser.cpp, parser.h: Parses the txt file and serve the inputs to the run.cpp
3. kalman_filter.cpp: Wraps all the functionality of the kalman filter process.
4. FusionEKF: Wraps the code to fuse sensor input from both RADAR and LASER. Also, serves as the entry point to the entire functionality.
5. main.cpp: Parses data, calls FusionEKF and uses uWebSocket to serve the input and algorithm output to the Udacity Simulator.

# Install and make Dummy run:
1. docker build -t ekf .
2. docker run -v /Users/sardhendu/workspace/self-driving-vehicle/src/extended_kalman_filters:/extended_kalman_filters -it ekf bash
3. g++ -std=c++11 run.cpp parser.cpp kalman_filter.cpp fusion_ekf.cpp -o run


# Kalman Filters:
----------------
DataSet:
This project uses two different sensor data,  LASER (LIDAR) and RADAR.

**LIDAR:** This sensor data represents a point cloud of the environment around it. In an ideal case one would perform object detection, collect the point clond data from the LIDAR sensor data and determine the position of the object of interest. For this project the data is already preprocessed and the object position are already presented to us. The processed data is represented the **"cartesian coordinate" system

**RADAR:** This sensor data represents the polar coordinate system, where the inputs are teh object distance, its angle and the radial velocity. The radial velocity are produced using dopler affect by radar sensors. In an ideal world, we would preprocess the environmnet around us to find the object and collect RADAR information for the object.

**Kalman Filters:** In short Kalman Filters uses series of measurement to estimate a value, when the values is not directly observed or is affected by noise. The Kalman Filter can to approximated to a linear model where the states are linear combination of the prior and prediction states. The estimates and the uncertainities are learned using Gaussian distribution. It works is two steps,

   * Prediction Step:  The prediction state produces the estimates and uncertainity for teh current state.
        -> Estmates are the real-time position of the object.
        -> Uncertainity accounts for the process noise (The change in velocity over time of the object) and the uncertainity in the estimates learned by the covariance matrix.

   * Update Step: The update step uses the measurement of the sensors, compare it to our predictions and update the values.
        ->  The new estimates are calculated by weighting the predicted estimates and the measurement estimates based on uncertainity in the prediction and the uncertainity in the measurement. IF the uncertainity in teh prediction is higher the new estimates are more closer to the measurement estimate and vice-a-versa
        -> Uncertainity mostly accounts for the noise in the measurement reading.




![Ouptut-Plot](https://github.com/Sardhendu/self-driving-vehicle/blob/master/src/extended_kalman_filters/images/output_png.png)