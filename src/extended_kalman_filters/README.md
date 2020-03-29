
# What does the code structure say

1. main.cpp:
    -> Reads data from the .txt file, stores it into the MeasurementPackage object and passes to the FusionEKF.ProcessMeasurement method.
    -> Receives the predictions from FusionEKF
    -> Send the predictions and ground truth to the tools.RMSE method to get the RMSE values
    -> And uses several function within to make connection with the uWebSocket to the simulator. It provides the sensor input data, the prediction data and the position and velocity RMSE to the simulator.

2.



What do the Files do:

1. run.cpp: Dummy main function that replicates the functionality of main.cpp to run the kalman filter algorithm without the uWebSocket requirement. This is mainly useful to debug the code and ensure the code works as expected.
2. parser.cpp, parser.h: Parses the txt file and serve the inputs to the run.cpp
3. kalman_filter.cpp: Wraps all the functionality of the kalman filter process.
4. FusionEKF: Wraps the code to fuse sensor input from both RADAR and LASER. Also, servs as the entry point to the entire functionality.
5. main.cpp: Parses data, calls FusionEKF and uses uWebSocket to serve the input and algorithm output to the Udacity Simulator.

Install :
1. docker build -t ekf .
2. docker run -v /Users/sardhendu/workspace/self-driving-vehicle/src/extended_kalman_filters:/extended_kalman_filters -it ekf bash
3. g++ -std=c++11 run.cpp parser.cpp kalman_filter.cpp fusion_ekf.cpp -o run





![Orig-Preprocess-Images](https://github.com/Sardhendu/self-driving-vehicle/blob/master/src/behavioural_cloning/image/input_img.png)
