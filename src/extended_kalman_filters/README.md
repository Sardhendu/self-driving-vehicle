
# What does the code structure say

1. main.cpp:
    -> Reads data from the .txt file, stores it into the MeasurementPackage object and passes to the FusionEKF.ProcessMeasurement method.
    -> Receives the predictions from FusionEKF
    -> Send the predictions and ground truth to the tools.RMSE method to get the RMSE values
    -> And uses several function within to make connection with the uWebSocket to the simulator. It provides the sensor input data, the prediction data and the position and velocity RMSE to the simulator.

2.



Install :
1. docker build -t ekf .
2. docker run -v /Users/sardhendu/workspace/self-driving-vehicle/src/extended_kalman_filters:/extended_kalman_filters -it ekf bash
3. g++ -std=c++11 run.cpp parser.cpp kalman_filter.cpp fusion_ekf.cpp -o run
