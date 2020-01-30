

# Project: Lane Lines (For straight Roads) [Detailed Report](https://github.com/Sardhendu/self-driving-vehicle/blob/master/src/lane_lines/REPORT.md)

This project implements advanced computer vision techniques to detect lane lines. Below are some technique employed.

  * Camera Calibration, Distortion correction
  * Perspective Transform
  * Sliding Window technique to find lane region
  * Polynomial fit, Moving Average.
  * Pipeline to run the algorithm on videos.
  * Improvements on the basic tools to let to a more robust algorithm.

### 75 frames output video
![](https://github.com/Sardhendu/self-driving-vehicle/blob/master/src/lane_lines/images/laneline_output.gif) 

## Install and Run:
1. **Clone Repo**: git clone [git@github.com:Sardhendu/self-driving-vehicle.git]()
2. Install [*Pipenv*](https://pipenv-fork.readthedocs.io/en/latest/)
3. cd self-driving-vehicle
3. **Install Libraries**: pipenv sync
4. Download **test_images**, **test_videos**, **test_videos_output** from [Udacity's Repo](https://github.com/udacity/CarND-LaneLines-P1) and place it inside  **/self-driving-vehicle/src/lane_lines/data/** 
5. A simple way to start is to run the Jupyter Notebook [P1.ipynb](https://github.com/Sardhendu/self-driving-vehicle/blob/master/src/lane_lines/P1.ipynb)
6. Abstracted Methods can be found at [tools.py](https://github.com/Sardhendu/self-driving-vehicle/blob/master/src/lane_lines/tools.py)

Reference:
Udacity Self-driving-car-engineer : [Repository](https://github.com/udacity/CarND-LaneLines-P1)
