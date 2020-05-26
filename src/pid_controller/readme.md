# Project 8: PID controller
-----------

A PID controller is a process/system that output the steering angle that a car should take while following a trajectory. In the project **behaviour_planning** we used a deep-neural network to generate the steering angle to make the vehicle move in a track. 


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
  * Steer value: The steering angle (-1, 1)
  
## PID Controller:
PID controller stands for Proportional Integral and differential. 

* **Proportional (P-controller)**: The proportional part of the algorithm states that the steering angle is proportional to the CTE. A P controller makes the vehicle oscillate near the trsjectory line. Below is a plot showcasing the oscillating behaviour of a P-controller.

    * Below is a plot for P-controller with/without drift using params: tau_p=0.3
    
![P-Controller Oscillating Behaviour](https://github.com/Sardhendu/self-driving-vehicle/blob/master/src/pid_controller/images/p_controller.png)


* **Proportional-Differential (PD-controller)**:  The differential part of a PD controller removes the oscillating behaviour of a P controller. The idea is the put a differential penalty to the P-controller for larger increment and decrement in the value. This removes the oscillating behaviour but the PD-controlled takes more time to converge to the trejectory. However when converged it remains closer to the trajectory.

    * Below is a plot for PD-controller with/without drift using params: tau_p=0.3, tau_d=3.0

![PD-Controller Oscillating with/without drift](https://github.com/Sardhendu/self-driving-vehicle/blob/master/src/pid_controller/images/pd_controller.png)


* **Proportional-differential-Integral (PID-Controller)**: Our vehicle can be faced with drifts because of several reasons such as manufacturing defect, irregularity in road. Drift can cause the PD controller to not converge. As humans, we often intuitively understand drift and steer harder in the opposite direction when encountered with drift, however the controller may not have the same intuition. Increasing the coefficients of a PD controller can bring the car closer to the trajectory, but this would introduce random jerks and oscillation. In a PID controlled we add bias to our controller steering taking into account the total error CTE for a longer duration. So we add penalty to the steering output with the integral of CTE.

  * Below is a plot for PID-controller with params: tau_p=0.3, tau_d=3.0, tau_i=0.009


![PID-Controller](https://github.com/Sardhendu/self-driving-vehicle/blob/master/src/pid_controller/images/pid_controller.png)

## Tuning PID coefficients:
 * The parameters **tau_p** and **tau_i** and **tau_d** can be decided empirically with trial and error. While many different variations can work, it is recommended to empirically find the initial parameters and then run algorithms like **Twiddle**, **Stocastic Gradient Descent** or other optimization method to find the optimal values. In our experiments we use Twiddle to find the optimal param values.   

