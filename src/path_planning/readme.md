
# Project 6 Path Planning:
----------------

In this project, we learn to drive a car in a highway. Path planning by itself is a difficult problem. It consumes data from all other segments of self-driving car such as **Localization (car in car is in real world)**, **Sensor fusion (tracking other vehicles)** and **Behaviour planning (finite state machines)**

## Installation
1. Install Term-3 Simulator [here](https://github.com/udacity/self-driving-car-sim/releases/tag/T3_v1.2)
2. Convert it into binary
   * chmod +x /location_of_unzip_file/term2_sim.app/Contents/MacOS/term2_sim_mac
3. Install the package with uWebHook
   * chmod u+x install-mac.sh
   * ./install-mac.sh
4. Build the Project
   * chmod u+x ./build.sh
   * ./build.sh
5. Start the simulator and run path planning



  For compilers to find we may need to set:

  export LDFLAGS="-L/usr/local/opt/openssl@1.1/lib"
  export CPPFLAGS="-I/usr/local/opt/openssl@1.1/include"


## Dataset:

1. **Highway Map Data (highway_map.csv):**
   This file contains 5 columns each indicating the way points in the map

   * The track contains 181 waypoints.
   * Each way points the center point between the two yellow center line in the road.
   * Column 1:      x position of waypoint in map coordinate.
   * Column 2:      y position of waypoint in map coordinate.
   * Column 3:      s position of waypoint in frenet coordinate.
   * Column 4 & 5:  d position vector of waypoint in frenet coordinate. (the d vector has a magnitude of 1)
              the d vector can be used to calculate the lane number

  ```
      * each lane is 4 meters wide
         l1  l2  l3   l4  l5  l6
        |   |   |   ||   |   |   |
        |   |   |   ||   |   |   |
        |-12| -8|-4 ||  4|  8| 12|
        |   |   |   ||   |   |   |
        |   |   |   ||   |   |   |
      * To be in the center of a particular lane say l6 in map coordinate. we simple do,
      * (l6x, l6y) = ((x, y) + (d1, s2)) * (8 + 2) , 8-> distance of l6 from center, 2->to reach the center of the lane

 ```


2. **Data from the Simulator:**

  * car_x:              x position of the car in the map coordinate frame (derived from **localizing** the car)
  * car_y:              y position of the car in the map coordinate frame (derived from **localizing** the car)
  * car_s:              s position (longitudinal displacement) of the car in the frenet coordinate frame
  * car_d:              d position (lateral displacement) of the car in the frenet coordinate frame
  * car_yaw:            car's yaw angle in the map coordinate frame
  * car_speed:          speed of the car in miles/hr
  * previous_path_x:    list of x positions in map frame of the polynomial path in the previous (t-1) state.
  * previous_path_x:    list of y positions in map frame the polynomial path in the previous (t-1) state
  * sensor_fusion:      list of information on each vehicle in the right lane.
    * Data format:       [ id, x, y, vx, vy, s, d]
        * id:             Vehicle id
        * x:              x position in map coordinate
        * y:              y position in map coordinate
        * vx:             velocity in x direction
        * vy:             velocity in y direction
        * s:              s value in frenet coordinate
        * d:              d value in frenet coordinate


## Files:
1. **prediction.cpp**: Initializes the prediction dictionary at each timestep consuming data from the sensors:
2. **vehicle.cpp**: Wraps the central logic to the path planning algorithm. Using prediction data and finite state process. It outputs the best trajectory for the vehicle.
3. **utils.h**: Helper function for transformations between spaces.
4. **cost.h**: Implements the penalty for every trajectory.
5. **main.cpp**: Create the web hook to connect to the simulator.

 
 
## Algorithm Components:
1. **Trajectory Generation**: The trajectory generation process is pretty straight forward where we use the **spline** library to generate piece-wise polynomial that the vehicle should traverse. The trajectory is generated for 50 points in future. A typical workflow looks like.
   * Given Trajectory = *p1, p2, p3, p4, p5, .......... p48, p49, p50*
   * Say at time step t the the simulator consumes p1, p2 and p3.
   * So the previously trajectory are returned by the simulator, such as: *p4, p5, p6, .......... p48, p49, p50*
   * So at time step t+1 we only shift the trajctory to *p1, p2, p3 ... p45, p46, p47* and generate 3 new points in the future that is *p48, p49, p50*
   * So the trajectory at timestep t+1 looks like *p1, p2, p3, p4, p5, .......... p48, p49, p50*

2. **Finite State System** Here we make decision about the next state of our vehicle. We predominantly use 5 state machine.
    
    **States**:
      * *Keep Lane (KL)*: States that the vehicle can traverse from KL state are *PLCR, PLCL and KL*.
      * *Prepare Lane Change Left (PLCL)* States that the vehicle can traverse from here are *LCL, KL, PLCL*
      * *Prepare Lane Change Right (PLCR)* States that the vehicle can traverse from here are *LCR, KL, PLCR*
      * *Lane Change Left (LCL)* States that the vehicle can traverse from here are *KL, LCL*
      * *Lane Change Right (LCR)* States that the vehicle can traverse from here are *KL, LCR*
    
    **Actions/State Kinematics** The action at the particular state are constraint by several factors, such as **the speed limit of the lane, nearby vehicles** and others. The two main actions we find are:
      * *speed/velocity*: This determines the velocity that the vehicle should be driving for any of the state.
      * *lane*: This determines the lane that the car should keep while at above state.
      
3. **Trajectory Cost** We generate trajectories for all the possible states. For example, if the car is at *Lane=1* and *state=KL*, then the car can take either of the three states *KL, PLCL, PLCR*. So we generate trajectories for each of the possible states and  attribute a cost to each trajectory given the constraint. The trajectory that has the minimum cost is chosen. We primarily focus on using three cost functions.

   *Insufficiency Cost* In short this cost function penalizes trajectories based in the average velocity of the car. The high the velocity the lower the cost. Intuitively, we want to be able to move as fast as we can following traffic rules and lane rules. We can always go at a speed of 0.1 m/s and reach our goal, but this would increase out time to reach the goal. Therefore, we penalize slower trajectory.
   
   *Lane Cost*: With this cost function we penalize lanes with slower traffic.
   
   *Lane Change Cost* When making lane changes we should consider other vehicle around us. Lane change decisions that could result in a collision or unsafe driving are penalized.
   
   Each cost function has its own significance. While Lane Change Cost and the Insufficiency cost, are weighted higher Lane cost is not very significant and hence its weighted lower.
   

#### Gotchas:
  * The entire process is run in the future timesteps. Why do this? this helps us avoid weird collisions and smooth our path. Say our predictions goes to t=50 steps ahead. So when the vehicle/simulator is at step t then we actually make predictions for only  
  * The car movement should be defined in frenet coordinate system.
  * When building polynomials or trajectories that involve math (which it will). Its a good idea to do the math in vehicle coordinate system. Note we deal with three coordinate systems.
    * **Map coordinate system** Vehicle position in real world xy axis
    * **Frenet coordinate system** Vehicle position in s and d
    * **Vehicle coordinate system** The path that the vehicle points is the x coordinate. All the sensor reading are taken in vehicle coordinate system.
    
    **How to create better smooth trajectory** 
     * Transform the localized xy points from map coordinate frame to vehicle coordinate frame (Remember Localization using particle filter).
     * Do your math of building out the polynomials or whatever.
     * Transform back the xy point to map coordinate frame using the same transformation.

#### Common isssues while installing:

   When you see this error:
   ld: cannot link directly with dylib/framework, your binary is not an allowed client of /usr/lib/libcrypto.dylib for architecture x86_64

   1. Check files in your Mac with similar names
   	ls -lh lib{crypto,ssl}*
   	actual file provided = -rwxr-xr-x  1 root  wheel    32K Apr  6 15:46 libcrypto.dylib
   2. Check what provided by openssl
       1. brew list openssl | grep lib
       2. Openssl provided file = /usr/local/Cellar/openssl@1.1/1.1.1g/lib/libcrypto.1.1.dylib
   3. The problem with new versions of MAC is that we don’t have the privilege t change anything in the lib folder, such as mv libssl.dylib libssl.dylib.bak error: mv: rename libssl.dylib to libssl.dylib.bak: Read-only file system
   4. mount /dev/disk1s5 on / (apfs, local, read-only, journaled). Says that the mount is in read-only model
   5. we remount in write mode
   6. So we remount in write mode
       1. Sudo mount -r -w /
  7. Now we rename the existing libssl.dylib and libcrypto.dylib
    -> sudo mv libssl.dylib libssl.dylib.bak
    -> sudo mv libcrypto.dylib libcrypto.dylib.bak
  8. Check if the rename was successful
    -> ls -lh lib{crypto,ssl}*
  9. Copy the libssl and libcrypto files from Openssl
    -> sudo cp /usr/local/Cellar/openssl@1.1/1.1.1g/lib/libssl.1.1.dylib ./
    -> sudo cp /usr/local/Cellar/openssl@1.1/1.1.1g/lib/libcrypto.1.1.dylib ./
  10. Create the link
    sudo ln -s libssl.1.1.dylib libssl.dylib
    sudo ln -s libcrypto.1.1.dylib libcrypto.dylib
