### Project: Particle Filter in C++

This project is aimed to build a particle filter in C++ to localize a vehicle in the real-worlds.


### Output Video sneak peek

![output-video](https://github.com/Sardhendu/self-driving-vehicle/blob/master/src/particle_filter/images/sneak_peak.gif)

### Installation and Local run:
--------------
```bash
cd src/particle_filter
docker build -t pfilter .
docker run -v ./particle_filter:/particle_filter -it pfilter bash
chmod u+x install-ubuntu.sh
./install-ubuntu.sh
g++ -std=c++11 run.cpp particle_filter.cpp -o run
./run
```


### Running with Udacity's simulator:
---------------
```bash
mkdir build
cd build
cmake ..
make
./particle_filter
```

### Dataset Files:
-------------------
Included are some dataset that can help run/debug the algorithm before testing it with the udacity simulator.

   - **observation**: This folder contains separate files for each timstep:
      - Each row in depicts the landmark px, py observed by the sensor data
   - **map_data.txt**: Contains landmark position in map/real-world coordinate frame.
   - **control_data.txt**: The vehicle control data (velocity and yaw_rate)
   - **gt_data**: The ground truth vehicle position (x, y, theta) for every timestep   
   - **prediction_gt_data**: The ourputs of the prediction for plots.

### Code files
--------------------

   - **particle_filter.cpp**: Contains the entire particle_filter algorithm code.
   - **parser.cpp**: Code to read/write data and some helper functions.
   - **run.cpp**: Helper code to run/debug the algorithm with simulator (used data from ./files/*).
   - **main.cpp**: The main functions that call the uWebSocket to communicate with the simulator.


### Goal:
--------------------
   - We have to localize/find the vehicle position in the map/real-world coordinate frame.

### What we need:
-------------------
   1. We need a way to map sensor data from vehicle frame to the map frame at each time step.
   2. We need lamdmark locations since localization takes place by using Landmarks as reference points in both the vehicle and the map frame.
   3. We need a way to match landmarks in vehicle frame to landmarks in map-frame.

### Idea:
------------------
The idea is pretty simple, think of generating 100's of point that represents particle in *map-frame*. Think of these particles as transformation vectors with (x,y=translation, theta=orientation) to transform observation from vehicle-frame to map-frame. Use these transformations/particles to transform landmarks location from *vehicle-frame* to *map-frame*. Provide high weights to the particle/transformations that matches the car-lankmark distance using real-world landmark data and transformed landmark data. Particle closest to the vehicle/car would have higher weights.    

### Frame:
----------------
    * Map Frame:
       - Particles
       - Landmarks position
    * Vehicle Frame:
       - Sensor data
          - Current vechicle position
          - Landmarks position

### Process:
----------------
1. **Initialization**:
    - We initialize particles say 100 of them in an approximate area using GPS coordinates of the vehicle
    - Points are sampled from gaussian distribution with mean = GPS location and a standard deviation of say 20-50 meters or something like that.

2. **Prediction Step / Move**:
    - Here we use the velocity and yaw_rate from sensors to move the car or find new states for particle using particle information from previous timestep.
    - In this step we simply predict the current car/particle location in vehicle-frame using sensor readings and previous state of particle.
    - Though velocity and theta are measured by vehicle, they dont have direct dependency on positions in the vehicle frame. As we are multiplying the velocity and yaw_rate to particles position and as long as we are consistent in our implementation the filter would learn the correct particle position.

3. **Update**:
    * *Transformation*:
       - For each particle we transform all landmark observations to map-frame.
    * *Data Association**: Here we associate each transformed landmark to their nearest landmark in map-frame. This is done by nearest neighbor method.
    * *Update*:
       - We have,
          - *Transformed landmarks*: Estimated landmark in map-frame
          - *landmarks*: The actual landmark position in map-frame
       - Think of the update step as calculating the likelihood of *Transformed landmarks* and *landmarks* for each particle and providing weights to each particle based on their likelihood value. This process ensure providing higher weights to particle near the vehicle.
       - The likelihood criteria used is the product of the probability density (computed using multivariate gaussian).


4. **Resample**:
    - Finally we sample the particles with repeatation based on their weights. We employ sampling-wheel that ensures particles with higher weights to be resampled many-many times.


### Prediction Plot
-------------------

![analysis_plot](https://github.com/Sardhendu/self-driving-vehicle/blob/master/src/particle_filter/images/gt_prediction_plot.png)
