

### Installation:
--------------
cd src/particle_filter
docker build -t pfilter .
docker run -v /Users/sardhendu/workspace/self-driving-vehicle/src/particle_filter:/particle_filter -it pfilter bash
chmod u+x install-ubuntu.sh
./install-ubuntu.sh
g++ -std=c++11 run.cpp particle_filter.cpp -o run
./run

### Running with Udacity's simulator:
---------------
   - mkdir build
   - cd build
   - cmake ..
   - make
   - ./particle_filter

About Data Files:

  1. observation: This folder contains separate files for each timstep:
      -> Each row in depicts the landmark px, py observed by the sensor data



Goal: We have to localize/find the vehicle position in the map/real-world coordinate frame.

What we need:
1. We need a way to map sensor data from vehicle frame to the map frame at each time step.
1. We need lamdmark locations since localization takes place by using Landmarks as reference points in both the vehicle and the map frame.
2. We need a way to match landmarks in vehicle frame to landmarks in map-frame.

### Idea:
The idea of the particle filter is to generate many-many points in **map-frame** and importance weight particles whose distance to each landmark in map-frame matches the distance of vehicle to landmarks in vehicle frame.

(OR Rather)

Think of particles as random transformation with translation and rotation values (x, y, theta). We use all the transformation to transform the landmark observation (fetched using sensors) from vehicle-frame to map-frame. Finally we provide higher weights to transformations that minimize the distance of landmark-car in vehicle-frame and landmark-car in map-frame

### Frame:
    * Map Frame:
       - Particles
       - Landmarks
    * Vehicle Frame:
       - Sensor data
          - Current vechicle position
          - Landmarks position

### Process:
1. **Initialization**:
    - We initialize particles say 100 of them in an approximate area using GPS coordinates of the vehicle
    - Points are sampled from gaussian distribution with mean = GPS location and a standard deviation of say 20-50 meters or something like that.

2. **Prediction Step**:
    - We assume each particle to be the car's location in the vehicle coordinate frame.
    - In this step we simply predict the current car/particle location in vehicle-frame using sensor readings.

3. **Update**:
    * *Transformation*:
       - For each particle we transform all landmark observations to map-frame.
    * *Update*:
       - We compute the new weights for each particle, which is the product of the probability density (computed using multivariate gaussian). This process ensure providing higher weights to particle near the vehicle.


4. **Resample**:
    - Finally we sample the particles with repeatation based on their weights. We employ sampling-wheel that ensures particles with higher weights to be resampled many-many times.


### Prediction Plot





