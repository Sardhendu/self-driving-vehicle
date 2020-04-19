
cd src/particle_filter
docker build -t pfilter .
docker run -v /Users/sardhendu/workspace/self-driving-vehicle/src/particle_filter:/particle_filter -it pfilter bash
chmod u+x install-ubuntu.sh
./install-ubuntu.sh
g++ -std=c++11 run.cpp particle_filter.cpp -o run
./run

Running with Udacity's simulator:
-----------

./clean.sh
./build.sh
./run.sh

mkdir build
cd build
cmake ..
make
./particle_filter

About Data Files:

  1. observation: This folder contains separate files for each timstep:
      -> Each row in depicts the landmark px, py observed by the sensor data
