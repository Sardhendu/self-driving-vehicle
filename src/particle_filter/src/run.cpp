#include <math.h>
#include <iostream>
#include <string>
#include <ctime>
#include <iomanip>
#include "json.hpp"
#include "map.h"
#include "parser.h"
#include "particle_filter.h"

// for convenience
using nlohmann::json;
using std::string;
using std::vector;

int main(){
  // Set up parameters here
  double delta_t = 0.1;  // Time elapsed between measurements [sec]
  double sensor_range = 50;  // Sensor range [m]

  // GPS measurement uncertainty [x [m], y [m], theta [rad]]
  double sigma_pos [3] = {0.3, 0.3, 0.01};
  // Landmark measurement uncertainty [x [m], y [m]]
  double sigma_landmark [2] = {0.3, 0.3};

  // Landmark Map coordinates
  Map map;
  if (!read_map_data("../data/map_data.txt", map)) {
    std::cout << "Error: Could not open map file" << std::endl;
    return -1;
  }
  cout << "Size of Map::landmark_list = " << map.landmark_list.size() << "\n";

  // Vehicle Control Data
  vector<control> control_meas;
  if (!read_control_data("../data/control_data.txt", control_meas)) {
    std::cout << "Error: Could not open map file" << std::endl;
    return -1;
  }
  cout << "Total count of control measurements = " << control_meas.size() << "\n";


  // Ground Truth Data
  vector<ground_truth> gt;
  if (!read_gt_data("../data/gt_data.txt", gt)) {
    std::cout << "Error: Could not open map file" << std::endl;
    return -1;
  }
  cout << "Total count of ground-truth measurements = " << gt.size() << "\n";


  // Read the Landmark data at each timesteps
  int num_time_steps = 2;  //control_meas.size();
  ParticleFilter pf;
  for (int t; t<num_time_steps; t++){
    cout << "Running timestep ....................... " << t << "\n";
    ostringstream file;
		file << "../data/observation/observations_" << setfill('0') << setw(6) << t+1 << ".txt";

    cout << "\tReading filename = " << "../data/observation/observations_" << setfill('0') << setw(6) << t+1 << ".txt" << "\n";
		vector<landmark> observations;
		if (!read_landmark_data(file.str(), observations)) {
			cout << "Error: Could not open observation file " << t+1 << endl;
			return -1;
		}
    cout << "\tsensed landmark count = " << observations.size() << "\n";


    // ----------------------------------------------------------------------
    // Initialize
    // ----------------------------------------------------------------------
    if (!pf.initialized()){
      pf.init(gt[t].x, gt[t].y, gt[t].theta, sigma_pos);
      pf.print_particle_attributes(2);
      // cout << "Particles Initialized: Total Count = " << pf.particles.size() << "\n";
    }
    else {
      pf.predict(delta_t, control_meas[t-1].velocity, control_meas[t-1].yaw_rate, sigma_pos);
      pf.print_particle_attributes(2);
      // pf.dataAssociation(observations, map.landmark_list);
      pf.updateWeights(sensor_range, sigma_landmark, observations, map);
    }
  }
}
