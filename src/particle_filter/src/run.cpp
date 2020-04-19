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
  // Write to the output Files
  ofstream out_gt_prediction_file;
  outFile_lidar.open("../data/gt_prediction.txt");

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
  int num_time_steps = 100; //control_meas.size();
  ParticleFilter pf;
  for (int t; t<num_time_steps; t++){
    // cout << "Running timestep ....................... " << t << "\n";
    ostringstream file;
		file << "../data/observation/observations_" << setfill('0') << setw(6) << t+1 << ".txt";

    // cout << "\tReading filename = " << "../data/observation/observations_" << setfill('0') << setw(6) << t+1 << ".txt" << "\n";
		vector<landmark> observations;
		if (!read_landmark_data(file.str(), observations)) {
			cout << "Error: Could not open observation file " << t+1 << endl;
			return -1;
		}
    // cout << "\tsensed landmark count = " << observations.size() << "\n";


    // ----------------------------------------------------------------------
    // Initialize
    // ----------------------------------------------------------------------
    if (!pf.initialized()){
      pf.init(gt[t].x, gt[t].y, gt[t].theta, sigma_pos);
      pf.print_particle_attributes(2);
      // cout << "Particles Initialized: Total Count = " << pf.particles.size() << "\n";
    }
    else {
      pf.prediction(delta_t, sigma_pos, control_meas[t-1].velocity, control_meas[t-1].yawrate);
      // pf.print_particle_attributes(2);
    }

    // ----------------------------------------------------------------------
    // Update weights and Resample
    // ----------------------------------------------------------------------
    pf.updateWeights(sensor_range, sigma_landmark, observations, map);
    pf.resample();


    // ----------------------------------------------------------------------
    // Model Analysis
    // ----------------------------------------------------------------------
    vector<Particle> particles = pf.particles;
    int num_particles = particles.size();
    double highest_weight = -1.0;
    Particle best_particle;
    double weight_sum = 0.0;
    for (int i = 0; i < num_particles; ++i) {
      if (particles[i].weight > highest_weight) {
        highest_weight = particles[i].weight;
        best_particle = particles[i];
      }

      weight_sum += particles[i].weight;
    }

    if (t == 0){
      out_gt_prediction_file << "gt_x, gt_y, gt_theta, pr_id, pr_x, pr_ym, pr_theta, pr_weight" << "\n";
    }
    else{
      out_gt_prediction_file << gt[t].x << "," << gt[t].y << "," << gt[t].theta << ","
                             << best_particle.id << "," best_particle.x, << "," << best_particle.y << "," << best_particle.y << ","
                             << highest_weight << "\n";
    }

    cout << "\ntime step =  ............................ " << t << "\n";
    cout << "\thighest and average weights = " << highest_weight << " " << weight_sum/num_particles << "\n";
    cout << "\tbest particle = " << best_particle.id << " " << best_particle.x << " " << best_particle.y << " " << best_particle.theta << "\n";
    cout << "\tground truth = " << gt[t].x << " " << gt[t].y << " " << gt[t].theta << "\n";


  }

  out_gt_prediction_file.close();

  return 0;
}
