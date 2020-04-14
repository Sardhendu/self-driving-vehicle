#include <math.h>
#include <iostream>
#include <string>
#include <ctime>
#include <iomanip>
#include "json.hpp"
#include "map.h"
#include "parser.h"
// #include "particle_filter.h"

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

  Map map;
  if (!read_map_data("../data/map_data.txt", map)) {
    std::cout << "Error: Could not open map file" << std::endl;
    return -1;
  }
  cout << "Size of Map::landmark_list = " << map.landmark_list.size() << "\n";

  vector<control> control_meas;
  if (!read_control_data("../data/control_data.txt", control_meas)) {
    std::cout << "Error: Could not open map file" << std::endl;
    return -1;
  }
  cout << "Total count of control measurements = " << control_meas.size() << "\n";

  vector<ground_truth> gt;
  if (!read_gt_data("../data/gt_data.txt", gt)) {
    std::cout << "Error: Could not open map file" << std::endl;
    return -1;
  }
  cout << "Total count of ground-truth measurements = " << gt.size() << "\n";

}

// string hasData(string s) {
//   auto found_null = s.find("null");
//   auto b1 = s.find_first_of("[");
//   auto b2 = s.find_first_of("]");
//   if (found_null != string::npos) {
//     return "";
//   } else if (b1 != string::npos && b2 != string::npos) {
//     return s.substr(b1, b2 - b1 + 1);
//   }
//   return "";
// }
