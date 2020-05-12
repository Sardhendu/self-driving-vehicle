#include <map>
#include <vector>

#include "prediction.h"
#include "utils.h"



using std::vector;
using std::map;

void Prediction::setPredctions(
  vector<vector<double>> sensor_fusion_data,
  string state = "CS"
){
  predictions_dict.clear();
  for (int i=0; i<sensor_fusion_data.size(); i++){
    Traffic single_vehicle;
    vector<Traffic> prediction_trajectory;
    single_vehicle.id = sensor_fusion_data[i][0];
    single_vehicle.x = sensor_fusion_data[i][1];
    single_vehicle.y = sensor_fusion_data[i][2];
    single_vehicle.vx = sensor_fusion_data[i][3];
    single_vehicle.vy = sensor_fusion_data[i][4];
    single_vehicle.s = sensor_fusion_data[i][5];
    single_vehicle.d = sensor_fusion_data[i][6];
    single_vehicle.state = state;
    single_vehicle.lane = getLane(sensor_fusion_data[i][6]);
    prediction_trajectory.push_back(single_vehicle);
    predictions_dict[i] = prediction_trajectory;

    std::cout << "Prediction \t\t "
    << single_vehicle.id << " "
    << single_vehicle.x << " "
    << single_vehicle.y << " "
    << single_vehicle.vx  << " "
    << single_vehicle.vy << " "
    << single_vehicle.s << " "
    << single_vehicle.d << " "
    << single_vehicle.state << " "
    << single_vehicle.lane << "\n";
  }

}



map<int, vector<Traffic>> Prediction::getPredictions(){
  // for (int i=0; i <predictions_dict.size(); i++){
  //
  // }
  return predictions_dict;
}


Traffic Prediction::getNearestVehicleAhead(
  double car_s,
  int car_lane
){

  Traffic nearest_vehicle;
  double nearest_vehicle_s;
  int nearest_vehicle_id;
  int counter_ = 0;

  map<int, vector<Traffic>>::iterator it = predictions_dict.begin();
  while (it != predictions_dict.end()){
    Traffic curr_vehicle = it->second[0];
    // std::cout << "curr_vehicle_id \t\t " << it -> first
    // << curr_vehicle.id << " "
    // << curr_vehicle.x << " "
    // << curr_vehicle.y << " "
    // << curr_vehicle.vx  << " "
    // << curr_vehicle.vy << " "
    // << curr_vehicle.s << " "
    // << curr_vehicle.d << " "
    // << curr_vehicle.state << " "
    // << curr_vehicle.lane << "\n";
    // Check if the current car_s if < curr_vehicle.s
    if (car_s < curr_vehicle.s){
      if (counter_ == 0){
        nearest_vehicle = it -> second[0];
        nearest_vehicle_s = curr_vehicle.s;
        nearest_vehicle_id = it -> first;
        // std::cout << "------------------> Nearest_vehicle \n ";
      }
      else{
        if (
          curr_vehicle.s < nearest_vehicle_s &&
          car_lane == curr_vehicle.lane
        ){
            nearest_vehicle_s = curr_vehicle.s;
            nearest_vehicle = curr_vehicle;
            nearest_vehicle_id = it -> first;
            // std::cout << "------------------> Nearest_vehicle \n ";
          }
      }

      counter_ += 1;
    }

    it++;
  }
  return nearest_vehicle;
}


Traffic Prediction::getNearestVehicleBehind(
  double car_s,
  int car_lane
){

  Traffic nearest_vehicle;
  double nearest_vehicle_s;
  int nearest_vehicle_id;
  int counter_ = 0;

  map<int, vector<Traffic>>::iterator it = predictions_dict.begin();
  while (it != predictions_dict.end()){
    Traffic curr_vehicle = it->second[0];
    std::cout << "curr_vehicle_id \t\t " << it -> first
    << curr_vehicle.id << " "
    << curr_vehicle.x << " "
    << curr_vehicle.y << " "
    << curr_vehicle.vx  << " "
    << curr_vehicle.vy << " "
    << curr_vehicle.s << " "
    << curr_vehicle.d << " "
    << curr_vehicle.state << " "
    << curr_vehicle.lane << "\n";
    // Check if the current car_s if < curr_vehicle.s
    if (car_s > curr_vehicle.s){
      if (counter_ == 0){
        nearest_vehicle = it -> second[0];
        nearest_vehicle_s = curr_vehicle.s;
        nearest_vehicle_id = it -> first;
        std::cout << "------------------> Nearest_vehicle \n ";
      }
      else{
        if (
          curr_vehicle.s > nearest_vehicle_s &&
          car_lane == curr_vehicle.lane
        ){
            nearest_vehicle_s = curr_vehicle.s;
            nearest_vehicle = curr_vehicle;
            nearest_vehicle_id = it -> first;
            std::cout << "------------------> Nearest_vehicle \n ";
          }
      }

      counter_ += 1;
    }

    it++;
  }
  return nearest_vehicle;
}

/*
TODOS:

1. write a funciton to get the attributes of the nearest vehicle in the lane forward and backward. Use this to generate new vehicle kinematics
 -> prediction:
    1. get_nearest_vehicle_ahead
    2. get_nearest_vehicle_behind

 -> Vehicle:
    1. get_kinematics: Generates kinematics for each trjectory points in the futures using information from vehile ahead and behind.

*/
