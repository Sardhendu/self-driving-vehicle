#include <map>
#include <vector>
#include <cmath>
#include "prediction.h"
#include "utils.h"


using std::vector;
using std::map;

void Prediction::setPredictions(
  vector<vector<double>> sensor_fusion_data,
  int prev_trajectory_size,
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
    single_vehicle.speed = sqrt(
      sensor_fusion_data[i][3]*sensor_fusion_data[i][3] +
      sensor_fusion_data[i][4]*sensor_fusion_data[i][4]
    );
    single_vehicle.state = state;
    single_vehicle.lane = getLane(sensor_fusion_data[i][6]);

    std::cout << "Prediction   " << " "
    << "id = " << single_vehicle.id << " "
    << "x = " << single_vehicle.x << " "
    << "y = " << single_vehicle.y << " "
    << "vx = " << single_vehicle.vx  << " "
    << "vy = " << single_vehicle.vy << " "
    << "s = " << single_vehicle.s << " "
    << "d = " << single_vehicle.d << " "
    << "speed = " << single_vehicle.speed << " "
    << "state = " <<single_vehicle.state << " "
    << "lane = " <<single_vehicle.lane << " ";

    // Estimate the Car s in future using its velocity in xy direction
    single_vehicle.s += ((double)prev_trajectory_size*SEC_TO_VISIT_NEXT_POINT*single_vehicle.speed);

    std::cout << "s_future = " << single_vehicle.s << "\n";

    prediction_trajectory.push_back(single_vehicle);
    predictions_dict[i] = prediction_trajectory;
  }
}



map<int, vector<Traffic>> Prediction::getPredictions(){
  // for (int i=0; i <predictions_dict.size(); i++){
  //
  // }
  return predictions_dict;
}


Traffic Prediction::getNearestVehicleAheadInLane(
  map<int, vector<Traffic>> traffic_ahead, int curr_lane
){
  std::cout << "\t[Traffic Ahead]" << "\n";
  Traffic nearest_vehicle_ahead;
  if (traffic_ahead.count(curr_lane) > 0){
    int idx;
    double min_s_value = 999999;
    vector<Traffic> traffic_ahead_in_lane = traffic_ahead[curr_lane];
    for (int i=0; i<traffic_ahead_in_lane.size(); i++){
        std::cout << "\t\tVehicle Ahead in lane = " << curr_lane << " " << traffic_ahead_in_lane[i].s << " " << traffic_ahead_in_lane[i].d << " " << traffic_ahead_in_lane[i].lane << " " << traffic_ahead_in_lane[i].speed << "\n";
        if (traffic_ahead_in_lane[i].s <= min_s_value){
          nearest_vehicle_ahead = traffic_ahead_in_lane[i];
          min_s_value = traffic_ahead_in_lane[i].s;
        }
      }

    }
    std::cout << "\t\t[NEAREST] Vehicle Ahead in lane = " << curr_lane << " "<< nearest_vehicle_ahead.s << " " << nearest_vehicle_ahead.d << " " << nearest_vehicle_ahead.lane << " " << nearest_vehicle_ahead.speed << "\n";

    return nearest_vehicle_ahead;
  }



Traffic Prediction::getNearestVehicleBehindInLane(
  map<int, vector<Traffic>> traffic_behind, int curr_lane
){
  std::cout << "\t[Traffic Behind]" << "\n";
  Traffic nearest_vehicle_behind;
  if (traffic_behind.count(curr_lane) > 0){
    int idx;
    double max_s_value = -999999;
    vector<Traffic> traffic_behind_in_lane = traffic_behind[curr_lane];
    for (int i=0; i<traffic_behind_in_lane.size(); i++){
        std::cout << "\t\tVehicle Behind in lane = " << curr_lane << " " << traffic_behind_in_lane[i].s << " " << traffic_behind_in_lane[i].d << " " << traffic_behind_in_lane[i].lane << " " << traffic_behind_in_lane[i].speed << "\n";
        if (traffic_behind_in_lane[i].s >= max_s_value){
          nearest_vehicle_behind = traffic_behind_in_lane[i];
          max_s_value = traffic_behind_in_lane[i].s;
        }
      }
    }

    std::cout << "\t\t[NEAREST] Vehicle Behind in lane = " << curr_lane << " " << nearest_vehicle_behind.s << " " << nearest_vehicle_behind.d << " " << nearest_vehicle_behind.lane<< " " << nearest_vehicle_behind.speed << "\n";

    return nearest_vehicle_behind;
  }


vector<map<int, vector<Traffic>>> Prediction::getTraffic(double car_s){
  map<int, vector<Traffic>> traffic_ahead;
  map<int, vector<Traffic>> traffic_behind;
  map<int, vector<Traffic>>::iterator it = predictions_dict.begin();
  while (it != predictions_dict.end()){
    Traffic curr_vehicle = it->second[0];
    if (curr_vehicle.lane >=0 && curr_vehicle.lane <=2){
      // std::cout << "curr_vehicle.lane === " << curr_vehicle.lane << " " << car_s << " " << curr_vehicle.s << "\n";
      if (car_s < curr_vehicle.s){
        traffic_ahead[curr_vehicle.lane].push_back(curr_vehicle);
      }
      else if (car_s >= curr_vehicle.s){
        traffic_behind[curr_vehicle.lane].push_back(curr_vehicle);
      }

    }
    it++;
  }

  return {traffic_ahead, traffic_behind};
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
