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

  for (int i=0; i<sensor_fusion_data.size(); i++){
    Traffic single_vehicle;
    single_vehicle.id = i;
    single_vehicle.x = sensor_fusion_data[i][0];
    single_vehicle.y = sensor_fusion_data[i][1];
    single_vehicle.vx = sensor_fusion_data[i][2];
    single_vehicle.vy = sensor_fusion_data[i][3];
    single_vehicle.s = sensor_fusion_data[i][4];
    single_vehicle.d = sensor_fusion_data[i][5];
    single_vehicle.state = state;
    single_vehicle.lane = getLane(sensor_fusion_data[i][5]);
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
