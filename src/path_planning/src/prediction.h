#ifndef PREDICTION_H
#define PREDICTION_H

#include <map>
#include <string>
#include <vector>

using std::vector;
using std::string;
using std::map;

struct Traffic {
  int id;
  double x;
  double y;
  double vx;
  double vy;
  double s;
  double d;
  double v;
  int lane;
  double speed;
  string state = "CS";

Traffic(){
  id = -1;
  x = -1.0;
  y = -1.0;
  vx = -1.0;
  vy = -1.0;
  s = -1.0;
  d = -1.0;
  v = -1.0;
  lane = -1;
  speed = -1.0;
  string state = "NA";
};

};

class Prediction {
private:
  double SEC_TO_VISIT_NEXT_POINT = 0.02;
public:
  map<int, vector<Traffic>> predictions_dict;
  Prediction() {};
  ~Prediction() {};

  void setPredictions(
    vector<vector<double>> sensor_fusion_data,
    int prev_trajectory_size,
    string state
  );

  map<int, vector<Traffic>> getPredictions();
  vector<map<int, vector<Traffic>>> getTraffic(double car_s);
  Traffic getNearestVehicleAheadInLane(map<int, vector<Traffic>> traffic_ahead, int curr_lane);
  Traffic getNearestVehicleBehindInLane(map<int, vector<Traffic>> traffic_behind, int curr_lane);

};

#endif  // PREDICTION_H
