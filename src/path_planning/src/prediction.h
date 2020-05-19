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
  map<int, vector<Traffic>> getTrafficAhead();
  Traffic getNearestVehicleAhead(double car_s, int car_lane);
  Traffic getNearestVehicleBehind(double car_s, int car_lane);
};

#endif  // PREDICTION_H
