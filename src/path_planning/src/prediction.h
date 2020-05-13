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
  string state = "CS";
};

class Prediction {
public:
  map<int, vector<Traffic>> predictions_dict;

  Prediction() {};
  ~Prediction() {};

  void setPredictions(
    vector<vector<double>> sensor_fusion_data,
    string state
  );

  map<int, vector<Traffic>> getPredictions();
  Traffic getNearestVehicleAhead(double car_s, int car_lane);
  Traffic getNearestVehicleBehind(double car_s, int car_lane);
};

#endif  // PREDICTION_H
