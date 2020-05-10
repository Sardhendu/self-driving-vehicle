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
  int lane;
  string state = "CS";
};

class Prediction {
public:
  vector<Traffic> prediction_trajectory;
  map<int, vector<Traffic>> predictions_dict;

  Prediction() {};
  ~Prediction() {};

  void setPredctions(
    vector<vector<double>> sensor_fusion_data,
    string state
  );

  map<int, vector<Traffic>> getPredictions();
};

#endif  // PREDICTION_H
