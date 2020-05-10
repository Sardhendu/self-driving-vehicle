#ifndef VEHICLE_H
#define VEHICLE_H
#include <vector>
#include <utility>

using std::vector;

class Vehicle {

public:

  double x_map;
  double y_map;
  double s_frn;
  double d_frn;
  double car_yaw;
  double car_speed;
  vector<double> waypoints_s_map;
  vector<double> waypoints_x_map;
  vector<double> waypoints_y_map;

  Vehicle() {};
  ~Vehicle() {};

  void setVehicle(
    double x,
    double y,
    double s,
    double d,
    double yaw,
    double speed,
    vector<double> map_waypoints_s,
    vector<double> map_waypoints_x,
    vector<double> map_waypoints_y
  );

  vector<vector<double>> generateTrajectory(
    vector<double> previous_path_x,
    vector<double> previous_path_y
    // auto previous_path_x,
    // auto previous_path_y
  );

  // vector<vector<double>> moveSmoothlyInOneLane();
};

#endif // VEHICLE_H
