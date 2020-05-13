#ifndef VEHICLE_H
#define VEHICLE_H
#include <vector>
#include <utility>
#include "prediction.h"

using std::vector;

class Vehicle {

public:
  vector<double> max_lane_velocity = {49.5/2.24, 49.5/2.24, 49.5/2.24}; // In meters/sec
  double car_v_prev = 0;
  double car_v = 0.2;
  double car_a = 0.2;
  double car_a_max = 0.224;               // maximum acceleration permitted
  // double increment_velocity = 0.2;
  double sec_to_visit_next_point = 0.02; // how many seconds should the car take to visit the next point (px, py at t+1, when the car is at t)
  int target_trajectory_distance = 30; // meters that the car should look ahead for trajectory generation
  int target_trajectory_points = 50; // num of future points to generate in the trajectory
  int buffer_distance = 20; //Assuming we keep 10 m distance from any car ahead of us

  double car_x;
  double car_y;
  double car_s;
  double car_d;
  double car_yaw;
  double car_speed;
  int car_lane;
  vector<double> waypoints_s_map;
  vector<double> waypoints_x_map;
  vector<double> waypoints_y_map;
  vector<vector<double>> sensor_fusion_data;

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
    vector<double> map_waypoints_y,
    vector<vector<double>> sensor_fusion_data
  );

  vector<vector<double>> generateTrajectory(
    vector<double> previous_path_x,
    vector<double> previous_path_y
    // auto previous_path_x,
    // auto previous_path_y
  );

  vector<vector<double>> keepLaneTrajectory(
    double curr_v,      // current velocity
    vector<double> previous_path_x,
    vector<double> previous_path_y
  );

  // vector<vector<double>> moveSmoothlyInOneLane();
  double get_kinematics();

  Prediction prediction_obj;
};

#endif // VEHICLE_H
