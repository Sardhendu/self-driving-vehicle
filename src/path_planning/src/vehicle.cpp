#include <iostream>
#include <vector>
#include "vehicle.h"
#include "utils.h"
#include "spline.h"
#include "experiments.h"

using std::vector;


void Vehicle::setVehicle(
  double x,
  double y,
  double s,
  double d,
  double yaw,
  double speed,
  vector<double> map_waypoints_s,
  vector<double> map_waypoints_x,
  vector<double> map_waypoints_y
){
  // std::cout << "set s: " << s << "\n";
  // std::cout << "set d: " << d << "\n";
  x_map = x;
  y_map = y;
  s_frn = s;
  d_frn = d;
  car_yaw = yaw;
  car_speed = speed;
  waypoints_s_map = map_waypoints_s;
  waypoints_x_map = map_waypoints_x;
  waypoints_y_map = map_waypoints_y;
  // std::cout << "set s_frn: " << s_frn << "\n";
  // std::cout << "set d_frn: " << d_frn << "\n";
}


vector<vector<double>> Vehicle::generateTrajectory(
  vector<double> previous_path_x,
  vector<double> previous_path_y
){
  std::cout << "set s_frn: " << s_frn << "\n";
  std::cout << "set d_frn: " << d_frn << "\n";
  int lane_num = getLane(d_frn);
  double curr_velocity = max_lane_velocity[lane_num];

  std::cout << "curr_velocity = " << curr_velocity << "\n";

  vector<vector<double>> trajectoryXY = moveSmoothlyAvoidingColisionInOneLane(
    x_map,
    y_map,
    s_frn,
    d_frn,
    car_yaw,
    waypoints_s_map,
    waypoints_x_map,
    waypoints_y_map,
    previous_path_x,
    previous_path_y,
    curr_velocity,
    target_trajectory_distance,
    target_trajectory_points,
    sec_to_visit_next_point
  );

  // std::cout << "s: " << s << "\n";
  // vector<vector<double>> trajectoryXY_ = moveSmoothlyInOneLane(
  //   s_frn,
  //   d_frn,
  //   waypoints_s_map,
  //   waypoints_x_map,
  //   waypoints_y_map
  // );


  return trajectoryXY;
}
