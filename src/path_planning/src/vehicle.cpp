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
  std::cout << "set s: " << s << "\n";
  std::cout << "set d: " << d << "\n";
  x_map = x;
  y_map = y;
  s_frn = s;
  d_frn = d;
  car_yaw = yaw;
  car_speed = speed;
  waypoints_s_map = map_waypoints_s;
  waypoints_x_map = map_waypoints_x;
  waypoints_y_map = map_waypoints_y;
  std::cout << "set s_frn: " << s_frn << "\n";
  std::cout << "set d_frn: " << d_frn << "\n";
}


vector<vector<double>> Vehicle::generateTrajectory(
  vector<double> previous_path_x,
  vector<double> previous_path_y
){
  std::cout << "set s_frn: " << s_frn << "\n";
  std::cout << "set d_frn: " << d_frn << "\n";
  // std::cout << "s: " << s << "\n";
  vector<vector<double>> trajectoryXY = moveSmoothlyInOneLane(
    s_frn,
    d_frn,
    waypoints_s_map,
    waypoints_x_map,
    waypoints_y_map
  );
  return trajectoryXY;
}


// vector<vector<double>> Vehicle::moveSmoothlyInOneLane(){
//   vector<double> next_x_vals;
//   vector<double> next_y_vals;
//
//   double dist_inc = 0.3; // 0.5 indicates ~ 50 miles/hr
//   for (int i=0; i<50; i++){
//       // Make sure that we stay in the same lane
//       // Calculate the next position (one step ahead in the future)
//       double next_s = s + (dist_inc*(i+1)); // Move ahead in the same direction
//       double next_d = d; // stay on the sane lane
//
//       // Get vehicle map coordinate based on waypoints
//       vector<double> map_xy = getXY(
//         next_s,
//         next_d,
//         map_waypoints_s,
//         map_waypoints_x,
//         map_waypoints_y
//       );
//
//       // To stay on the same lane
//       double next_x = map_xy[0];
//       double next_y = map_xy[1];
//
//
//       next_x_vals.push_back(next_x);
//       next_y_vals.push_back(next_y);
//   }
//
//   // we dont need all points to fit the polynomial only 5 points would do
//   vector<double> X = {
//     next_x_vals[0], next_x_vals[9], next_x_vals[19], next_x_vals[29], next_x_vals[39], next_x_vals[49]
//   };
//   vector<double> Y = {
//     next_y_vals[0], next_y_vals[9], next_y_vals[19], next_y_vals[29], next_y_vals[39], next_y_vals[49]
//   };
//   // std::cout << "Fiting piecewise polynomial with spline " << "\n";
//   tk::spline spl;
//   spl.set_points(X,Y);
//
//   vector<double> smooth_next_y_vals;
//   for (int k =0; k<next_x_vals.size(); k++){
//     double smooth_y = spl(next_x_vals[k]);
//     // std::cout << next_x_vals[k] << " " << next_y_vals[k] << " " <<smooth_y << "\n";
//     smooth_next_y_vals.push_back(smooth_y);
//
//   }
//
//   return {next_x_vals, smooth_next_y_vals};
// }
