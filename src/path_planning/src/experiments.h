#ifndef EXPERIMENTS_H
#define EXPERIMENTS_H

#include <vector>
#include "helpers.h"
#include "spline.h"
using namespace std;

vector<vector<double>> move_smoothly_in_the_lane(
  double car_s,
  double car_d,
  vector<double> map_waypoints_s,
  vector<double> map_waypoints_x,
  vector<double> map_waypoints_y
){
  vector<double> next_x_vals;
  vector<double> next_y_vals;

  double dist_inc = 0.3; // 0.5 indicates ~ 50 miles/hr
  for (int i=0; i<50; i++){
      // Make sure that we stay in the same lane
      // Calculate the next position (one step ahead in the future)
      double next_s = car_s + (dist_inc*(i+1)); // Move ahead in the same direction
      double next_d = car_d; // stay on the sane lane

      // Get vehicle map coordinate based on waypoints
      vector<double> map_xy = getXY(
        next_s,
        next_d,
        map_waypoints_s,
        map_waypoints_x,
        map_waypoints_y
      );

      // To stay on the same lane
      double next_x = map_xy[0];
      double next_y = map_xy[1];


      next_x_vals.push_back(next_x);
      next_y_vals.push_back(next_y);
  }

  // we dont need all points to fit the polynomial only 5 points would do
  vector<double> X = {
    next_x_vals[0], next_x_vals[9], next_x_vals[19], next_x_vals[29], next_x_vals[39], next_x_vals[49]
  };
  vector<double> Y = {
    next_y_vals[0], next_y_vals[9], next_y_vals[19], next_y_vals[29], next_y_vals[39], next_y_vals[49]
  };
  cout << "Fiting piecewise polynomial with spline " << "\n";
  tk::spline s;
  s.set_points(X,Y);

  vector<double> smooth_next_y_vals;
  for (int k =0; k<next_x_vals.size(); k++){
    double smooth_y = s(next_x_vals[k]);
    cout << next_x_vals[k] << " " << next_y_vals[k] << " " <<smooth_y << "\n";
    smooth_next_y_vals.push_back(smooth_y);

  }

  return {next_x_vals, smooth_next_y_vals};
}

#endif

// Todos;
/*
  Funciton 1: Write a function to get the lane given any car position
  Function 2: Transform map coordinate to vehicle coordinate
  Function 3: Transform vehicle coordinate to map coordinate.
  Funciton 4: Calculate the speed magnitude of any car given velocity vector. (In Frenet)
  Function 5: Calculate the path (s, d) of other cars atleast for few timestems. (In Frenet)
  Function 6: Function that check potential collision in all feasible lane
*/
