#ifndef EXPERIMENTS_H
#define EXPERIMENTS_H
#include <iostream>
#include <vector>
#include <math.h>
#include "utils.h"
#include "spline.h"


inline vector<vector<double>> moveSmoothlyInOneLane(
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
  // std::cout << "Fiting piecewise polynomial with spline " << "\n";
  tk::spline s;
  s.set_points(X,Y);

  vector<double> smooth_next_y_vals;
  for (int k =0; k<next_x_vals.size(); k++){
    double smooth_y = s(next_x_vals[k]);
    std::cout << next_x_vals[k] << " " << next_y_vals[k] << " " <<smooth_y << "\n";
    smooth_next_y_vals.push_back(smooth_y);

  }

  return {next_x_vals, smooth_next_y_vals};
}


inline vector<vector<double>> moveSmoothlyAvoidingColisionInOneLane(
  double car_x_map,
  double car_y_map,
  double car_s,
  double car_d,
  double car_yaw,
  vector<double> map_waypoints_s,
  vector<double> map_waypoints_x,
  vector<double> map_waypoints_y,
  vector<double> previous_path_x,
  vector<double> previous_path_y,

  double curr_v,      // current velocity
  int target_trajectory_distance,
  int target_trajectory_points,
  double sec_to_visit_next_point
){
  /*
  Idea: Here we use last t-m points from the previous path and use them to generate paths for t+n timesteps.
  Using the old path actually makes a better smooth trajectory.

  Step 3:
     1. We have fit the sline with 5 anchor/reference points from t-2, t-1, t+30m, t+60m, t+90m
     2. We have a polynomial function
     Whats next:
     1. We need a target_distance (how far do we want to predict) -> say 30m
     2. We need to sample n x_points from 0->target_distance spaces in such a
        way that our car goes in the desired velocity at that timestep. (Now this velocity can change in future timestep)
        if n =
     4. Using the x_points we predcit the y_points uisng the spline fitted polynomial
     5. The data previous_path_xy returned from simulator are actually the paths that we generated at
        step t-1 that were not used by the simulator to drive the car. So why not we add those

  */
  std::cout << "\t x_map = " << car_x_map << " y_map = " << car_y_map << " car_yaw = " << car_yaw << " s = " << car_s << " d = " << car_d << "\n";
  std::cout << "\tlen(previous_path_x) = " << previous_path_x.size() << "\n";
  std::cout << "\tcurrently velocity = " << curr_v << "\n";
  std::cout << "\ttarget_trajectory_distance = " << target_trajectory_distance << "\n";
  std::cout << "\tsec_to_visit_next_point = " << sec_to_visit_next_point << "\n";


  vector<double> anchor_points_x;
  vector<double> anchor_points_y;

  double ref_x_map = car_x_map;
  double ref_y_map = car_y_map;
  double ref_yaw = deg2rad(car_yaw);

  int prev_path_size = previous_path_x.size();

  // ----------------------------------------------------------------------
  // Step 1: Find Anchor points to fit polynomial using spline
  // ----------------------------------------------------------------------
  if (prev_path_size < 2){
    /*
      When we dont have any data on previous path we simple use the current
      car yaw to generate car position for t-m timesteps
    */
    double prev_x_map = car_x_map - cos(car_yaw);
    double prev_y_map = car_y_map - sin(car_yaw);

    anchor_points_x.push_back(prev_x_map);
    anchor_points_x.push_back(car_x_map);

    anchor_points_y.push_back(prev_y_map);
    anchor_points_y.push_back(car_y_map);

  }
  else{
    ref_x_map = previous_path_x[prev_path_size-1];
    ref_y_map = previous_path_y[prev_path_size-1];

    double prev_x_map_t2 = previous_path_x[prev_path_size-2];
    double prev_y_map_t2 = previous_path_y[prev_path_size-2];

    double ref_yaw = atan2(
      ref_x_map - prev_x_map_t2,
      ref_y_map - prev_y_map_t2
    );

    anchor_points_x.push_back(prev_x_map_t2);
    anchor_points_x.push_back(ref_x_map);

    anchor_points_y.push_back(prev_y_map_t2);
    anchor_points_y.push_back(ref_y_map);
  }

  // Genreate few points in the future using CarPosiiton in Frenet Coordinate Frame
  double next_d = 4*getLane(car_d) + 2;
  std::cout << "\tlane num = " <<  getLane(car_d) << "\n";

  vector<int> distances {30, 60, 90};

  for (int nxy=0; nxy<distances.size(); nxy++){
    vector<double> next_xy = getXY(
      car_s+distances[nxy],
      next_d,
      map_waypoints_s,
      map_waypoints_x,
      map_waypoints_y
    );
    anchor_points_x.push_back(next_xy[0]);
    anchor_points_y.push_back(next_xy[1]);
    std::cout << "\tdistance = " << distances[nxy] << " next_x = " << next_xy[0] << " nxt_y = " << next_xy[1] << "\n";
  }

  // Now we convert the points in map coordinate frame to vehicle coordinate
  // frame using the ref_x, ref_y and ref_yaw
  for (int np=0; np<anchor_points_x.size(); np++){
    std::cout << "\t-> Map Frame: x = " << anchor_points_x[np] << " y = " << anchor_points_y[np] << "\n";
  }
  std::cout << "\tref_x_map = " << ref_x_map << " ref_y_map = " << ref_y_map << "ref_yaw = " << ref_yaw << "\n";

  vector<vector<double>> transXY = transformMapToVehicleFrame(
    ref_x_map,
    ref_y_map,
    ref_yaw,
    anchor_points_x,
    anchor_points_y
  );
  vector<double> anchor_points_x_v = transXY[0];
  vector<double> anchor_points_y_v = transXY[1];

  for (int np=0; np<anchor_points_x_v.size(); np++){
    std::cout << "\t-> Vehicle Frame: x = " << anchor_points_x_v[np] << " y = " << anchor_points_y_v[np] << "\n";
  }

  // ----------------------------------------------------------------------
  // Step 2: Fit peicewise polynomial function to anchor points using spline
  // ----------------------------------------------------------------------
  tk::spline spl;
  spl.set_points(anchor_points_x_v, anchor_points_y_v);
  double target_y = spl(target_trajectory_distance);
  double target_hypoteneous = sqrt(
    (target_trajectory_distance*target_trajectory_distance) +
    (target_y * target_y)
  );


  // ----------------------------------------------------------------------
  // Step 3: n-step Trajectory Generation
  // ----------------------------------------------------------------------
  double N = target_hypoteneous/(curr_v*sec_to_visit_next_point);
  double add_x = 0;
  vector<double> next_points_x_v;
  vector<double> next_points_y_v;
  for (int i=0; i<=target_trajectory_points-prev_path_size; i++){
    double x_point = add_x + (target_trajectory_distance/N);
    double y_point = spl(x_point);
    std::cout << "\tNext vals vehicle: x_point = " << x_point << " y_point = " << y_point << "\n";

    next_points_x_v.push_back(x_point);
    next_points_y_v.push_back(y_point);

    add_x = x_point;
  }

  vector<vector<double>> next_points_xy_m = transformVehicleToMapFrame(
    ref_x_map,
    ref_y_map,
    ref_yaw,
    next_points_x_v,
    next_points_y_v
  );

  vector<double> next_points_x_m;
  vector<double> next_points_y_m;

  // Add the points from previous trajectory and
  for (int i=0; i<prev_path_size; i++){
    next_points_x_m.push_back(previous_path_x[i]);
    next_points_y_m.push_back(previous_path_y[i]);
  }

  for (int i=0; i<next_points_xy_m[0].size(); i++){
    next_points_x_m.push_back(next_points_xy_m[0][i]);
    next_points_y_m.push_back(next_points_xy_m[1][i]);
  }

  for (int fp=0; fp<next_points_x_m.size(); fp++){
    std::cout << "\tFINAL VEHICLE: num = " << fp << " x =" << next_points_x_m[fp] << " y = " << next_points_y_m[fp] << "\n";
  }
return {next_points_x_m, next_points_y_m};


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

  Class 1: Vehicle:
    -> private :
        -> id
        -> position (s)
        -> velocity (v)
        -> and others
        -> horizon (timsteps that we want to project the behaviour of other vehicles)
    -> methods:
        -> position_at_timestep
        -> prediction_for_timesteps (creates new vehicle object)


Class 2: My Vehicle:
  1. end goal: determine a trajectory
  2. whats needed:
    -> position = s
    -> Velocity = v
    -> accelearation = a
    -> state={
          CS (constant speed),
          KL (keep lane) ,
          PLCL (Plan lane change left),
          PLCR (Plan lane change Right),
          LCL (Lane Change Left),
          LCR (Lane Change Right)
        }
  3. Things to think about:
    -> Every State will have its own Trajectory
    -> And a trajectory is a list of (lane, pos, vel, acc, state),
    -> say for steps = 2, we may have
        trajectory = [
              [t=1 Vehicle(lane, pos, vel, acc, state)]
              [t=2 Vehicle(lane, pos, vel, acc, state)]
        ]
  -> methods

      -> get_kinematics: Uses prediction(other vehicle info) to derive the next moves (new_position, new_velocity, new_acceleartion)
      -> get_vehicle_ahead
      -> get_vehicle_behind
*/
