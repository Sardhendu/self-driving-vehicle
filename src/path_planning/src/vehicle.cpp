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
  vector<double> map_waypoints_y,
  vector<vector<double>> sensor_fusion_data
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

  prediction_obj.setPredctions(sensor_fusion_data, "CS");
}


vector<vector<double>> Vehicle::generateTrajectory(
  vector<double> previous_path_x,
  vector<double> previous_path_y
){
  // std::cout << "set s_frn: " << s_frn << "\n";
  // std::cout << "set d_frn: " << d_frn << "\n";
  int car_lane = getLane(d_frn);
  curr_velocity = getVelocity(
    curr_velocity,
    increment_velocity,
    max_lane_velocity[car_lane]
  );

  std::cout << "Car ===========================>"
  << "\n\t x_map \t" << x_map
  << "\n\t y_map \t " << y_map
  << "\n\t s_frn \t" << s_frn
  << "\n\t d_frn \t" << d_frn
  << "\n\t car_yaw degrees \t" << car_yaw
  << "\n\t car_speed miles/hr \t" << car_speed
  << "\n\t curr_velocity meter/sec \t" << curr_velocity
  << "\n\t car_lane \t" << car_lane << "\n";

  std::cout << "generateTrajectory \n"
  << "\t curr_lane_velocity: " << curr_velocity
  << "\t max_lane_velocity:" << max_lane_velocity[car_lane]
  << "\n";

  Traffic vehicle_ahead = prediction_obj.getNearestVehicleAhead(
    s_frn,
    car_lane
  );

  Traffic vehicle_behind = prediction_obj.getNearestVehicleBehind(
    s_frn,
    car_lane
  );

  vector<vector<double>> trajectoryXY = keepLaneTrajectory(
    curr_velocity,
    previous_path_x,
    previous_path_y
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



vector<vector<double>> Vehicle::keepLaneTrajectory(
  double curr_v,      // current velocity
  vector<double> previous_path_x,
  vector<double> previous_path_y
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
  // std::cout << "\t x_map = " << x_map << " y_map = " << y_map << " car_yaw = " << car_yaw << " s = " << s_frn << " d = " << d_frn << "\n";
  // std::cout << "\tlen(previous_path_x) = " << previous_path_x.size() << "\n";
  // std::cout << "\tcurrently velocity = " << curr_v << "\n";
  // std::cout << "\ttarget_trajectory_distance = " << target_trajectory_distance << "\n";
  // std::cout << "\tsec_to_visit_next_point = " << sec_to_visit_next_point << "\n";


  vector<double> anchor_points_x;
  vector<double> anchor_points_y;

  double ref_x_map = x_map;
  double ref_y_map = y_map;
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
    double prev_x_map = x_map - cos(car_yaw);
    double prev_y_map = y_map - sin(car_yaw);

    anchor_points_x.push_back(prev_x_map);
    anchor_points_x.push_back(x_map);

    anchor_points_y.push_back(prev_y_map);
    anchor_points_y.push_back(y_map);

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
  double next_d = 4*getLane(d_frn) + 2;
  // std::cout << "\tlane num = " <<  getLane(d_frn) << "\n";

  vector<int> distances {30, 60, 90};

  for (int nxy=0; nxy<distances.size(); nxy++){
    vector<double> next_xy = getXY(
      s_frn+distances[nxy],
      next_d,
      waypoints_s_map,
      waypoints_x_map,
      waypoints_y_map
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
  // std::cout << "\tref_x_map = " << ref_x_map << " ref_y_map = " << ref_y_map << "ref_yaw = " << ref_yaw << "\n";

  vector<vector<double>> transXY = transformMapToVehicleFrame(
    ref_x_map,
    ref_y_map,
    ref_yaw,
    anchor_points_x,
    anchor_points_y
  );
  vector<double> anchor_points_x_v = transXY[0];
  vector<double> anchor_points_y_v = transXY[1];

  // for (int np=0; np<anchor_points_x_v.size(); np++){
  //   std::cout << "\t-> Vehicle Frame: x = " << anchor_points_x_v[np] << " y = " << anchor_points_y_v[np] << "\n";
  // }

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
    // std::cout << "\tNext vals vehicle: x_point = " << x_point << " y_point = " << y_point << "\n";

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
  //
  // for (int fp=0; fp<next_points_x_m.size(); fp++){
  //   std::cout << "\tFINAL VEHICLE: num = " << fp << " x =" << next_points_x_m[fp] << " y = " << next_points_y_m[fp] << "\n";
  // }
return {next_points_x_m, next_points_y_m};


}
