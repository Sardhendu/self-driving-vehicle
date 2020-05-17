#include <iostream>
#include <vector>
#include "vehicle.h"
#include "utils.h"
#include "spline.h"
#include "experiments.h"

#include <stdlib.h>


using std::vector;
using std::min;
using std::max;


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
  vector<vector<double>> sensor_fusion_data,
  double end_path_s,
  double end_path_d
){
  // std::cout << "set s: " << s << "\n";
  // std::cout << "set d: " << d << "\n";
  car_x = x;
  car_y = y;
  car_s = s;
  car_d = d;
  car_yaw = yaw;
  car_speed = speed;
  car_lane = getLane(car_d);
  waypoints_s_map = map_waypoints_s;
  waypoints_x_map = map_waypoints_x;
  waypoints_y_map = map_waypoints_y;
  std::cout << "Fuckin Car Lane ====> " << car_lane << "\n";
  // goal_s = end_path_s;
  // goal_d = end_path_d;
  // distance_to_goal = distanceCalculate(goal_d, goal_s, car_d, car_s);
  // std::cout << "Distance to goal ================= " << distance_to_goal << "\n";
  // std::cout << "\tcar_s = " << car_s << " goal_s " << goal_s << "\n";
  // std::cout << "\tcar_d = " << car_d << " goal_d " << goal_d << "\n";

  // std::cout << "set car_s: " << car_s << "\n";
  // std::cout << "set car_d: " << car_d << "\n";
  prediction_obj.setPredictions(sensor_fusion_data, "CS");
}


vector<vector<double>> Vehicle::generateTrajectory(
  vector<double> previous_path_x,
  vector<double> previous_path_y
){
  std::cout << "Fuckin Car Lane MotherFucker ====> " << car_lane << "\n";
  vector<string> future_states = getNextStates(car_state);
  // srd::cout << "future_states ................" << future_states << "\n";

  // ------------------------------------------------------------------------
  // Get Future State Kinematics
  // ------------------------------------------------------------------------
  string selected_state = "KL";
  Kinematics KLK;
  Kinematics PLCLK;
  Kinematics PLCRK;
  Kinematics LCLK;
  Kinematics LCRK;
  deque<Trajectory> KLK_TRAJECTORY;
  deque<Trajectory> PLCLK_TRAJECTORY;
  deque<Trajectory> PLCRK_TRAJECTORY;
  deque<Trajectory> LCLK_TRAJECTORY;
  deque<Trajectory> LCRK_TRAJECTORY;

  map<string, double> state_velocity;
  std::cout << "\n[Future_states kinematics] = ..............................................................." << "\n" ;
  std::cout << "Car Lane = " << car_lane << " car_state " << car_state << "\n";

  for (int i=0; i<future_states.size(); i++){
    std::cout << "[Future State] = ............." << future_states[i]  << "\n";

    if (future_states[i] == "KL"){
      KLK = keepLaneKinematics(car_lane);


      KLK_TRAJECTORY = keepLaneTrajectory(
        KLK.velocity,
        KLK.lane,
        previous_path_x,
        previous_path_y,
        final_trajectory
      );

      double total_velocity = 0;
      double total_acceleration = 0;
      for (int i =0; i<KLK_TRAJECTORY.size(); i++){
        total_velocity += KLK_TRAJECTORY[i].v;
        total_acceleration += KLK_TRAJECTORY[i].a;
      }
      state_velocity[future_states[i]] = total_velocity;
      std::cout << "\tKL:\t"
      << "car_v = " << KLK.velocity
      << " intended_lane = " << KLK.lane
      << " total_velocity = " << total_velocity
      << " total_acceleration = " << total_acceleration << "\n";
    }

    if (future_states[i] == "PLCL"){
      PLCLK = prepareLaneChangeKinematics(future_states[i], car_lane);
      std::cout << "\tPLCL:\t" << "car_v = " << PLCLK.velocity << " intended_lane = " << PLCLK.lane << "\n";

      PLCLK_TRAJECTORY = keepLaneTrajectory(
        PLCLK.velocity,
        PLCLK.lane,
        previous_path_x,
        previous_path_y,
        final_trajectory
      );

      double total_velocity = 0;
      double total_acceleration = 0;
      for (int i =0; i<PLCLK_TRAJECTORY.size(); i++){
        total_velocity += PLCLK_TRAJECTORY[i].v;
        total_acceleration += PLCLK_TRAJECTORY[i].a;
      }

      state_velocity[future_states[i]] = total_velocity;

      std::cout << "\tPLCL:\t"
      << "car_v = " << PLCLK.velocity
      << " intended_lane = " << PLCLK.lane
      << " total_velocity = " << total_velocity
      << " total_acceleration = " << total_acceleration << "\n";

    }

    if (future_states[i] == "PLCR"){
      PLCRK = prepareLaneChangeKinematics(future_states[i], car_lane);
      std::cout << "\tPLCR:\t" << "car_v = " << PLCRK.velocity << " intended_lane = " << PLCRK.lane << "\n";
      PLCRK_TRAJECTORY = keepLaneTrajectory(
        PLCRK.velocity,
        PLCRK.lane,
        previous_path_x,
        previous_path_y,
        final_trajectory
      );

      double total_velocity = 0;
      double total_acceleration = 0;
      for (int i =0; i<PLCRK_TRAJECTORY.size(); i++){
        total_velocity += PLCRK_TRAJECTORY[i].v;
        total_acceleration += PLCRK_TRAJECTORY[i].a;
      }
      state_velocity[future_states[i]] = total_velocity;

      std::cout << "\tPLCR:\t"
      << "car_v = " << PLCRK.velocity
      << " intended_lane = " << PLCRK.lane
      << " total_velocity = " << total_velocity
      << " total_acceleration = " << total_acceleration << "\n";
    }

    if (future_states[i] == "LCL"){
      LCLK = laneChangeKinematics(future_states[i], car_lane);
      std::cout << "\tLCL:\t" << "car_v = " << LCLK.velocity << " intended_lane = " << LCLK.lane << "\n";

      LCLK_TRAJECTORY = keepLaneTrajectory(
        LCLK.velocity,
        LCLK.lane,
        previous_path_x,
        previous_path_y,
        final_trajectory
      );

      double total_velocity = 0;
      double total_acceleration = 0;
      for (int i =0; i<LCLK_TRAJECTORY.size(); i++){
        total_velocity += LCLK_TRAJECTORY[i].v;
        total_acceleration += LCLK_TRAJECTORY[i].a;
      }
      state_velocity[future_states[i]] = total_velocity;

      std::cout << "\tLCL:\t"
      << "car_v = " << LCLK.velocity
      << " intended_lane = " << LCLK.lane
      << " total_velocity = " << total_velocity
      << " total_acceleration = " << total_acceleration << "\n";
    }

    if (future_states[i] == "LCR"){
      LCRK = laneChangeKinematics(future_states[i], car_lane);
      std::cout << "\tLCR:\t" << "car_v = " << LCRK.velocity << " intended_lane = " << LCRK.lane << "\n";

      LCRK_TRAJECTORY = keepLaneTrajectory(
        LCRK.velocity,
        LCRK.lane,
        previous_path_x,
        previous_path_y,
        final_trajectory
      );
      double total_velocity = 0;
      double total_acceleration = 0;
      for (int i =0; i<LCRK_TRAJECTORY.size(); i++){
        total_velocity += LCRK_TRAJECTORY[i].v;
        total_acceleration += LCRK_TRAJECTORY[i].a;
      }
      state_velocity[future_states[i]] = total_velocity;
      std::cout << "\tLCR:\t"
      << "car_v = " << LCRK.velocity
      << " intended_lane = " << LCRK.lane
      << " total_velocity = " << total_velocity
      << " total_acceleration = " << total_acceleration << "\n";

    }
  }
  std::cout << "\n............................................................................................" << "\n" ;


  std::cout <<"\nstate_velocity DIct "<< "\n";
  double max_velocity_ = -10000;
  for (auto& t: state_velocity){
    std::cout << "\t" << t.first << " " << t.second << "\n";
    if (t.second > max_velocity_){
      max_velocity_ = t.second;
      car_state = t.first;
    }
  }

  std::cout  << "[Selected State] ..........................." <<  car_state << "\n";
  if (car_state == "KL"){
    car_v = KLK.velocity;
    car_lane = KLK.lane;
    final_trajectory = KLK_TRAJECTORY;
    std::cout << "\t car_state = " << car_state << " car_v = " << car_v << " car_lane = " << car_lane << "\n";
  }
  else if (car_state == "PLCL"){
    car_v = PLCLK.velocity;
    car_lane = PLCLK.lane;
    final_trajectory = PLCLK_TRAJECTORY;
    std::cout << "\t car_state = " << car_state << " car_v = " << car_v << " car_lane = " << car_lane << "\n";
  }
  else if (car_state == "PLCR"){
    car_v = PLCRK.velocity;
    car_lane = PLCRK.lane;
    final_trajectory = PLCRK_TRAJECTORY;
    std::cout << "\t car_state = " << car_state << " car_v = " << car_v << " car_lane = " << car_lane << "\n";
  }
  else if (car_state == "LCL"){
    car_v = LCLK.velocity;
    car_lane = LCLK.lane;
    final_trajectory = LCLK_TRAJECTORY;
    std::cout << "\t car_state = " << car_state << " car_v = " << car_v << " car_lane = " << car_lane << "\n";
  }
  else if (car_state == "LCR"){
    car_v = LCRK.velocity;
    car_lane = LCRK.lane;
    final_trajectory = LCRK_TRAJECTORY;
    std::cout << "\t car_state = " << car_state << " car_v = " << car_v << " car_lane = " << car_lane << "\n";
  }
  else{
    // std::cout << "[prepareLaneChangeKinematics] Provided state = " << state << " Excepted = PLCL and PLCR" << "\n";
    exit (EXIT_FAILURE);
  }


  std::cout << "generateTrajectory \n"
  << "\t curr_lane_velocity: " << car_v
  << "\t max_lane_velocity:" << max_lane_velocity[car_lane]
  << "\n";



  std::cout << "Car ===========================>"
  << "\n\t car_x \t" << car_x
  << "\n\t car_y \t " << car_y
  << "\n\t car_s \t" << car_s
  << "\n\t car_d \t" << car_d
  << "\n\t car_yaw degrees \t" << car_yaw
  << "\n\t car_speed miles/hr \t" << car_speed
  << "\n\t car_v meter/sec \t" << car_v
  << "\n\t car_v_prev meter/sec \t" << car_v_prev
  << "\n\t car_a meter/sec \t" << car_a
  << "\n\t car_lane \t" << car_lane << "\n";

  car_a = car_v - car_v_prev;         // acceleration = change_of_velocity/time
  car_v_prev = car_v;

  vector<double> trajectory_x_points;
  vector<double> trajectory_y_points;
  for (int i=0; i<final_trajectory.size(); i++){
    trajectory_x_points.push_back(final_trajectory[i].x_map);
    trajectory_y_points.push_back(final_trajectory[i].y_map);
  }
  return {trajectory_x_points, trajectory_y_points};
}



// void trajectoryCost(deque<Trajectory> trajectory){
//   duble total_distance_to_goal
// }


// -----------------------------------------------------------------------------
// Prepare Lane Change Kinematics
// -----------------------------------------------------------------------------
Kinematics Vehicle::prepareLaneChangeKinematics(
  string state,
  int curr_lane
){

  int intended_lane;
  if (state == "PLCL"){
    intended_lane = curr_lane -  1;
  }
  else if (state == "PLCR"){
    intended_lane = curr_lane + 1;
  }
  else{
      std::cout << "[prepareLaneChangeKinematics] Provided state = " << state << " Excepted = PLCL and PLCR" << "\n";
      exit (EXIT_FAILURE);
  }

  std::cout << "[prepareLaneChangeKinematics] " << "state: " <<state<<" curr_lane: "<<curr_lane << " intended_lane: " << intended_lane << "\n";

  Traffic vehicle_ahead = prediction_obj.getNearestVehicleAhead(
    car_s, intended_lane
  );

  Traffic vehicle_behind = prediction_obj.getNearestVehicleBehind(
    car_s, intended_lane
  );

  double nw_v = getKinematics(vehicle_ahead, vehicle_behind, intended_lane);

  Kinematics PLCK;
  PLCK.velocity = nw_v;
  if (vehicle_behind.lane == intended_lane){
    if (car_s-vehicle_behind.s <= lane_change_vehicle_behind_buffer){
      // Stay in current lane if there is a vehicle in the front in the intended lane
      PLCK.lane = curr_lane;
    }
    else{
      PLCK.lane = intended_lane;
    }
  }
  else{
    PLCK.lane = intended_lane;
  }
  return PLCK;
}

// -----------------------------------------------------------------------------
// Keep Lane Kinematics
// -----------------------------------------------------------------------------
Kinematics Vehicle::keepLaneKinematics(
  int curr_lane
){
  std::cout << "[keepLaneKinematics] " << "\n";

  Traffic vehicle_ahead = prediction_obj.getNearestVehicleAhead(
    car_s, curr_lane
  );

  Traffic vehicle_behind = prediction_obj.getNearestVehicleBehind(
    car_s, curr_lane
  );

  double nw_v = getKinematics(vehicle_ahead, vehicle_behind, curr_lane);

  Kinematics KLK;
  KLK.velocity = nw_v;
  KLK.lane = curr_lane;
  return KLK;
}

// -----------------------------------------------------------------------------
// Lane Change Kinematics
// -----------------------------------------------------------------------------
Kinematics Vehicle::laneChangeKinematics(
  string state, int curr_lane
){
  int intended_lane;
  if (state == "LCL"){
    intended_lane = curr_lane - 1;
  }
  else if (state == "LCR"){
    intended_lane = curr_lane + 1;
  }
  else{
      std::cout << "[laneChangeKinematics] Provided state = " << state << " Excepted = LCL and LCR" << "\n";
      exit (EXIT_FAILURE);
  }

  std::cout << "[laneChangeKinematics]" << " state: " <<state<<" curr_lane: "<<curr_lane << " intended_lane: " << intended_lane << "\n";

  Traffic vehicle_ahead = prediction_obj.getNearestVehicleAhead(
    car_s, intended_lane
  );

  Traffic vehicle_behind = prediction_obj.getNearestVehicleBehind(
    car_s, intended_lane
  );

  double nw_v = getKinematics(vehicle_ahead, vehicle_behind, intended_lane);

  Kinematics LCK;
  LCK.velocity = nw_v;
  LCK.lane = intended_lane;
  return LCK;
}


// -----------------------------------------------------------------------------
// Get kinematics
// -----------------------------------------------------------------------------
double Vehicle::getKinematics(
  Traffic vehicle_ahead, Traffic vehicle_behind, int intended_lane
){
  /* Vehicle ahead and Vehicle behind can still return vehicle not in our lane, in cases where there are no vhicle in our lane
      So here we check that condition

      TODO:
        1. The output Kinematics is highly dependent on whether the vehicle ahead is in the same lane as our vehicle.
          What if a vehicle tries to cut in abruptly into our lane. Hence the velocity kinematics should depend on d value not lane
            To think:
              1. Use collision_buffer_distance to understand if a vehicle is trying to cut in.
                  If distance out_vehcicle to vehicle_in_front is << 35 it means the vehicle is cutting in.
              2. A much better way is to track every vehicle within a certain distance and
                  -> USe s and d in normalized coordinate system to understand our distance from the vehicle
                    Why normalized: because d << s
        2. Here we only generate kinematics using only the nearest vehicle,
          however to find the best lane we need to consider all the vehciles we can sense, their velocity
          and their distance from our car
  */
  double new_velocity;
  double max_v_a = car_v + car_a_max;   // To avoid jerk
  std::cout << "\tintended_regular_velocity = " << max_v_a << "\n";
  std::cout << "\t[getKinematics]: \n"
  << "\t\tcar_lane = " << car_lane
  << "\t\tvehicle_lane = " << vehicle_ahead.lane << "\n";

  if (vehicle_ahead.lane == intended_lane){
    std::cout << "\t\tTHERE IS A VEHICLE AHEAD =========> " << "\n";
    std::cout << "\t\tvehicle_ahead:\t s = " <<  vehicle_ahead.s << " v = " << vehicle_ahead.v << "\n";
    std::cout << "\t\tmy_vehicle:\t s = " <<  car_s << " v = " << car_v << " a = " << car_a << "\n";
    // If there is a vehicle behind make our car move with the speed that of the front
    // if (vehicle_behind.lane == car_lane){
    //   // Choose to follow the traffic velocity when there is a car ahead
    //   std::cout << "THERE IS A VEHICLE BEHIND =========> " << "\n";
    //   std::cout << "vehicle_behind:\t s = " <<  vehicle_behind.s << " v = " << vehicle_behind.v << "\n";
    //   new_velocity = vehicle_ahead.v;
    // }
    // else{

    /*
      Assuming our goal is to reach the preffered position that is buffer_meters behind the car ahead, we use the
      Jerk minimization equation to calculate the preferend velocity the vehicle should given that we use the same acceleration
      goal s(t)(vehicle_ahead.s - buffer_distance) = car.s + (max_velocity_ahead - vehicle_ahead.v)*t + car.a*(t**2)
    */
      double max_velocity_ahead = vehicle_ahead.s - car_s - collision_buffer_distance + vehicle_ahead.v - 0.5*car_a;
      std::cout << "\t\tmax_velocity_ahead = " << max_velocity_ahead << "\n";
      new_velocity = min(max_velocity_ahead, max_lane_velocity[car_lane]);
      new_velocity = min(new_velocity, max_v_a);
      std::cout << "\t\tnew_velocity = " << new_velocity << "\n";
    }
  // }
  else{
    // When there are no cars ahead, increment the cars velocity untill max permitted by lane
    // new_velocity = min(max_v_a, max_lane_velocity[car_lane]);
    new_velocity = min(car_v+0.2, max_lane_velocity[car_lane]);
    std::cout << "\t\tnew_velocity = " << new_velocity << "\n";
  }

  return new_velocity;

}


// -----------------------------------------------------------------------------
// Keep Lane Trajectory
// -----------------------------------------------------------------------------
deque<Trajectory> Vehicle::keepLaneTrajectory(
  double curr_v,      // current velocity
  int curr_lane,      // Required for proper trajectory
  vector<double> previous_path_x,
  vector<double> previous_path_y,
  deque<Trajectory> trajectories
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
  // std::cout << "\t x_map = " << x_map << " y_map = " << y_map << " car_yaw = " << car_yaw << " s = " << car_s << " d = " << car_d << "\n";
  // std::cout << "\tlen(previous_path_x) = " << previous_path_x.size() << "\n";
  // std::cout << "\tcurrently velocity = " << curr_v << "\n";
  // std::cout << "\predict_distance = " << predict_distance << "\n";
  // std::cout << "\tsec_to_visit_next_point = " << sec_to_visit_next_point << "\n";
  vector<double> anchor_points_x;
  vector<double> anchor_points_y;

  double ref_x_map = car_x;
  double ref_y_map = car_y;
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
    double prev_x_map = car_x - cos(car_yaw);
    double prev_y_map = car_y - sin(car_yaw);

    anchor_points_x.push_back(prev_x_map);
    anchor_points_x.push_back(car_x);

    anchor_points_y.push_back(prev_y_map);
    anchor_points_y.push_back(car_y);

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
  double next_d = 4*curr_lane + 2;
  // std::cout << "\tlane num = " <<  getLane(car_d) << "\n";


  for (int nxy=0; nxy<poly_fit_distances.size(); nxy++){
    vector<double> next_xy = getXY(
      car_s+poly_fit_distances[nxy],
      next_d,
      waypoints_s_map,
      waypoints_x_map,
      waypoints_y_map
    );
    anchor_points_x.push_back(next_xy[0]);
    anchor_points_y.push_back(next_xy[1]);
    // std::cout << "\tdistance = " << poly_fit_distances[nxy] << " next_x = " << next_xy[0] << " nxt_y = " << next_xy[1] << "\n";
  }

  // Now we convert the points in map coordinate frame to vehicle coordinate
  // frame using the ref_x, ref_y and ref_yaw
  // for (int np=0; np<anchor_points_x.size(); np++){
  //   std::cout << "\t-> Map Frame: x = " << anchor_points_x[np] << " y = " << anchor_points_y[np] << "\n";
  // }
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
  double target_y = spl(predict_distance);
  double target_hypoteneous = sqrt(
    (predict_distance*predict_distance) +
    (target_y * target_y)
  );


  // ----------------------------------------------------------------------
  // Step 3: n-step Trajectory Generation
  // ----------------------------------------------------------------------
  double N = target_hypoteneous/(curr_v*sec_to_visit_next_point);
  double add_x = 0;
  vector<double> next_points_x_v;
  vector<double> next_points_y_v;
  std::cout << "\t[Generate Tajectory]\t "
  << " previous_path_size = " << prev_path_size
  << " new_paths = " << trajectory_length-prev_path_size << "\n";


  int new_points_count = trajectory_length-prev_path_size;
  for (int i=0; i<new_points_count; i++){
    double x_point = add_x + (predict_distance/N);
    double y_point = spl(x_point);
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

  // vector<double> next_points_x_m;
  // vector<double> next_points_y_m;

  // Add the points from previous trajectory and
  // for (int i=0; i<prev_path_size; i++){
  //   next_points_x_m.push_back(previous_path_x[i]);
  //   next_points_y_m.push_back(previous_path_y[i]);
  // }

  if (!trajectories.empty()){
    for (int i=0; i<new_points_count; i++){
        // std::cout << "\tRemoving " << i << "\t " << "dat = " <<trajectories[0].id << "\t" << trajectories[0].x_map << "\t" << trajectories[0].y_map << "\n";
        trajectories.pop_front();
    }
  }
  // Insert New trajectory points
  for (int i=0; i<next_points_xy_m[0].size(); i++){
    // std::cout << "\tAdding: "<< i << "\t " << "dat = " << i << "\t" << next_points_xy_m[0][i] << "\t" << next_points_xy_m[1][i] << "\n";
    // next_points_x_m.push_back(next_points_xy_m[0][i]);
    // next_points_y_m.push_back(next_points_xy_m[1][i]);

    Trajectory single_point;
    single_point.id = i;
    single_point.lane = curr_lane;
    single_point.x_map = next_points_xy_m[0][i];
    single_point.y_map = next_points_xy_m[1][i];
    single_point.v = curr_v;
    single_point.a = car_a;
    single_point.state = car_state;
    trajectories.push_back(single_point);
  }
  std::cout << "\t\tFinal Trajectory Length = "<< trajectories.size() << "\n";
  //
  // for (int fp=0; fp<next_points_x_m.size(); fp++){
  //   std::cout << "\tFINAL VEHICLE: num = " << fp << " x =" << next_points_x_m[fp] << " y = " << next_points_y_m[fp] << "\n";
  // }

return trajectories; // trajectories;

}


vector<string> Vehicle::getNextStates(string current_state){
  vector<string> states_vector;
  if (current_state == "KL"){
     if(car_lane == 0){
      states_vector = {"KL", "PLCR"};
    }
    else if(car_lane == 2){
      states_vector = {"KL", "PLCL"};
    }
    else{
      states_vector = {"KL", "PLCL", "PLCR"};
    }

  }
  else if (current_state == "PLCL"){
    states_vector.push_back("KL");
    if (car_lane != 0){
      states_vector.push_back("PLCL");
      states_vector.push_back("LCL");
    }
  }
  else if (current_state == "PLCR"){
    states_vector.push_back("KL");
    if (car_lane != 1){
      states_vector.push_back("PLCR");
      states_vector.push_back("LCR");
    }
  }
  else{
    states_vector = {"KL"};
  }

  for (int i=0; i<states_vector.size(); i++){
    std::cout << " Future State = " << states_vector[i] << " " ;
  }
  std::cout << "\n";
  return states_vector;
}




  // // ------------------------------------------------------------------------
  // // Small Hack to test
  // // ------------------------------------------------------------------------
  // Traffic vehicle_ahead = prediction_obj.getNearestVehicleAhead(
  //   car_s, car_lane
  // );
  // int curr_lane;
  // std::cout << vehicle_ahead.lane << " " << car_lane << "\n";
  // if (vehicle_ahead.lane == car_lane){
  //   if (vehicle_ahead.s - car_s <= collision_buffer_distance){
  //     curr_lane = car_lane + 1;
  //   }
  //   else{
  //     curr_lane = car_lane;
  //   }
  // }
  // else{
  //     curr_lane = car_lane;
  //   }
  // // ------------------------------------------------------------------------

  // std::cout << "curr_lanecurr_lanecurr_lanecurr_lanecurr_lanecurr_lane  " << curr_lane << "\n";
  // curr_lane = getLane(car_d);
