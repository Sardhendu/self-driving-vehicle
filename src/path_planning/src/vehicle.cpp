#include <iostream>
#include <vector>
#include "vehicle.h"
#include "utils.h"
#include "spline.h"
#include "experiments.h"
#include "cost.h"
#include <stdlib.h>
#include <assert.h>

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
  double end_path_d,
  int prev_trajectory_size
){
  car_x = x;
  car_y = y;
  car_s = s;
  car_d = d;
  car_yaw = yaw;
  car_speed = speed;
  std::cout << "Actual car position in Frenet coordinate synstem: \n" << "\ts = " << car_s << "\td = "<< car_d << "\n";
  waypoints_s_map = map_waypoints_s;
  waypoints_x_map = map_waypoints_x;
  waypoints_y_map = map_waypoints_y;

  if (prev_trajectory_size > 0){
    /*
      This is an important step. In order to avoid collision (mostly due to unforeseen behavious
      of other vehicles) we should use the most future trajectory point to estimate the next car position.
    */
    car_s = end_path_s;
    car_d = end_path_d;
  }
  std::cout << "Future car position in Frenet coordinate synstem: \n" << "\ts = " << car_s << "\td = "<< car_d << "\n";
  std::cout << "prev_trajectory_size ====> " << prev_trajectory_size << "\n";

  prediction_obj.setPredictions(sensor_fusion_data, prev_trajectory_size, "CS");
}


vector<vector<double>> Vehicle::generateTrajectory(
  vector<double> previous_path_x,
  vector<double> previous_path_y
){
  vector<string> list_of_future_states = getNextStates(car_state);


  // ------------------------------------------------------------------------
  // Get Traffic Ahead and Behind
  // ------------------------------------------------------------------------
  vector<map<int, vector<Traffic>>> traffic_ = prediction_obj.getTraffic(car_s);
  map<int, vector<Traffic>> traffic_ahead = traffic_[0];
  map<int, vector<Traffic>> traffic_behind = traffic_[1];

  // ------------------------------------------------------------------------
  // Get Future State Kinematics
  // ------------------------------------------------------------------------
  std::cout << "\n[Future_states kinematics] = ..............................................................." << "\n" ;
  std::cout << "Car Lane = " << car_lane << " car_state " << car_state << "\n";

  Kinematics KN;
  deque<Trajectory> TJ;
  vector<deque<Trajectory>> list_of_trajectories;
  vector<Kinematics> list_of_kinematics;

  for (int i=0; i<list_of_future_states.size(); i++){
    std::cout << "[Future State] = ............................................" << list_of_future_states[i]  << "\n";
    if (list_of_future_states[i] == "KL"){
      KN = keepLaneKinematics(car_lane, traffic_ahead, traffic_behind);
    }
    if (list_of_future_states[i] == "PLCL" || list_of_future_states[i] == "PLCR"){
      KN = prepareLaneChangeKinematics(list_of_future_states[i], car_lane, traffic_ahead, traffic_behind);
    }
    if (list_of_future_states[i] == "LCL" || list_of_future_states[i] == "LCR"){
      KN = laneChangeKinematics(list_of_future_states[i], car_lane, traffic_ahead, traffic_behind);
    }
    std::cout << "\t" << list_of_future_states[i] << ":\t" << "car_v = " << KN.velocity << " intended_lane = " << KN.lane << "\n";

    TJ = generateTrajectoryForState(KN.velocity, KN.lane, previous_path_x, previous_path_y, FINAL_TRAJECTORY);
    list_of_kinematics.push_back(KN);
    list_of_trajectories.push_back(TJ);
  }

  // ------------------------------------------------------------------------
  // Lane Change and Traffic Cost
  // ------------------------------------------------------------------------
  map<int, double> lane_traffic_cost = laneTrafficCost(traffic_ahead, car_s);
  map<int, double> lane_change_cost = laneChangeCost(traffic_behind, traffic_ahead, car_s, car_v);


  // ------------------------------------------------------------------------
  // Genrate Trajectory
  // ------------------------------------------------------------------------
  std::cout << "\n[Optimal State Kinematics].........................................................." << "\n" ;
  int op_num = getOptimalTrajectoryNum(
    list_of_trajectories,
    list_of_kinematics,
    list_of_future_states,
    lane_traffic_cost,
    lane_change_cost
  );
  std::cout << "\toptimal num = " << op_num << "\n";
  car_v = list_of_kinematics[op_num].velocity;
  car_lane = list_of_kinematics[op_num].lane;
  FINAL_TRAJECTORY = list_of_trajectories[op_num];
  car_state = list_of_future_states[op_num];
  std::cout << "\t[Optimal] car_state = " << car_state << " car_v = " << car_v << " car_lane = " << car_lane << "\n";

  std::cout << "generateTrajectory \n"
  << "\t curr_lane_velocity: " << car_v
  << "\t MAX_LANE_VELOCITY:" << MAX_LANE_VELOCITY[car_lane]
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
  for (int i=0; i<FINAL_TRAJECTORY.size(); i++){
    trajectory_x_points.push_back(FINAL_TRAJECTORY[i].x_map);
    trajectory_y_points.push_back(FINAL_TRAJECTORY[i].y_map);
  }
  return {trajectory_x_points, trajectory_y_points};
}



// -----------------------------------------------------------------------------
// Optimal Trajectory
// -----------------------------------------------------------------------------
int Vehicle::getOptimalTrajectoryNum(
  vector<deque<Trajectory>> list_of_trajectories,
  vector<Kinematics> list_of_kinematics,
  vector<string> list_of_states,
  map<int, double> lane_traffic_cost,
  map<int, double> lane_change_cost
){

  vector<double> insufficiency_cost;
  for (int i=0; i<list_of_trajectories.size(); i++){
    deque<Trajectory> trajectory_ = list_of_trajectories[i];
    Kinematics kinematics_ = list_of_kinematics[i];
    string state_ = list_of_states[i];

    double total_velocity = 0;
    double total_acceleration = 0;
    for (int j =0; j<trajectory_.size(); j++){
      total_velocity += trajectory_[j].v;
      total_acceleration += trajectory_[j].a;
    }

    insufficiency_cost.push_back(total_velocity);

    std::cout << "\t" << state_ << ":\t"
    << "car_v = " << kinematics_.velocity
    << " intended_lane = " << kinematics_.lane
    << " max_velocity = " << kinematics_.max_velocity
    << " total_velocity = " << total_velocity
    << " total_acceleration = " << total_acceleration << "\n";
  }

  int optimial_trajectory_num = 0;
  double max_veclocity = -9999;

  for (int i=0; i<insufficiency_cost.size(); i++){
    if (insufficiency_cost[i] > max_veclocity && insufficiency_cost[i]>insufficiency_cost[0]+0.2){
        optimial_trajectory_num = i;
    }
  }


  vector<double> norm_velocity = normalize(insufficiency_cost);
  for (int i=0; i<norm_velocity.size(); i++){
    insufficiency_cost[i] = 1 - norm_velocity[i];
  }

  // Find the best lane given state velocity and lane traffic
  int minimum_cost_trajectory = 0;
  double minimum_cost = 99999;

  std::cout << "[Cumulative Cost] \n";
  for (int i=0; i<insufficiency_cost.size(); i++){
    int lane = list_of_kinematics[i].lane;

    std::cout << "\tstate = " << list_of_states[i]
    << "\n\t\tlane = " << list_of_kinematics[i].lane;

    assert (lane >= 0);
    assert (lane <= 2);
    double cumulative_cost = (
      INSUFFICIENCY_COST_WEIGHT*insufficiency_cost[i] +
      LANE_TRAFFIC_COST_WEIGHT*lane_traffic_cost[lane] +
      LANE_CHANGE_COST_WEIGHT*lane_change_cost[lane]
    );

    std::cout << "\n\t\tinsufficiency_cost = " << insufficiency_cost[i]
    << "\n\t\tlane_traffic_cost = " << lane_traffic_cost[lane]
    << "\n\t\tlane_change_cost = " << lane_change_cost[lane]
    << "\n\t\tweighted_cumulative_cost = " << cumulative_cost << "\n";
    if (cumulative_cost < minimum_cost){
        minimum_cost = cumulative_cost;
        minimum_cost_trajectory = i;
    }
  }
  std::cout << "MINIMUM COST TRAJECTORY NUM = " << minimum_cost_trajectory << "\n";

  return minimum_cost_trajectory;
}



// -----------------------------------------------------------------------------
// Keep Lane Kinematics
// -----------------------------------------------------------------------------
Kinematics Vehicle::keepLaneKinematics(
  int curr_lane,
  map<int, vector<Traffic>> traffic_ahead,
  map<int, vector<Traffic>> traffic_behind
){
  std::cout << "[keepLaneKinematics] " << "\n";

  Traffic vehicle_ahead = prediction_obj.getNearestVehicleAheadInLane(
    traffic_ahead, curr_lane
  );

  Traffic vehicle_behind = prediction_obj.getNearestVehicleBehindInLane(
    traffic_behind, curr_lane
  );

  vector<double> v_k  = getKinematics(vehicle_ahead, vehicle_behind, curr_lane);
  double nw_v = v_k[0];
  double max_v = v_k[1];


  Kinematics KLK;
  KLK.velocity = nw_v;
  KLK.lane = curr_lane;
  KLK.max_velocity = max_v;
  return KLK;
}


// -----------------------------------------------------------------------------
// Prepare Lane Change Kinematics
// -----------------------------------------------------------------------------
Kinematics Vehicle::prepareLaneChangeKinematics(
  string state,
  int curr_lane,
  map<int, vector<Traffic>> traffic_ahead,
  map<int, vector<Traffic>> traffic_behind
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

  Traffic vehicle_ahead = prediction_obj.getNearestVehicleAheadInLane(
    traffic_ahead, intended_lane
  );

  Traffic vehicle_behind = prediction_obj.getNearestVehicleBehindInLane(
    traffic_behind, intended_lane
  );


  //Assert to ensure
  if (vehicle_ahead.id != -1){
    assert(vehicle_ahead.lane == intended_lane);
  }

  if (vehicle_behind.id != -1){
    assert(vehicle_behind.lane == intended_lane);
  }
  // assert(vehicle_behind.lane)
  vector<double> v_k = getKinematics(vehicle_ahead, vehicle_behind, intended_lane);
  double nw_v = v_k[0];
  double max_v = v_k[1];

  // std::cout << "\t\t[Vehicle Behind]: "
  // << " s = " << vehicle_behind.s
  // << " speed = " << vehicle_behind.speed
  // << " distance = " << car_s -vehicle_behind.s << "\n";

  Kinematics PLCK;
  PLCK.velocity = nw_v;
  PLCK.max_velocity = max_v;

  if (vehicle_behind.id != -1){
    if (car_s-vehicle_behind.s <= LC_VEHICLE_BEHIND_BUFFER){
      std::cout << "\t\tTHERE IS A VEHICLE BEHIND IN BUFFER =========> " << "\n";
      // Stay in current lane if there is a vehicle in the front in the intended lane
      PLCK.lane = curr_lane;
    }
    else{
      PLCK.lane = intended_lane;
    }
  }
  else{
    PLCK.lane = intended_lane;;
  }


  return PLCK;
}


// -----------------------------------------------------------------------------
// Lane Change Kinematics
// -----------------------------------------------------------------------------
Kinematics Vehicle::laneChangeKinematics(
  string state, int curr_lane,
  map<int, vector<Traffic>> traffic_ahead,
  map<int, vector<Traffic>> traffic_behind
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

  Traffic vehicle_ahead = prediction_obj.getNearestVehicleAheadInLane(
    traffic_ahead, intended_lane
  );

  Traffic vehicle_behind = prediction_obj.getNearestVehicleBehindInLane(
    traffic_behind, intended_lane
  );

  vector<double> v_k  = getKinematics(vehicle_ahead, vehicle_behind, intended_lane);
  double nw_v = v_k[0];
  double max_v = v_k[1];

  Kinematics LCK;
  LCK.velocity = nw_v;
  LCK.max_velocity = max_v;
  LCK.lane = intended_lane;
  return LCK;
}


// -----------------------------------------------------------------------------
// Get kinematics
// -----------------------------------------------------------------------------
vector<double> Vehicle::getKinematics(
  Traffic vehicle_ahead, Traffic vehicle_behind, int intended_lane
){
  /* Vehicle ahead and Vehicle behind can still return vehicle not in our lane, in cases where there are no vhicle in our lane
      So here we check that condition

      TODO:
        1. The output Kinematics is highly dependent on whether the vehicle ahead is in the same lane as our vehicle.
          What if a vehicle tries to cut in abruptly into our lane. Hence the velocity kinematics should depend on d value not lane
            To think:
              1. Use VEHICLE_AHEAD_BUFFER to understand if a vehicle is trying to cut in.
                  If distance out_vehcicle to vehicle_in_front is << 35 it means the vehicle is cutting in.
              2. A much better way is to track every vehicle within a certain distance and
                  -> USe s and d in normalized coordinate system to understand our distance from the vehicle
                    Why normalized: because d << s
        2. Here we only generate kinematics using only the nearest vehicle,
          however to find the best lane we need to consider all the vehciles we can sense, their velocity
          and their distance from our car
  */
  double new_velocity;
  double max_v_a;   // To avoid jerk
  max_v_a = car_v + MAXIMUM_ACCELERATION;

  std::cout << "\t[getKinematics]: \n"
  << "\t\tcar_lane = " << car_lane << "\tintended_lane = " << intended_lane << "\tcar_v_old = " << car_v << "\tcar_v_new = " << max_v_a
  << "\n\t\tvehicle_lane = " << vehicle_ahead.lane << "\n";

  double max_velocity_ahead = -10000;
  if (vehicle_ahead.lane == intended_lane && vehicle_ahead.id != -1){
    /*
      Assuming our goal is to reach the preffered position that is buffer_meters behind the car ahead, we use the
      Jerk minimization equation to calculate the preferend velocity the vehicle should given that we use the same acceleration
      goal s(t)(vehicle_ahead.s - buffer_distance) = car.s + (max_velocity_ahead - vehicle_ahead.v)*t + car.a*(t**2)
    */
      // std::cout << "\t\t[Vehicle Ahead]: "
      // << " s = " << vehicle_ahead.s
      // << " speed = " << vehicle_ahead.speed
      // << " distance = " << vehicle_ahead.s - car_s << "\n";

      if (VEHICLE_AHEAD_BUFFER >= vehicle_ahead.s - car_s){
        // std::cout << "\t\tTHERE IS A VEHICLE AHEAD IN BUFFER =========> " << "\n";
        max_v_a = car_v - MAXIMUM_ACCELERATION;
        // std::cout << "\t\tactual_velocity = " << max_v_a << "\n";
      }
      max_velocity_ahead = vehicle_ahead.s - car_s - VEHICLE_AHEAD_BUFFER + vehicle_ahead.speed - 0.5*car_a;
      std::cout << "\t\tmax_velocity_ahead before = " << max_velocity_ahead << "\n";
      if (max_velocity_ahead<vehicle_ahead.speed){
        max_velocity_ahead = vehicle_ahead.speed;
      }
      std::cout << "\t\tmax_velocity_ahead after = " << max_velocity_ahead << "\n";
      new_velocity = min(max_velocity_ahead, max_v_a);
    }
  // }
  else{
    // When there are no cars ahead, increment the cars velocity untill max permitted by lane
    // new_velocity = min(max_v_a, MAX_LANE_VELOCITY[car_lane]);
    new_velocity = max_v_a;
    max_velocity_ahead = 99999;
  }

  std::cout << "\t\tnew_velocity = " << new_velocity << "\n";
  new_velocity = min(new_velocity, MAX_LANE_VELOCITY[car_lane]);
  return {new_velocity, max_velocity_ahead};

}


// -----------------------------------------------------------------------------
// Keep Lane Trajectory
// -----------------------------------------------------------------------------
deque<Trajectory> Vehicle::generateTrajectoryForState(
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
    // Change the reference to the most future point of the previous trajectory because we use
    // car_s not as the vehicle is currently but where the vehicle will be in the future using trajectory
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
  double next_d = LANE_WIDTH*curr_lane + 2;


  for (int nxy=0; nxy<POLY_FIT_DISTANCES.size(); nxy++){
    vector<double> next_xy = getXY(
      car_s+POLY_FIT_DISTANCES[nxy],
      next_d,
      waypoints_s_map,
      waypoints_x_map,
      waypoints_y_map
    );
    anchor_points_x.push_back(next_xy[0]);
    anchor_points_y.push_back(next_xy[1]);
  }

  // Now we convert the points in map coordinate frame to vehicle coordinate
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
  double target_y = spl(PREDICT_DISTANCE);
  double target_hypoteneous = sqrt(
    (PREDICT_DISTANCE*PREDICT_DISTANCE) +
    (target_y * target_y)
  );


  // ----------------------------------------------------------------------
  // Step 3: n-step Trajectory Generation
  // ----------------------------------------------------------------------
  double N = target_hypoteneous/(curr_v*SEC_TO_VISIT_NEXT_POINT);
  double add_x = 0;
  vector<double> next_points_x_v;
  vector<double> next_points_y_v;
  std::cout << "\t[Generate Tajectory]\t "
  << " previous_path_size = " << prev_path_size
  << " new_paths = " << TRAJECTORY_LENGTH-prev_path_size << "\n";


  int new_points_count = TRAJECTORY_LENGTH-prev_path_size;
  for (int i=0; i<new_points_count; i++){
    double x_point = add_x + (PREDICT_DISTANCE/N);
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

  if (!trajectories.empty()){
    for (int i=0; i<new_points_count; i++){
        trajectories.pop_front();
    }
  }
  // Insert New trajectory points
  for (int i=0; i<next_points_xy_m[0].size(); i++){
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
  std::cout << "[getNextStates]\t" << "car_lane = "<< car_lane << " car_state = " << current_state << "\n";
  vector<string> states_vector;
  if (current_state == "KL"){
    if(car_lane == 0){
      states_vector = {"KL", "PLCR"};
    }
    else if(car_lane == 2){
      states_vector = {"KL", "PLCL"};
    }
    else if(car_lane == 1){
      states_vector = {"KL", "PLCL", "PLCR"};
    }
    else{
      exit (EXIT_FAILURE);
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
    if (car_lane != 2){
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
