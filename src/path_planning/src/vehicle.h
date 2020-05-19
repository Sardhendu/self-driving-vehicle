#ifndef VEHICLE_H
#define VEHICLE_H
#include <vector>
#include <utility>
#include <deque>
#include "prediction.h"

using std::vector;
using std::deque;

struct Kinematics {
  double velocity;
  int lane;
  double max_velocity;
};


struct Trajectory {
  int id;
  int lane;
  float x_map;
  float y_map;
  float v;    // velocity
  float a;    // acceleration
  string state;
};


class Vehicle {

public:
  vector<double> max_lane_velocity = {49.5/2.24, 49.5/2.24, 49.5/2.24}; // In meters/sec
  double car_v_prev = 0;
  double car_v = 0.2;
  double car_a = 0.2;
  double MAXIMUM_ACCELERATION = 0.224;               // maximum acceleration permitted
  double MAXIMUM_DECCELERATION = 0.224;
  // double increment_velocity = 0.2;
  double SEC_TO_VISIT_NEXT_POINT = 0.02; // how many seconds should the car take to visit the next point (px, py at t+1, when the car is at t)
  int VEHICLE_AHEAD_BUFFER = 20; // 35 Assuming we keep 10 m distance from any car ahead of us
  int LC_VEHICLE_BEHIND_BUFFER = 15;
  vector<int> POLY_FIT_DISTANCES = {30, 60, 90};  // In meters
  int PREDICT_DISTANCE = 30; // meters that the car should look ahead for trajectory generation
  int TRAJECTORY_LENGTH = 50; // num of future points to generate in the trajectory
  int HACK = 5;

  double car_x;
  double car_y;
  double car_s;
  double car_d;
  double car_yaw;
  double car_speed;
  int car_lane=1;
  string car_state = "KL";
  vector<double> waypoints_s_map;
  vector<double> waypoints_x_map;
  vector<double> waypoints_y_map;
  double goal_s;
  double goal_d;
  double distance_to_goal;
  vector<vector<double>> sensor_fusion_data;
  deque<Trajectory> FINAL_TRAJECTORY;

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
    vector<vector<double>> sensor_fusion_data,
    double end_path_s,                            // the s value of the most future trajectory
    double end_path_d,                            // the d value of the most future trajectory
    int prev_trajectory_size
  );

  // -----------------------------------------------------------------------------
  // Generate Trajectory
  // -----------------------------------------------------------------------------
  vector<vector<double>> generateTrajectory(
    vector<double> previous_path_x,
    vector<double> previous_path_y
    // auto previous_path_x,
    // auto previous_path_y
  );

  deque<Trajectory> generateTrajectoryForState(
      double curr_v,      // current velocity
      int curr_lane,
      vector<double> previous_path_x,
      vector<double> previous_path_y,
      deque<Trajectory> trajectories
    );

  // deque<Trajectory> generateTrajectoryForState(
  //     string state, vector<double> previous_path_x, vector<double> previous_path_y
  //   )

  // -----------------------------------------------------------------------------
  // Get kinematics
  // -----------------------------------------------------------------------------
  /*
    This module determines the velocity, acceleration and position of our vehicle given
    the nearest vehicle ahead and behind

    We need to implement stuff from Jerk Minimization
    1. Jerk minimization equation: s is the s position in frenet system
      s_i -> position
      v_i -> velocity
      a_i -> acceleration
      Position: s(t) (goal) = s_i + v_i(t) + 0.5*a_i(t**2)
  */
  vector<double> getKinematics(
    Traffic vehicle_ahead,
    Traffic vehicle_behind,
    int intended_lane
  );

  Kinematics keepLaneKinematics(
    int curr_lane,
    map<int, vector<Traffic>> traffic_ahead,
    map<int, vector<Traffic>> traffic_behind
  );
  Kinematics laneChangeKinematics(
    string state, int curr_lane,
    map<int, vector<Traffic>> traffic_ahead,
    map<int, vector<Traffic>> traffic_behind
  );

  Kinematics prepareLaneChangeKinematics(
    string state, int curr_lane,
    map<int, vector<Traffic>> traffic_ahead,
    map<int, vector<Traffic>> traffic_behind
  );

  int getOptimalTrajectoryNum(
    vector<deque<Trajectory>> list_of_trajectories,
    vector<Kinematics> list_of_kinematics,
    vector<string> list_of_states,
    vector<double> list_of_lane_cost
  );


  // -----------------------------------------------------------------------------
  // Generaate Next States
  // -----------------------------------------------------------------------------
  /*
    Here we define the finite state machine:
    KL      -> keep Lane
    PLCL    -> Prepare lane change left (Here we check for vehicles in the left lane)
    PLCR    -> Prepare lane change right (Here we check for vehicles in the right lane)
    LCL     -> Lane change left
    LCR     -> Lane Change Right
  */
  vector<string> getNextStates(string current_state);
  Prediction prediction_obj;
};

#endif // VEHICLE_H
