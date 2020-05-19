#include <iostream>
#include <math.h>

using std::abs;


inline double logistic(double x){
  // A function that returns a value between 0 and 1 for x in the range[0, infinity] and - 1 to 1 for x in
  // the range[-infinity, infinity]. Useful for cost functions.
  return 2.0 / (1 + exp(-x)) - 1.0;
}


// inline double goalDistanceCost(
//   int goal_lane, int car_lane, int intended_car_lane, doule distance_to_goal
// ){
//   /*
//     When our vehicle is away from the goal lane (end point) then the cost should increase.
//       -> If our current car_s is near to goal_s then the cost should be higher.
//       -> If our current car_s is far from goal_s then the cost should be lower
//
//     When our vehicle is in the goal lane (end point) then the cost should decrease.
//       -> If our current car_s is near to goal_s then the cost should be lower.
//       -> If our current car_s is far from goal_s then the cost should be lower but relatively not so much
//
//     [THE COSR SHOULD BE SCLAED BY DISTANCE BETWEEN THE OUR CURRENT car_s and goal_s]
//   */
//   double cost;
//   distance_to_goal = abs(distance_to_goal);
//
//   if (distance_to_goal != 0){
//     double lane_cost = 2.0*goal_lane - intended_lane - car_lane;
//     cost = 1 - 2*exp(-1 * abs(lane_cost/distance_to_goal))
//   }
//   else{
//     cost = 1.0;
//   }
//
//   return cost;
//
// }


inline vector<double> laneCost(
  map<int, vector<Traffic>> traffic_ahead, double car_s
){
  std::cout << "[LANE TRAFFIC COST]" <<"\n";

  int LANE_DISTANCE_FOR_TRAFFIC = car_s + 150; // Basically we look 150 m ahead to see which lane has least traffic
  double MAX_SPEED = 22;

  std::cout << "\ttraffic_distance = "<<LANE_DISTANCE_FOR_TRAFFIC << "\n";

  vector<int> lanes = {0, 1, 2};
  vector<double> lane_traffic_time;
  vector<double> lane_priors_score = {1.0, 1.0, 1.0};
  double total_time = 0;
  // Here we can also add priors for each lane. Say we know from previous Data
  // that a particular lane has more traffic at a particular time. So add the prior
  // to avoid that lane
  for (int i =0 ; i<lanes.size(); i++){
    int curr_lane = lanes[i];
    double avg_speed = 0;
    // double max_dist = -9999;
    double lane_traffic = 1;

    if (traffic_ahead.count(curr_lane) > 0){
      vector<Traffic> traffic_ahead_in_lane = traffic_ahead[curr_lane];

      double nearest_vehicle_s = 99999;
      for (int i=0; i<traffic_ahead_in_lane.size(); i++){
        avg_speed += traffic_ahead_in_lane[i].speed;

        if (traffic_ahead_in_lane[i].s < nearest_vehicle_s){
          nearest_vehicle_s = traffic_ahead_in_lane[i].s;
        }
      }
      avg_speed /= traffic_ahead_in_lane.size();
      double time_ = (LANE_DISTANCE_FOR_TRAFFIC - nearest_vehicle_s) / avg_speed;
      total_time += time_;
      lane_traffic_time.push_back(time_);
    }
    else{
      // When no traffic we can drive with full speed and assume we reach in 0 time
      lane_traffic_time.push_back(0);
    }
  }


  for (int i=0 ; i<lane_traffic_time.size(); i++){
    std::cout << "\tlane_num = " << i << " total_cost = "<< lane_traffic_time[i] << " norm_cost = "<< lane_traffic_time[i] << "\n";
    lane_traffic_time[i] /= (total_time+0.0000001);
  }
  return lane_traffic_time;
}

// double inefficiencyCost(
//   int intended_lane, int car_lane, Traffic vehicle_ahead
// ){
//   /*
//
//   It is very important to find the lane speed based on the vehicle's in that lane
//   because
//     1. We dont want to get stuck in a slow lane
//     2. We want to reach our goal as soon as possible
//   */
//
// }
