#include <iostream>
#include <math.h>

using std::abs

double goalDistanceCost(
  int goal_lane, int car_lane, int intended_car_lane, doule distance_to_goal
){
  /*
    When our vehicle is away from the goal lane (end point) then the cost should increase.
      -> If our current car_s is near to goal_s then the cost should be higher.
      -> If our current car_s is far from goal_s then the cost should be lower

    When our vehicle is in the goal lane (end point) then the cost should decrease.
      -> If our current car_s is near to goal_s then the cost should be lower.
      -> If our current car_s is far from goal_s then the cost should be lower but relatively not so much

    [THE COSR SHOULD BE SCLAED BY DISTANCE BETWEEN THE OUR CURRENT car_s and goal_s]
  */
  double cost;
  distance_to_goal = abs(distance_to_goal);

  if (distance_to_goal != 0){
    double lane_cost = 2.0*goal_lane - intended_lane - car_lane;
    cost = 1 - 2*exp(-1 * abs(lane_cost/distance_to_goal))
  }
  else{
    cost = 1.0;
  }

  return cost;

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
