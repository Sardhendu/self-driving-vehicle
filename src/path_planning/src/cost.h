#include <iostream>
#include <math.h>
#include <vector>

using std::abs;



inline double logistic(double x){
  // A function that returns a value between 0 and 1 for x in the range[0, infinity] and - 1 to 1 for x in
  // the range[-infinity, infinity]. Useful for cost functions.
  return 2.0 / (1 + exp(-x)) - 1.0;
}

inline vector<double> normalize(vector<double> values){
  double total_sum = 0;
  for (int i=0; i<values.size(); i++){
    total_sum  += values[i];
  }

  for (int i=0; i<values.size(); i++){
    values[i] /= total_sum;
  }
  return values;
}


inline map<int, double> laneTrafficCost(
  map<int, vector<Traffic>> traffic_ahead, double car_s
){
  /*
  This Method returns the cost of travelling in a particular lane
  Uses Sensor data for about 200m ahead
  */

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
    double lane_traffic = 1;

    // If there is traffic agead in the lane
    if (traffic_ahead.count(curr_lane) > 0){
      vector<Traffic> traffic_ahead_in_lane = traffic_ahead[curr_lane];

      double nearest_vehicle_s = 99999;
      for (int j=0; j<traffic_ahead_in_lane.size(); j++){
        // Check if the vehicle ahead is in the range on Lane Traffic
        if (traffic_ahead_in_lane[j].s <= LANE_DISTANCE_FOR_TRAFFIC){
          avg_speed += traffic_ahead_in_lane[j].speed;

          if (traffic_ahead_in_lane[j].s < nearest_vehicle_s){
            nearest_vehicle_s = traffic_ahead_in_lane[j].s;
          }
        }

      }

      if (nearest_vehicle_s != 99999){
        avg_speed /= traffic_ahead_in_lane.size();
        double time_ = (LANE_DISTANCE_FOR_TRAFFIC - nearest_vehicle_s) / avg_speed;
        total_time += time_;
        lane_traffic_time.push_back(time_);
      }
      else{
        // When the vehicle is not in the traffic range then the cost for that lane is 0
        lane_traffic_time.push_back(0);
      }

    }
    else{
      // When no traffic we can drive with full speed and assume we reach in 0 time
      lane_traffic_time.push_back(0);
    }
  }

  map<int, double> lane_cost_dict;
  for (int i=0 ; i<lane_traffic_time.size(); i++){
    std::cout << "\tlane_num = " << i << " total_cost = "<< lane_traffic_time[i] << " norm_cost = "<< lane_traffic_time[i] << "\n";
    lane_cost_dict[i] = lane_traffic_time[i] / (total_time+0.0000001);
  }

  std::cout<< "\t[FINAL COST]\tlane=0 = " << lane_cost_dict[0] << "\tlane=1 = "<< lane_cost_dict[1] << "\tlane=2 = " << lane_cost_dict[2] << "\n";
  std::cout<<"\n";
  return lane_cost_dict;
}



// inline map<int, double> laneChangeCost(
//   map<int, vector<Traffic>> traffic_behind, map<int, vector<Traffic>> traffic_ahead, double car_s, double car_v
// ){
//   /*
//     Scenario: Say we are in lane=0 nad PLCR state. Say there is a vehicle in lane 1 about 10m behind us
//     So for the next lane:
//
//     Normally its not safe to switch lane in such a case because it can result in collision.
//       1. PLCR chooses lane=0 becasue it checks for lanes behind for a certain buffer
//       2. LCR will always choose lane=1
//
//      So, we still have some probability for the next state LCR to have lower cost than PLCR suing only the insufficiency_cost
//      and lane_traffic_cost. hence here we penalize any such decisions
//
//     One questions we may want to ask.
//     1. When is there a high chance of collision switching lane.
//       -> When the car is far but moving at a higher speed
//       -> When the car is very near to us and moving at the speed closer to us
//
//       So we want to penalize such occasions
//
//     A simple formula would be:
//
//     if other_car_s is within the buffer:
//       time_taken = (car_speed - other_vehicle_speed) / (car_s - other_car_s)
//       iime_taken  (-inf, inf), where
//
//     where -ve and +ve values closer to 0 have high chance of collision and hence should be penalized more.
//
//     */
//
//
//   int LANE_DISTANCE_FOR_TRAFFIC_BEHIND = car_s - 20;  // looking 50
//   vector<int> lanes = {0, 1, 2};
//
//   map<int, double> lane_change_cost;
//
//   double total_sum = 0;
//   for (int i =0 ; i<lanes.size(); i++){
//     int curr_lane = lanes[i];
//
//     // Run only when there is traffic behind
//     if (traffic_behind.count(curr_lane) > 0){
//       std::cout<<"Vehicle Behind for = " <<  curr_lane << "\n";
//       vector<Traffic> traffic_behind_in_lane = traffic_behind[curr_lane];
//
//       // Find the vehicle that is most closest to us
//       double nearest_vehicle_s = -99999;
//       for (int j=0; j<traffic_behind_in_lane.size(); j++){
//         // Check if the vehicle ahead is in the range on Lane Traffic
//         std::cout << "traffic_behind_in_lane = " << car_s << " " <<traffic_behind_in_lane[j].s << " " << traffic_behind_in_lane[j].speed << "\n";
//         if (traffic_behind_in_lane[j].s >= LANE_DISTANCE_FOR_TRAFFIC_BEHIND){
//           // std::cout << "HEHEHEHEHEHEHEHEHEHE " << "\n";
//           if (traffic_behind_in_lane[j].s > nearest_vehicle_s){
//             nearest_vehicle_s = traffic_behind_in_lane[j].s;
//           }
//         }
//         if (nearest_vehicle_s != -99999){
//           double time_to_reach_us = (car_s - traffic_behind_in_lane[j].s) / (car_v - traffic_behind_in_lane[j].speed + 0.000001);
//           lane_change_cost[i] = time_to_reach_us;
//           total_sum += time_to_reach_us;
//         }
//       }
//     }
//   }
//
//   // Normalize the values
//   if (total_sum > 0){
//     for (int i=0; i<lanes.size(); i++){
//       int curr_lane = lanes[i];
//
//         if (lane_change_cost.count(curr_lane) > 0){
//           lane_change_cost[i] /= total_sum;
//         }
//       }
//   }
//
//   // Compute the cost with exponential function
//   double total_sum1;
//   for (int i=0; i<lanes.size(); i++){
//     int curr_lane = lanes[i];
//     if (lane_change_cost.count(curr_lane) > 0){
//       std::cout<<"[laneChangeCost] "
//       << "\n\tlane = " << lanes[i] << " actual_cost = "<< lane_change_cost[i] << " exp_cost = " << exp(lane_change_cost[i]) << "\n";
//       lane_change_cost[i] = exp(lane_change_cost[i]);
//       total_sum1 += lane_change_cost[i];
//     }
//     else{
//       // Whwn no vehicle in lane the cost incured to lane change = 0
//       lane_change_cost[i] = 0;
//     }
//   }
//
//
//   // Normalze again
//   if (total_sum1 > 0){
//     for (int i=0; i<lanes.size(); i++){
//       int curr_lane = lanes[i];
//
//         if (lane_change_cost.count(curr_lane) > 0){
//           lane_change_cost[i] /= total_sum1;
//         }
//       }
//     }
//
//   std::cout<<"[laneChangeCost] "
//   << "\n\tlane=0 = " << lane_change_cost[0] << " lane=1 = "<< lane_change_cost[1] << " lane=2 = " << lane_change_cost[2] << "\n";
//   return lane_change_cost;
//
// }



inline map<int, double> laneChangeCost(
  map<int, vector<Traffic>> traffic_behind, map<int, vector<Traffic>> traffic_ahead, double car_s, double car_v
){
  /*
    Scenario: Say we are in lane=0 nad PLCR state. Say there is a vehicle in lane 1 about 10m behind us
    So for the next lane:

    Normally its not safe to switch lane in such a case because it can result in collision.
      1. PLCR chooses lane=0 becasue it checks for lanes behind for a certain buffer
      2. LCR will always choose lane=1

     So, we still have some probability for the next state LCR to have lower cost than PLCR suing only the insufficiency_cost
     and lane_traffic_cost. hence here we penalize any such decisions

    One questions we may want to ask.
    1. When is there a high chance of collision switching lane.
      -> When the car is far but moving at a higher speed
      -> When the car is very near to us and moving at the speed closer to us

      So we want to penalize such occasions

    A simple formula would be:

    if other_car_s is within the buffer:
      time_taken = (car_speed - other_vehicle_speed) / (car_s - other_car_s)
      iime_taken  (-inf, inf), where

    where -ve and +ve values closer to 0 have high chance of collision and hence should be penalized more.

    */


  int NEAREST_VEHICLE_BUFFER = 30;  // looking 50
  vector<int> lanes = {0, 1, 2};


  vector<Traffic> nearest_vehicle;
  for (int i =0 ; i<lanes.size(); i++){
    int curr_lane = lanes[i];
    double nearest_vehicle_s = 99999;
    Traffic nearest_vehicle_in_lane;

    // Run only when there is traffic behind
    if (traffic_behind.count(curr_lane) > 0){
      // Find the vehicle that is most closest to us
      vector<Traffic> traffic_behind_in_lane = traffic_behind[curr_lane];
      for (int j=0; j<traffic_behind_in_lane.size(); j++){
        // Check if the vehicle ahead is in the range on Lane Traffic
        // std::cout << "traffic_behind_in_lane = " << car_s << " " <<traffic_behind_in_lane[j].s << " " << traffic_behind_in_lane[j].speed << "\n";
        double vehicle_dist = abs(traffic_behind_in_lane[j].s - car_s);
        if (vehicle_dist < nearest_vehicle_s){
          nearest_vehicle_s = vehicle_dist;
          nearest_vehicle_in_lane = traffic_behind_in_lane[j];
        }
      }
    }

    // Run only when there is traffic ahead
    if (traffic_ahead.count(curr_lane) > 0){
      vector<Traffic> traffic_ahead_in_lane = traffic_ahead[curr_lane];
      for (int j=0; j<traffic_ahead_in_lane.size(); j++){
        // Check if the vehicle ahead is in the range on Lane Traffic
        double vehicle_dist = abs(traffic_ahead_in_lane[j].s - car_s);
        if (vehicle_dist < nearest_vehicle_s){
          nearest_vehicle_s = vehicle_dist;
          nearest_vehicle_in_lane = traffic_ahead_in_lane[j];
        }
      }
    }
    nearest_vehicle.push_back(nearest_vehicle_in_lane);
  }

  double total_time = 0;
  map<int, double> lane_change_cost;
  for (int i=0; i<nearest_vehicle.size(); i++){
    // Only if the vehicle is in the range
    double relative_dist = abs(nearest_vehicle[i].s - car_s);
    double relative_speed = abs(nearest_vehicle[i].speed - car_v);
    // Only when there is a vehicle in the lane
    if (nearest_vehicle[i].id != -1){
      if (relative_dist <= NEAREST_VEHICLE_BUFFER){
        lane_change_cost[i] = relative_dist/(relative_speed+0.000001);
        total_time += lane_change_cost[i];
      }
    }
  }

  std::cout<<"[LANE CHANGE COST] ";
  // Normalize the values
  double total_sum = 0;
  if (total_time > 0){
    for (int i=0; i<lanes.size(); i++){
      int curr_lane = lanes[i];
        if (lane_change_cost.count(curr_lane) > 0){
          std::cout << "\n\tlane = " << lanes[i] << " time_to_reach = "<< lane_change_cost[i] << " ";
          lane_change_cost[i] /= total_time;
          std::cout << " normed_time = " << lane_change_cost[i] << " ";
          lane_change_cost[i] = 1 - lane_change_cost[i];
          std::cout << " cost = " << lane_change_cost[i] << " ";
          total_sum += lane_change_cost[i];
        }
        else{
          // When no vehicle in lane the cost is 0
          lane_change_cost[i] = 0;
        }
      }
  }


  for (int i=0; i<lanes.size(); i++){
      lane_change_cost[i] /= (total_sum+0.000001);
  }


  std::cout<< "\n\t[FINAL COST]\tlane=0 = " << lane_change_cost[0] << "\tlane=1 = "<< lane_change_cost[1] << "\tlane=2 = " << lane_change_cost[2] << "\n";
  return lane_change_cost;

}
