#include <uWS/uWS.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"

#include "json.hpp"
#include "spline.h"

// #include "experiments.h"
#include "prediction.h"
#include "vehicle.h"
#include "experiments.h"

#include <typeinfo>


// for convenience
using nlohmann::json;
using std::string;
using std::vector;
using std::map;
// using namespace std;

string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_first_of("}");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}


int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  std::ifstream in_map_(map_file_.c_str(), std::ifstream::in);

  string line;
  while (getline(in_map_, line)) {
    std::istringstream iss(line);
    double x;
    double y;
    float s;
    float d_x;
    float d_y;
    iss >> x;
    iss >> y;
    iss >> s;
    iss >> d_x;
    iss >> d_y;
    map_waypoints_x.push_back(x);
    map_waypoints_y.push_back(y);
    map_waypoints_s.push_back(s);
    map_waypoints_dx.push_back(d_x);
    map_waypoints_dy.push_back(d_y);
  }

  // Prediction prediction_obj;
  Vehicle vehicle_obj;
  h.onMessage([
    // &prediction_obj,
    &vehicle_obj,
    &map_waypoints_x,
    &map_waypoints_y,
    &map_waypoints_s,
    &map_waypoints_dx,
    &map_waypoints_dy
    ](
      uWS::WebSocket<uWS::SERVER> ws,
      char *data,
      size_t length,
      uWS::OpCode opCode
    ) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);

        string event = j[0].get<string>();

        if (event == "telemetry") {
          // j[1] is the data JSON object

          // Main car's localization Data
          double car_x = j[1]["x"];         // Car x position in map coordinate
          double car_y = j[1]["y"];         // Car y position in map coordinate
          double car_s = j[1]["s"];         // Car s position in frenet coordinate
          double car_d = j[1]["d"];         // Car d position in frenet coordinate
          double car_yaw = j[1]["yaw"];     // Car yaw angle in the map
          double car_speed = j[1]["speed"]; // Car speed in miles/hr

          // Previous path data given to the Planner
          auto previous_path_x = j[1]["previous_path_x"];
          auto previous_path_y = j[1]["previous_path_y"];
          // Previous path's end s and d values
          double end_path_s = j[1]["end_path_s"];
          double end_path_d = j[1]["end_path_d"];


          // Sensor Fusion Data, a list of all other cars on the same side
          //   of the road.
          auto sensor_fusion = j[1]["sensor_fusion"];
          // std::cout << "sensor_fusion : "<< "\n" << sensor_fusion << "\n";
          // std::cout << "TYPE: " << typeid(sensor_fusion).name() << endl;
          std::cout << "\n\n\n";
          std::cout << "#----------------------------------\n";
          std::cout << "# Initiating New Tmestep\n";
          std::cout << "#----------------------------------\n";

          std::cout << "SENSOR FUSION ===-=-=--=-==-=-";
          for (int v=0; v<sensor_fusion.size(); v++){
            std::cout << sensor_fusion[v] << " ";
          }
          std::cout << "\n";


          json msgJson;

          vector<vector<double>> sensor_fusion_data;
          for (int ii=0; ii<sensor_fusion.size(); ii++){
            vector<double> my_vector {
              sensor_fusion[ii][0],
              sensor_fusion[ii][1],
              sensor_fusion[ii][2],
              sensor_fusion[ii][3],
              sensor_fusion[ii][4],
              sensor_fusion[ii][5],
              sensor_fusion[ii][6]
            };
            sensor_fusion_data.push_back(my_vector);
          }

          vector<double> previous_path_x_;
          vector<double> previous_path_y_;
          for (int ii=0; ii<previous_path_x.size(); ii++){
            previous_path_x_.push_back(previous_path_x[ii]);
            previous_path_y_.push_back(previous_path_y[ii]);
          }


          // prediction_obj.setPredctions(sensor_fusion_data, "CS");
          vehicle_obj.setVehicle(
            car_x,
            car_y,
            car_s,
            car_d,
            car_yaw,
            car_speed,
            map_waypoints_s,
            map_waypoints_x,
            map_waypoints_y,
            sensor_fusion_data,
            end_path_s,
            end_path_d
          );
          std::cout << "car_s " << car_s << "\n";
          std::cout << "car_d " << car_d << "\n";
          vector<vector<double>> trajectoryXY = vehicle_obj.generateTrajectory(
            previous_path_x_,
            previous_path_y_
          );
          // Prediction prediction_dict;
          // vector<vector<double>> trajectoryXY = moveSmoothlyInOneLane(
          //   car_s, car_d, map_waypoints_s, map_waypoints_x, map_waypoints_y
          // );

          msgJson["next_x"] = trajectoryXY[0];//next_xy[0];
          msgJson["next_y"] = trajectoryXY[1];//next_xy[1];

          auto msg = "42[\"control\","+ msgJson.dump()+"]";

          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }  // end "telemetry" if
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }  // end websocket if
  }); // end h.onMessage

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }

  h.run();
}
