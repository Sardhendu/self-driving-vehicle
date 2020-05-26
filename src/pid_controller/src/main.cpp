#include <math.h>
#include <uWS/uWS.h>
#include <iostream>
#include <string>
#include <fstream>
#include "json.hpp"
#include "PID.h"

// for convenience
using nlohmann::json;
using std::string;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_last_of("]");
  if (found_null != string::npos) {
    return "";
  }
  else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 1);
  }
  return "";
}

int main() {
  uWS::Hub h;

  PID pid;
  /**
   * TODO: Initialize the pid variable.
   */

  std::ofstream out_prediction_file;
  out_prediction_file.open("../files/gt_prediction.txt");

  out_prediction_file << "cte," << "steer_angle," << "kp," << "ki," << "kd," << "p_error,"  << "i_error," << "d_error," << "dkp," << "dki," << "dkd" << "\n";
  h.onMessage([&pid, &out_prediction_file](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {
      auto s = hasData(string(data).substr(0, length));

      if (s != "") {
        auto j = json::parse(s);

        string event = j[0].get<string>();

        if (event == "telemetry") {
          // j[1] is the data JSON object
          double cte = std::stod(j[1]["cte"].get<string>());
          double speed = std::stod(j[1]["speed"].get<string>());
          double angle = std::stod(j[1]["steering_angle"].get<string>());
          double steer_angle = 0.0;
          double throttle_value;
          /**
           * TODO: Calculate steering value here, remember the steering value is
           *   [-1, 1].
           * NOTE: Feel free to play around with the throttle and speed.
           *   Maybe use another PID controller to control the speed!
           */
           vector<double> params = pid.getParams();
           double kp = params[0];
           double ki = params[1];
           double kd = params[2];
           double p_error = params[3];
           double i_error = params[4];
           double d_error = params[5];
           double dkp = params[6];
           double dki = params[7];
           double dkd = params[8];

           steer_angle = pid.calculateSteeringValue();
           pid.UpdateError(cte);

           throttle_value = 0.75 - kp * p_error - kd * d_error - ki * i_error;
           std::cout <<" cte = "<< cte << " speed = " << speed << " angle = "
           << angle << " steer_angle = " << steer_angle << " throttle_value = "<< throttle_value << "\n";

           json msgJson;
           msgJson["steering_angle"] = steer_angle;
           msgJson["throttle"] = 0.3;

           out_prediction_file << cte << "," << steer_angle << ","
           << kp << "," << ki << "," << kd << ","
           << p_error << "," << i_error << "," << d_error << ","
           << dkp << "," << dki << "," << dkd << "\n";

          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          std::cout << msg << std::endl;
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }  // end "telemetry" if
      } else {
        // Manual driving
        string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }  // end websocket message if
  }); // end h.onMessage

  out_prediction_file.close();
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