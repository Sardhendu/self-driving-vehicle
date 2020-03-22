#include <iostream>
// #include <uWS/uWS.h>
#include <math.h>
#include <string>
#include <vector>
#include <iomanip>
#include <sstream>
#include <fstream>

#include "Eigen/Dense"
#include "utils.hpp"
#include "parser.h"
#include "fusion_ekf.h"


using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::string;
using std::vector;

using namespace std;


int main(){
  cout << "Hello World" << "\n";
  string filepath = "../data/obj_pose-laser-radar-synthetic-input.txt";
  fstream inFile;

  // datamembers to read json data
  string sensor_vals;
  string token;
  vector<string> tokens;

  inFile.open(filepath);
  if (!inFile) {
    cout << "Unable to open infile @" << filepath << "\n";
    exit(1);
  }

  MeasurementParser meas_parser;
  MeasurementPackage meas_package;
  FusionEKF ekf;
  while (getline(inFile, sensor_vals, '\n')){
    tokens.clear();
    istringstream iss(sensor_vals);
    while (getline(iss, token, '\t')){
      tokens.push_back(token);
    }

    cout << sensor_vals << "\n";
    meas_parser.setMeasurements(tokens);
    cout << "--------------------------------" << "\n";
    vector<float> meas_px_py = meas_parser.getMeasurements();
    printInfo(tokens);

    // Set the sensor input to the MeasurementPackage
    if (meas_parser.sensor_type == "L"){
      meas_package.sensor_type_ = MeasurementPackage::LASER;
      meas_package.raw_measurements_ = VectorXd(2);
      meas_package.raw_measurements_ << meas_px_py[0], meas_px_py[1];
      meas_package.timestamp_ = meas_parser.getTimestamp();
    }
    else if (meas_parser.sensor_type == "R"){
      meas_package.sensor_type_ = MeasurementPackage::RADAR;
      meas_package.raw_measurements_ = VectorXd(3);
      meas_package.raw_measurements_ << meas_px_py[0], meas_px_py[1], meas_px_py[3];
      meas_package.timestamp_ = meas_parser.getTimestamp();
    }
    else{
      cout << "sensor_type = " << meas_parser.sensor_type << "not Understood" << "\n";
      exit(1);
    }


    ekf.ProcessMeasurement(meas_package);
    cout << "++++++++++++++++++++++++++++++++" << "\n";




  }

  inFile.close();
  // if (length && length > 2 && data[0] == '4' && data[1] == '2'){
  //   auto s = hasData(string(data));
  // }
  return 0;
}
