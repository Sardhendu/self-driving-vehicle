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
#include "FusionEKF.h"
#include "tools.h"


using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::string;
using std::vector;

using namespace std;


int main(){
  cout << "Hello World" << "\n";
  string filepath = "../data/obj_pose-laser-radar-synthetic-input.txt";

  // Read the Input File
  fstream inFile;
  inFile.open(filepath);
  if (!inFile) {
    cout << "Unable to open infile @" << filepath << "\n";
    exit(1);
  }

  // Write to the output Files
  ofstream outFile_lidar;
  ofstream outFile_radar;
  outFile_lidar.open("../data/lidar_ekf_output.txt");
  outFile_radar.open("../data/radar_ekf_output.txt");

  // datamembers to read json data
  string sensor_vals;
  string token;
  vector<string> tokens;

  Tools tools;
  MeasurementParser meas_parser;
  MeasurementPackage meas_package;
  FusionEKF fusion_ekf;

  int rec_no = 0;
  vector<VectorXd> estimations;
  vector<VectorXd> ground_truth;

  while (getline(inFile, sensor_vals, '\n')){
    cout << "\n# --------------------------------------------------------------- \n ITERATION NUM =" << rec_no << "\n#---------------------------------------------------------------" << "\n";
    tokens.clear();
    istringstream iss(sensor_vals);
    while (getline(iss, token, '\t')){
      tokens.push_back(token);
    }

    meas_parser.setMeasurements(tokens);
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

    // Stash Groung Truth to ground_truth vector
    vector<float> gt_(4);
    gt_ = meas_parser.getGroundTruth();
    VectorXd gtruth(4);
    gtruth(0) = gt_[0];
    gtruth(1) = gt_[1];
    gtruth(2) = gt_[2];
    gtruth(3) = gt_[3];
    ground_truth.push_back(gtruth);

    // Stash Prediction to estimation vector
    fusion_ekf.ProcessMeasurement(meas_package);
    double p_x = fusion_ekf.ekf_.x_(0);
    double p_y = fusion_ekf.ekf_.x_(1);
    double v1  = fusion_ekf.ekf_.x_(2);
    double v2 = fusion_ekf.ekf_.x_(3);
    VectorXd estimate(4);
    estimate(0) = p_x;
    estimate(1) = p_y;
    estimate(2) = v1;
    estimate(3) = v2;
    estimations.push_back(estimate);
    // cout << "New Vals: " << p_x << " " << p_y << " "<< v1 << " " << v2 << "\n";

    VectorXd rmse(4);
    rmse = tools.CalculateRMSE(estimations, ground_truth);

    cout << "RMSE = ----------------------------------------> \n" << rmse << '\n';

    // Capture Prediction and Error Outputs into a File to compute statistics.

    if (rec_no == 0){
      outFile_lidar << "px_msmt,py_msmt,px_pred,py_pred,px_error,py_error,px_gt,py_gt,vx_gt,vy_gt,px_rmse,py_rmse,vx_rmse,vy_rmse" << "\n";
      outFile_radar << "rho_msmt,phi_msmt,rho_dot_msmt,rho_pred,phi_pred,rho_dot_pred,rho_error,phi_error,rho_dot_error,px_gt,py_gt,vx_gt,vy_gt,px_rmse,py_rmse,vx_rmse,vy_rmse" << "\n";
    }
    else{
      if (meas_parser.sensor_type == "L"){
        outFile_lidar << meas_package.raw_measurements_(0) << "," << meas_package.raw_measurements_(1) << ","
                      << fusion_ekf.ekf_.z_pred_(0) << "," << fusion_ekf.ekf_.z_pred_(1) << ","
                      << fusion_ekf.ekf_.y_(0) << "," << fusion_ekf.ekf_.y_(1) << ","
                      << gt_[0] << "," << gt_[1] << "," << gt_[2] << "," << gt_[3] << ","
                      << rmse(0) << "," << rmse(1) << "," << rmse(2) << "," << rmse(3)
                      << "\n";


      }
      else{
        outFile_radar << meas_package.raw_measurements_(0) << "," << meas_package.raw_measurements_(1) << "," << meas_package.raw_measurements_(2) << ","
                      << fusion_ekf.ekf_.z_pred_(0) << "," << fusion_ekf.ekf_.z_pred_(1) << "," << fusion_ekf.ekf_.z_pred_(2) << ","
                      << fusion_ekf.ekf_.y_(0) << "," << fusion_ekf.ekf_.y_(1) << "," << fusion_ekf.ekf_.y_(2) << ","
                      << gt_[0] << "," << gt_[1] << "," << gt_[2] << "," << gt_[3] << ","
                      << rmse(0) << "," << rmse(1) << "," << rmse(2) << "," << rmse(3)
                      << "\n";

      }

    }




    rec_no += 1;

  }

  inFile.close();
  outFile_lidar.close();
  outFile_radar.close();
  // if (length && length > 2 && data[0] == '4' && data[1] == '2'){
  //   auto s = hasData(string(data));
  // }
  return 0;
}


/*
1.08884
 1.0426
2.06104
2.11472
*/
