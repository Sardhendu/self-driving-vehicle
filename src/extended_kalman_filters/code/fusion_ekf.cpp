#include "Eigen/Dense"
#include "fusion_ekf.h"
// #include "kalman_filter.h"

using namespace std;

FusionEKF::FusionEKF(){

  previous_timestamp_ = 0;

  // Initialize Matrices
  R_laser_ = Eigen::MatrixXd(2, 2);
  R_radar_ = Eigen::MatrixXd(3, 3);
  H_laser_ = Eigen::MatrixXd(2, 4);
  // Hj_ = MatrixXd(3, 4);

  // Measrement noise from the laser sensor
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  // Measurement noise from the radar sensor
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  // Transotion matrix: to transform the pos_and_velocity_vector to pos_vector
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  // TODO: Write the Transition matrix for Radar


}

FusionEKF::~FusionEKF(){}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_package){
  if (measurement_package.sensor_type_ == MeasurementPackage::LASER){
      cout << "ahahahahahahahahahahahahahahahahahha" << "\n";
      // kf_.Init()
  }
  else if (measurement_package.sensor_type_ == MeasurementPackage::RADAR){
      cout << "090909090900909090990909009090090990" << "\n";
  }
  else{
    cout << "Sensor type = " << measurement_package.sensor_type_ << " not known" << "\n";
  }

}
