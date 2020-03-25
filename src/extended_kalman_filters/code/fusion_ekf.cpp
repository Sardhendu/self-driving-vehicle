#include <cmath>
#include "Eigen/Dense"
#include "fusion_ekf.h"
// #include "kalman_filter.h"

using namespace std;

FusionEKF::FusionEKF(){
  // is_initialized is required to initialize the x_ vector for the very first datapoint
  is_initialized_ = false;
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
  H_radar_ = Eigen::MatrixXd(3, 4);

  // Acceleration Noise Component
  noise_ax = 5;
  noise_ay = 5;

  // Initialize Laser input params
  x_ = Eigen::VectorXd(4, 1);
  F_ = Eigen::MatrixXd(4, 4);
  P_ = Eigen::MatrixXd(4, 4);
  P_ << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1000, 0,
        0, 0, 0, 1000;
  Q_ = Eigen::MatrixXd(4, 4);
}

FusionEKF::~FusionEKF(){}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_package){
  float dt;

  // Caltulate the change in time
  dt = (measurement_package.timestamp_ - previous_timestamp_) / 1000000.0;
  float dt_2 = dt * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3 * dt;

  // ------------------------------------------------------------------
  // Prediction State
  // ------------------------------------------------------------------
  cout << "LIDAR ==> " << "\n";
  cout << '\t' << "curr_t == " << measurement_package.timestamp_ << "\n";
  cout << '\t' << "prev_t == " << previous_timestamp_ << "\n";
  cout << '\t' << "delta_t == " << dt << "\n";

  F_ << 1, 0, dt, 0,
        0, 1, 0, dt,
        0, 0, dt, 0,
        0, 0, 0, dt;

  // TODO: THE P_laser_ and noise_ax, noise_ay values will change as time progresses
  Q_ << dt_4/4*noise_ax, 0, dt_3/2*noise_ax, 0,
        0, dt_4/4*noise_ay, 0, dt_3/2*noise_ay,
        dt_3/2*noise_ax, 0, dt_2*noise_ax, 0,
        0, dt_3/2*noise_ay, 0, dt_2*noise_ay;

  // ------------------------------------------------------------------
  // Initialize Predict and Update State
  // ------------------------------------------------------------------
  if (measurement_package.sensor_type_ == MeasurementPackage::LASER){
      if (!is_initialized_){
        x_ << measurement_package.raw_measurements_[0],
              measurement_package.raw_measurements_[1],
              0,
              0;
        is_initialized_ = true;
        return;
      }
      cout << '\t' << "[Before] px = " << x_[0]  << " py = "<< x_[1] <<"\n";
      cout << '\t' << "[Before] P_ == " << "\n" << P_ << "\n";
      kf.Init(x_, F_, P_, Q_, R_laser_, H_laser_);
      kf.Predict();
      kf.Update(measurement_package.raw_measurements_);
      x_ << kf.x_;
      P_ << kf.P_;
      cout << '\t' << "[After] px = " << x_[0]  << " py = "<< x_[1] <<"\n";
      cout << '\t' << "[After] vx = " << x_[2]  << " vy = "<< x_[3] <<"\n";
      cout << '\t' << "[After] P_ == " << "\n" << P_ << "\n";
  }
  else if (measurement_package.sensor_type_ == MeasurementPackage::RADAR){
      // if (!is_initialized_){
      //   x_ << measurement_package.raw_measurements_[0],
      //         measurement_package.raw_measurements_[1],
      //         0,
      //         0;
      //   is_initialized_ = true;
      //   return;
      // }
      float px = x_[0];
      float py = x_[1];
      float vx = x_[2];
      float vy = x_[3];
      float c1 = pow(px, 2) + pow(py, 2);
      float c2 = sqrt(c1);
      float c3 = c1*c2;
      H_radar_ << (px/c2), (py/c2), 0, 0,
                  -(py/c1), (px/c1), 0, 0,
                  py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

      kf.Init(x_, F_, P_, Q_, R_radar_, H_radar_);
      kf.Update(measurement_package.raw_measurements_);
      x_ << kf.x_;
      P_ << kf.P_;
      cout << H_radar_ << "\n";
  }
  else{
    cout << "Sensor type = " << measurement_package.sensor_type_ << " not known" << "\n";
  }

  previous_timestamp_ = measurement_package.timestamp_;

}
