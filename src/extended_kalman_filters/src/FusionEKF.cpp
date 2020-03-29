#include <cmath>
#include "Eigen/Dense"
#include "FusionEKF.h"

// #include "kalman_filter.h"

using namespace std;
using Eigen::VectorXd;
using Eigen::MatrixXd;

FusionEKF::FusionEKF(){
  // is_initialized is required to initialize the x_ vector for the very first datapoint
  is_initialized_ = false;
  previous_timestamp_ = 0;

  // Initialize Matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);

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
  H_radar_ = MatrixXd(3, 4);

  // Acceleration Noise Component
  noise_ax = 9.0;
  noise_ay = 9.0;

  // Initialize Laser input params
  // x_ = Eigen::VectorXd(4, 1);
  // F_ = Eigen::MatrixXd(4, 4);
  // Q_ = Eigen::MatrixXd(4, 4);


}

FusionEKF::~FusionEKF(){}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_package){
  // ------------------------------------------------------------------
  // Initialize States
  // ------------------------------------------------------------------
  if (!is_initialized_){
    cout << "Initializing States =============>" << "\n";
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;
    ekf_.P_ = MatrixXd(4, 4);
    ekf_.P_ << 1, 0, 0, 0,
              0, 1, 0, 0,
              0, 0, 1000, 0,
              0, 0, 0, 1000;

    previous_timestamp_ = measurement_package.timestamp_;

    if (measurement_package.sensor_type_ == MeasurementPackage::LASER){
        cout << "[Initializing] px = " << ekf_.x_[0]  << " py = "<< ekf_.x_[1] <<"\n";
        cout << "[Initializing] P_ == " << "\n" << ekf_.P_ << "\n";

        ekf_.x_ <<  measurement_package.raw_measurements_[0],
                  measurement_package.raw_measurements_[1],
                  0,
                  0;
    }
    else {
        //TODO: Convert the polar coordinate system to cartesian coordinate space
        double rho = measurement_package.raw_measurements_[0];
        double phi = measurement_package.raw_measurements_[1];
        double rho_dot = measurement_package.raw_measurements_[2];

        double x = rho * cos(phi);
        // if ( x < 0.0001 ) {
        //   x = 0.0001;
        // }
    	  double y = rho * sin(phi);
        // if ( y < 0.0001 ) {
        //   y = 0.0001;
        // }
    	  double vx = rho_dot * cos(phi);
    	  double vy = rho_dot * sin(phi);
        ekf_.x_ << x, y, vx , vy;
    }

    is_initialized_ = true;
    return;
  }

  cout << "[Prior]" << " x = \n" << ekf_.x_ << "\n";
  cout << "[Prior]" << " P = \n" << ekf_.P_ << "\n";

  // ------------------------------------------------------------------
  // Prediction Phase
  // ------------------------------------------------------------------
  // Calculate the new matrices
  double dt;
  dt = (measurement_package.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_package.timestamp_;

  // Initialize the Transition Matrix
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, dt, 0,
             0, 1, 0, dt,
             0, 0, 1, 0,
             0, 0, 0, 1;



  // Initialize the Process Noise
  double dt_2 = dt * dt;
  double dt_3 = dt_2 * dt;
  double dt_4 = dt_3 * dt;
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << dt_4/4*noise_ax, 0, dt_3/2*noise_ax, 0,
            0, dt_4/4*noise_ay, 0, dt_3/2*noise_ay,
            dt_3/2*noise_ax, 0, dt_2*noise_ax, 0,
            0, dt_3/2*noise_ay, 0, dt_2*noise_ay;


  // cout << "Timestep ==> " << "\n";
  // cout << '\t' << "prev_t == " << previous_timestamp_ << "\n";
  // cout << '\t' << "curr_t == " << measurement_package.timestamp_ << "\n";
  // cout << '\t' << "delta_t == " << dt << "\n";

  ekf_.Predict();
  // cout << "[Prediction]" << " x = \n" << ekf_.x_ << "\n";
  // cout << "[Prediction]" << " P = \n" << ekf_.P_ << "\n";

  // ------------------------------------------------------------------
  // Update Phase
  // ------------------------------------------------------------------
  if (measurement_package.sensor_type_ == MeasurementPackage::LASER){
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_package.raw_measurements_);
  }
  else {
    H_radar_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.H_ = H_radar_;
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_package.raw_measurements_);
  }


  // cout << "[Update]" << " x = \n" << ekf_.x_ << "\n";
  // cout << "[Update]" << " P = \n" << ekf_.P_ << "\n";

}
