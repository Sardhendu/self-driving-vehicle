#include <cmath>
#include "Eigen/Dense"
#include "fusion_ekf.h"
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
  H_radar_ = MatrixXd(3, 4);

  // Acceleration Noise Component
  noise_ax = 9;
  noise_ay = 9;

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
    kf.x_ = VectorXd(4);
    kf.x_ << 1, 1, 1, 1;
    kf.P_ = MatrixXd(4, 4);
    kf.P_ << 1, 0, 0, 0,
              0, 1, 0, 0,
              0, 0, 1000, 0,
              0, 0, 0, 1000;

    previous_timestamp_ = measurement_package.timestamp_;

    if (measurement_package.sensor_type_ == MeasurementPackage::LASER){
        cout << "[Initializing] px = " << kf.x_[0]  << " py = "<< kf.x_[1] <<"\n";
        cout << "[Initializing] P_ == " << "\n" << kf.P_ << "\n";

        kf.x_ <<  measurement_package.raw_measurements_[0],
                  measurement_package.raw_measurements_[1],
                  0,
                  0;
        // kf.Init(x_, F_, P_, Q_, R_laser_, H_laser_);
    }
    else {
      //TODO: Convert the polar coordinate system to cartesian coordinate space
        float s = 0;
    }

    is_initialized_ = true;
    return;
  }

  cout << "[Input]" << " x = \n" << kf.x_ << "\n";
  cout << "[Input]" << " P = \n" << kf.P_ << "\n";

  // ------------------------------------------------------------------
  // Prediction Phase
  // ------------------------------------------------------------------
  // Calculate the new matrices
  float dt;
  dt = (measurement_package.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_package.timestamp_;

  // Initialize the Transition Matrix
  kf.F_ = MatrixXd(4, 4);
  kf.F_ << 1, 0, dt, 0,
            0, 1, 0, dt,
            0, 0, dt, 0,
            0, 0, 0, dt;



  // Initialize the Process Noise
  float dt_2 = dt * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3 * dt;
  kf.Q_ = MatrixXd(4, 4);
  kf.Q_ << dt_4/4*noise_ax, 0, dt_3/2*noise_ax, 0,
            0, dt_4/4*noise_ay, 0, dt_3/2*noise_ay,
            dt_3/2*noise_ax, 0, dt_2*noise_ax, 0,
            0, dt_3/2*noise_ay, 0, dt_2*noise_ay;


  cout << "Timestep ==> " << "\n";
  cout << '\t' << "prev_t == " << previous_timestamp_ << "\n";
  cout << '\t' << "curr_t == " << measurement_package.timestamp_ << "\n";
  cout << '\t' << "delta_t == " << dt << "\n";

  kf.Predict();
  cout << "[Prediction]" << " x = \n" << kf.x_ << "\n";
  cout << "[Prediction]" << " P = \n" << kf.P_ << "\n";

  // ------------------------------------------------------------------
  // Update Phase
  // ------------------------------------------------------------------
  if (measurement_package.sensor_type_ == MeasurementPackage::LASER){
    kf.H_ = H_laser_;
    kf.R_ = R_laser_;
    kf.Update(measurement_package.raw_measurements_);
  }
  else {
      float px = kf.x_[0];
      float py = kf.x_[1];
      float vx = kf.x_[2];
      float vy = kf.x_[3];
      float c1 = pow(px, 2) + pow(py, 2);
      float c2 = sqrt(c1);
      float c3 = c1*c2;
      kf.H_ = MatrixXd(3, 4);

      //check division by zero
    	if(fabs(c1) < 0.0001){
    		cout << "ERROR - CalculateJacobian () - Division by Zero" << endl;
    		return Hj;
    	}
      else{
        kf.H_ << (px/c2), (py/c2), 0, 0,
                -(py/c1), (px/c1), 0, 0,
                  py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;
        
      }

      kf.R_ = R_radar_;
      kf.UpdateEKF(measurement_package.raw_measurements_);
  }


  cout << "[Update]" << " x = \n" << kf.x_ << "\n";
  cout << "[Update]" << " P = \n" << kf.P_ << "\n";

}
