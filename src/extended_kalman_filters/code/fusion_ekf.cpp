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

  // Acceleration Noise Component
  noise_ax = 5;
  noise_ay = 5;

  // Initialize Laser input params
  x_laser_ = Eigen::VectorXd(4, 1);
  F_laser_ = Eigen::MatrixXd(4, 4);
  P_laser_ = Eigen::MatrixXd(4, 4);
  P_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0,
              0, 0, 1000, 0,
              0, 0, 0, 1000;
  Q_laser_ = Eigen::MatrixXd(4, 4);
}

FusionEKF::~FusionEKF(){}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_package){
  float dt;

  // Caltulate the change in time
  dt = (measurement_package.timestamp_ - previous_timestamp_) / 1000000.0;
  float dt_2 = dt * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3 * dt;
  if (measurement_package.sensor_type_ == MeasurementPackage::LASER){
      cout << "LIDAR ==> " << "\n";
      cout << '\t' << "curr_t == " << measurement_package.timestamp_ << "\n";
      cout << '\t' << "prev_t == " << previous_timestamp_ << "\n";
      cout << '\t' << "delta_t == " << dt << "\n";

      x_laser_ << measurement_package.raw_measurements_[0],
                  measurement_package.raw_measurements_[1],
                  0,
                  0;
      F_laser_ << 1, 0, dt, 0,
                  0, 1, 0, dt,
                  0, 0, dt, 0,
                  0, 0, 0, dt;

      // TODO: THE P_laser_ and noise_ax, noise_ay values will change as time progresses
      Q_laser_ << dt_4/4*noise_ax, 0, dt_3/2*noise_ax, 0,
                  0, dt_4/4*noise_ay, 0, dt_3/2*noise_ay,
                  dt_3/2*noise_ax, 0, dt_2*noise_ax, 0,
                  0, dt_3/2*noise_ay, 0, dt_2*noise_ay;



      kf.Init(x_laser_, F_laser_, P_laser_, Q_laser_, R_laser_, H_laser_);
      // kf.predict
      // kf.update
  }
  else if (measurement_package.sensor_type_ == MeasurementPackage::RADAR){
      cout << "090909090900909090990909009090090990" << "\n";
  }
  else{
    cout << "Sensor type = " << measurement_package.sensor_type_ << " not known" << "\n";
  }

  previous_timestamp_ = measurement_package.timestamp_;

}
