#include "kalman_filter.h"
#include <cmath>
// #include "Eigen/Dense"
using Eigen::MatrixXd;
using Eigen::VectorXd;
#include <iostream>
using namespace std;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(
  VectorXd &x_in,  // Priors/Input/Updates
  MatrixXd &F_in,  // The transition matrix`
  MatrixXd &P_in,  // The covariance matrix (Captures uncertainity in the input x_in)
  MatrixXd &Q_in,  // The process noise matrix (Captures the uncertainity in the objects veclocity wrt to change in time -> (acceleration))
  MatrixXd &R_in,  // The measurement noise (Captures the uncertainity wrt to measurement) well sensors are not perfect.
  MatrixXd &H_in   // The Measurement transition matrix (basically transforms the measurement vector z into the prediction x space )
){
  x_ = x_in;
  F_ = F_in;
  P_ = P_in;
  Q_ = Q_in;
  R_ = R_in;
  H_ = H_in;
}

void KalmanFilter::Predict(){
  x_ = F_* x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const Eigen::VectorXd &z_in){
  /*
  Update Function for Linear Model (Laser or Lidar):
    Here we update the Object position based on input measurement for LIDAR
  */
  cout << "Updating For LIDAR ============> " << "\n";

  // Fetch the error
  VectorXd z_pred = H_ * x_;
  VectorXd y_ = z_in - z_pred;
  if (z_pred.size() != z_in.size()){
    cout << "z_pred = " << z_pred.size() << " != " << "z_in = " << z_in.size() << "\n";
    exit(0);
  }
  UpdateWithY(y_);

}


void KalmanFilter::UpdateEKF(const Eigen::VectorXd &z_in){
  /*
  Update Function the Non-Linear Model (Radar):
    Here we update the Object position based on input measurement from RADAR.
  */
  cout << "Updating for RADAR ============> " << "\n";
  // Convert the prediction cartesian coordinate to polar coordinate
  double px = x_(0);
  double py = x_(1);
  double vx = x_(2);
  double vy = x_(3);

  double rho = sqrt(px*px + py*py);
  double theta = atan2(py, px);
  cout << "theta == " << theta << "\n";
  if (theta > M_PI || theta < -M_PI){
    cout << "theta == " << theta << "\n";
    exit(0);
  }
  double rho_dot = (px*vx + py*vy) / rho;

  // Fetch the error
  VectorXd h = VectorXd(3);
  h << rho, theta, rho_dot;

  if (h.size() != z_in.size()){
    cout << "h = " << h.size() << " != " << "z_in = " << z_in.size() << "\n";
    exit(0);
  }

  VectorXd y_ = z_in - h;
  UpdateWithY(y_);

}


void KalmanFilter::UpdateWithY(const VectorXd &y_in){
  cout << "H_ = " << H_.size() << "\n";
  MatrixXd H_T = H_.transpose();
  MatrixXd S_ = H_ * P_ * H_T + R_;
  MatrixXd K_ =  P_ * H_T * S_.inverse();

  // New state
  x_ = x_ + (K_ * y_in);
  int x_size = x_.size();
  MatrixXd I_ = MatrixXd::Identity(x_size, x_size);
  P_ = (I_ - K_ * H_) * P_;
}
