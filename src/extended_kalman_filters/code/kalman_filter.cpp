#include "kalman_filter.h"
// #include "Eigen/Dense"
using Eigen::MatrixXd;
using Eigen::VectorXd;


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
  VectorXd z_pred = H_ * x_;
  MatrixXd H_T = H_.transpose();

  VectorXd y_ = z_in - z_pred;
  MatrixXd S_ = H_ * P_ * H_T + R_;
  MatrixXd K_ = P_ * H_T * S_.inverse();


  x_ = x_ + K_ * y_;
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K_ * H_) * P_;

}
