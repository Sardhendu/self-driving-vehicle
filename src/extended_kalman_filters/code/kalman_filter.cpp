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

// void KalmanFilter::Predict(){
//   MatrixXd x_pred();
// }
