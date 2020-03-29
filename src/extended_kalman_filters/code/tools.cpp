#include "tools.h"
#include <iostream>
#include "Eigen/Dense"

using namespace std;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    VectorXd rmse(4);
    rmse << 0,0,0,0;

    if(estimations.size() == 0){
      cout << "ERROR - CalculateRMSE () - The estimations vector is empty" << endl;
      return rmse;
    }

    if(ground_truth.size() == 0){
      cout << "ERROR - CalculateRMSE () - The ground-truth vector is empty" << endl;
      return rmse;
    }

    unsigned int n = estimations.size();
    if(n != ground_truth.size()){
      cout << "ERROR - CalculateRMSE () - The ground-truth and estimations vectors must have the same size." << endl;
      return rmse;
    }

    for(unsigned int i=0; i < estimations.size(); ++i){
      VectorXd diff = estimations[i] - ground_truth[i];
      diff = diff.array()*diff.array();
      rmse += diff;
    }

    rmse = rmse / n;
    rmse = rmse.array().sqrt();
    return rmse;
}


MatrixXd Tools::CalculateJacobian(const VectorXd &prediction_vector){

  MatrixXd Hj_ (3, 4);

  // Perform Sanity check
  if (prediction_vector.size() < 4){
    return Hj_;
  }

  double px = prediction_vector[0];
  double py = prediction_vector[1];
  double vx = prediction_vector[2];
  double vy = prediction_vector[3];

  double c1 = pow(px, 2) + pow(py, 2);
  double c2 = sqrt(c1);
  double c3 = c1*c2;


  if(fabs(c1) < 0.0001){
  	cout << "ERROR - CalculateJacobian () - Division by Zero" << endl;
  	return Hj_;
  }
  Hj_ << (px/c2), (py/c2), 0, 0,
        -(py/c1), (px/c1), 0, 0,
          py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;
  return Hj_;
}
