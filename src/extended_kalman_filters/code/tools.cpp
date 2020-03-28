#include "tools.h"
#include <iostream>
#include "Eigen/Dense"

using namespace std;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &y_pred, const vector<VectorXd> &y_true){
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  if (y_true.size() != y_pred.size() || y_pred.size() == 0){
    cout << "y_true=" << y_true.size() << " != " << "y_pred=" << y_pred.size() << "\n";
    return rmse;
  }
  else{
    cout << "y_true=" << y_true.size() << " == " << "y_pred=" << y_pred.size() << "\n";
  }

  for (unsigned int i=0; i < y_pred.size(); ++i){
    VectorXd residual = y_pred[i] - y_true[i];
    residual = residual.array()*residual.array();
    rmse += residual;
  }

  rmse = rmse/y_pred.size();
  rmse = rmse.array().sqrt();
  return rmse;
}
