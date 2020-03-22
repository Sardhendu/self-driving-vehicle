#ifndef FUSION_EKF_H
#define FUSION_EKF_H

#include <iostream>
#include "Eigen/Dense"
#include "measurement_package.h"
#include "kalman_filter.h"

class FusionEKF{
  private:
    long long previous_timestamp_;
    Eigen::MatrixXd R_laser_;
    Eigen::MatrixXd R_radar_;
    Eigen::MatrixXd H_laser_;

    // acceleration noise
    float noise_ax;
    float noise_ay;

    // Declare Lidar params
    Eigen::VectorXd x_laser_;
    Eigen::MatrixXd F_laser_;
    Eigen::MatrixXd P_laser_;
    Eigen::MatrixXd Q_laser_;

  public:
    // Create a constructor to store the default values
    FusionEKF();

    // Create a desctructor
    virtual ~FusionEKF();

    void ProcessMeasurement(const MeasurementPackage &meas_package_);

    KalmanFilter kf;

};

#endif
