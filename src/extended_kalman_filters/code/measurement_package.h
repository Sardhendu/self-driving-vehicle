#ifndef MEASUREMENT_PACKAGE_H
#define MEASERUMENT_PACKAGE_H

#include "Eigen/Dense"

class MeasurementPackage{
  public:
    enum SensorType{
      LASER,
      RADAR
    } sensor_type_;

    long long timestamp_;

    Eigen::VectorXd raw_measurements_;
};

#endif
