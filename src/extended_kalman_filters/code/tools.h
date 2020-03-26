#ifndef TOOLS_H_
#define TOOLS_H_

#include <vector>
#include "Eigen/Dense"

class Tools{
  public:
    Tools();
    virtual ~Tools();

    Eigen::VectorXd CalculateRMSE(const std::vector<Eigen::VectorXd> &y_pred,
                                  const std::vector<Eigen::VectorXd> &y_true);
};

#endif
