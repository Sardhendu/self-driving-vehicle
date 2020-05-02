#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <string>
#include <map>
#include <vector>
#include "Eigen/Dense"

using namespace std;
using Eigen::ArrayXd;
using std::string;
using std::vector;

class GNB {
 map<string, double> class_prior;

 public:
  /**
   * Constructor
   */
  GNB();

  /**
   * Destructor
   */
  virtual ~GNB();

  /**
   * Train classifier
   */
  void train(const vector<vector<double>> &data,
             const vector<string> &labels);

  /**
   * Predict with trained classifier
   */
  string predict(const vector<double> &sample);

  vector<string> possible_labels = {"left","keep","right"};
};

#endif  // CLASSIFIER_H
