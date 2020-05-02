#include <iostream>
#include "classifier.h"
#include <math.h>
#include <string>
#include <vector>

using Eigen::ArrayXd;
using std::string;
using std::vector;

using namespace std;
// Initializes GNB
GNB::GNB() {
  /**
   * TODO: Initialize GNB, if necessary. May depend on your implementation.
   */

}

GNB::~GNB() {}

void GNB::train(const vector<vector<double>> &data,
                const vector<string> &labels) {
  /**
   * Trains the classifier with N data points and labels.
   * @param data - array of N observations
   *   - Each observation is a tuple with 4 values: s, d, s_dot and d_dot.
   *   - Example : [[3.5, 0.1, 5.9, -0.02],
   *                [8.0, -0.3, 3.0, 2.2],
   *                 ...
   *                ]
   * @param labels - array of N labels
   *   - Each label is one of "left", "keep", or "right".
   *
   * TODO: Implement the training function for your classifier.
   */
   // Get the prior probability of all the unique classes
   double sz = labels.size();
   for (int i =0; i<labels.size(); i++){
     cout << labels[i] << "\n";
     if (class_prior.find(labels[i]) == class_prior.end()){
        class_prior[labels[i]] = 1.0/sz;
     }
     else{
       class_prior[labels[i]] += 1.0/sz;
     }
   }


   for (auto itr = class_prior.begin(); itr != class_prior.end(); ++itr) {
        cout << itr->first
             << '\t' << itr->second << '\n';
    }
   int o = 9;

}

string GNB::predict(const vector<double> &sample) {
  /**
   * Once trained, this method is called and expected to return
   *   a predicted behavior for the given observation.
   * @param observation - a 4 tuple with s, d, s_dot, d_dot.
   *   - Example: [3.5, 0.1, 8.5, -0.2]
   * @output A label representing the best guess of the classifier. Can
   *   be one of "left", "keep" or "right".
   *
   * TODO: Complete this function to return your classifier's prediction
   */
   int p =9;

  return this -> possible_labels[1];
}
