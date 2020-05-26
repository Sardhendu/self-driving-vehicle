#ifndef PID_H
#define PID_H
#include <vector>

using std::vector;

class PID {
 public:
  /**
   * Constructor
   */
  PID();

  /**
   * Destructor.
   */
  virtual ~PID();

  /**
   * Initialize PID.
   * @param (Kp_, Ki_, Kd_) The initial PID coefficients
   */
//  void Init(double Kp_, double Ki_, double Kd_);

  /**
   * Update the PID error variables given cross track error.
   * @param cte The current cross track error
   */
  void UpdateError(double cte);
  vector<double> getParams();

  /**
   * Calculate the total PID error.
   * @output The total PID error
   */
//  double TotalError();

    /*
    calculateSteeringValue: Calculate the ste
    */

    double calculateSteeringValue();
    vector<double> twiddleIt(double param, double delta);

 private:
  /**
   * PID Errors
   */
  double p_error=0;
  double i_error=0;
  double d_error;

  /**
   * PID Coefficients
   */
  double Kp=0.062;
  double Ki=0.0003;
  double Kd=1.3;
  int param_index = 2;

  /*
  * Parameter Updates:
  */
  double dkp = 0.1;
  double dki = 0.1;
  double dkd = 0.1;

  int tried_incrementing = 0;
  int tried_decrementing = 0;

  /*
  * Average CrossTrackError
  */
  int time_step=0;
  double least_error = 99999;
  double total_cte = 0;
  /**/


  bool twiddle_it = false;

};

#endif  // PID_H