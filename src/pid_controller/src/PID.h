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
  void Init(double Kp_, double Ki_, double Kd_);

  /**
   * Update the PID error variables given cross track error.
   * @param cte The current cross track error
   */
  void UpdateError(double cte);

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
  double Kp=0.2;
  double Ki=0.0004;
  double Kd=3;

  /*
  * Parameter Updates:
  */
  double dkp = 1;
  double dki = 1;
  double dkd = 1;

  int tried_incrementing = 0;
  int tried_decrementing = 0;

  /*
  * Average CrossTrackError
  */
  int time_step=0;
  double least_error = 99999;
  double total_cte = 0;
  /**/

};

#endif  // PID_H