#ifdef KALMAN_FILTER_H_
#define KALMAN_FILTER_H_

/*
Kalman filter is the central class
  1. Initializes the Matrices and Vectors
  2.
*/
class KalmanFilter{
  public:
    KalmanFilter(); // Create a constructor
    virtual ~KalamanFilter(); // Create a destructor

    // Function Declaration pass in the address reference instead of the actual values
    void Init(
      Eigen::VectorXd &x_in,  // Priors/Input/Updates
      Eigen::MatrixXd &F_in,  // The transition matrix`
      Eigen::MatrixXd &P_in   // The covariance matrix (Captures uncertainity in the input x_in)
      Eigen::MatrixXd &Q_in   // The process noise matrix (Captures the uncertainity in the objects veclocity wrt to change in time -> (acceleration))
      Eigen::MatrixXd &R_in   // The measurement noise (Captures the uncertainity wrt to measurement) well sensors are not perfect.
      Eigen::MatrixXd &H_in   // The Measurement transition matrix (basically transforms the measurement vector z into the prediction x space )
    );

    // Declare the Predict method
    void Predict();

    // Declare the Update Method
    void Update(const Eigen::VectorXd &z_in);

    // Define all the variables
    Eigen::VectorXd x_;
    Eigen::MatrixXd F_;
    Eigen::MatrixXd P_;
    Eigen::MatrixXd Q_;
    Eigen::MatrixXd R_;
    Eigen::MatrixXd H_;


};

#endif
