#ifndef PARTICLE_FILTER_H_
#define PARTICLE_FILTER_H_
#include "parser.h"

using namespace std;

struct Particle {
  /*
      The particle attributes define the transformatin matrix to transform
      an observation (landmark) in vehicle coordinate system to the landmark
      observation in map coordinate system.
        -- id
        -- x      (meters in x axis)  -- translation_x
        -- y      (meters in y axis)  -- translation_y
        -- theta  (orientation)       -- rotation_theta
        -- weight the closer the lankmark in vehicle coordinate are to the landmarks in map coordinate the higer the weights are
  */

  int id;
  double x;
  double y;
  double theta;
  double weight;

  std::vector<int> associations;
  std::vector<double> sense_x;
  std::vector<double> sense_y;
};


class ParticleFilter {
  private:
    int num_particles;
    bool is_initialized;

  public:
	vector<Particle> particles;
    // Define constructors
    ParticleFilter() : num_particles(0), is_initialized(false) {};
    ~ParticleFilter() {};

    const bool initialized() const {
		    return is_initialized;
	  }

    void print_particle_attributes(int particle_cnt);

    // ------------------------------------------------------------------------
    // Initialization
    /* Create random particles aroung the GPS location of the Car

      x:       initial GPS x-position to sample particles from
      y:       initial GPS x-position to sample particles from
      theta:   initial GPS x-position to sample particles from
      std:     (3) gaussian noise for sampling
    */
    void init(double gps_x, double gps_y, double theta, double std[]);


    // ------------------------------------------------------------------------
    // Prediction
    /*
      Prediction: we want to predict the vehicle position based on the observation
      from the previous timesteps
      delta_t:    time elapsed between two timestep
      velocity:   the velocity of car from t-1 to t in m/sec
      yaw_rate:   the yaw_rate of car from t-1 to t in rad/sec
      std:        the standard deviation to capture the control uncertainty
                  [std_x, std_y, std_yaw]

      We treat the particles as a guess for our vehicle's location in map coordinate frame.
      Here we simply use the velocity and yaw_rate of the vehicle to determine the new position
      of the particle.

      Some Caveats:
        1. A transformation is needed from behicle coordinate frame to map frame
        2. We resample partices before doing the prediction. This makes sure, we get the particlces most closest to the vehicle location
    */
    void prediction(double delta_t, double std[], double velocity, double yaw_rate);

    // ------------------------------------------------------------------------
    // Data Association
    /*
      Data Association: In data association, we associate each observed landmarks
      from the vehicle coordinate frame to a landmarks in the map frame.
        - observed_lankmarks: aka observation that are observed by the car sensors
        - map_landmark: landmarks in map frame are the reference landmarks
        - Fisrt we transform the observed_lankmarks in vehicle frame to observed_lankmarks in map frame using particles as reference
        - Then, We use k-nearest neighbor to get the nearest map_landmark for each observed_landmarks
    */
    // void dataAssociation(vector<landmark> observed_landmarks, vector<landmark> map_landmarks);

    void dataAssociation(vector<landmark> &observed_landmarks, vector<landmark> map_landmarks);

    // ------------------------------------------------------------------------
    // Update Weights
    /*
    What do we have:
      1. We have m observations (lankmark position calculated by sensons) in vehicle frame
      2. We have n landmarks position in map frame
      3. We have p particals in map frame

    What to do for each PARTICLE:
      1. We project the m observation into map frame (rotation and translation) using p as reference
      2. We find the m closest map lanndmarks from the set of n
      3. We compure the probability density usinf 1 and 2. When uncertainty is not included this is similar to using inverse least square between 1 and 2
      4. The computed probability density is the new weight of the partice
    */
    void updateWeights(
      double sensor_range, double std_landmark[],
      vector<landmark> &observed_landmarks, const Map &map_landmarks);

    // ------------------------------------------------------------------------
    // Resample: Using the resampling wheel
    /*
      From the previous step we have new set of particle weights.
        -> These weights define the importance od the particle.
        -> The closer the particle is to the vehicle, the higher is the weight.
        -> Here, we apply sampling particle for the next run based on there weights.
    */

    void resample();

    // ------------------------------------------------------------------------
    /* Set a particles list of associations, along with the associations'
     *   calculated world x,y coordinates
     * This can be a very useful debugging tool to make sure transformations
     *   are correct and assocations correctly connected
     */
    void SetAssociations(Particle& particle, const std::vector<int>& associations,
                         const std::vector<double>& sense_x,
                         const std::vector<double>& sense_y);

    /**
        * Used for obtaining debugging information related to particles.
        */
    std::string getAssociations(Particle best);
    std::string getSenseCoord(Particle best, std::string coord);
};

#endif
