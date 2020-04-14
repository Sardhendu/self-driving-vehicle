#ifndef PARTICLE_FILTER_H_
#define PARTICLE_FILTER_H_


struct Praticle {
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
};


class ParticleFilter {
  private:
    vector<Particle> particles;
    bool is_initialized;

  public:
    // Define constructors
    ParticleFilter() {}
    ~ParticleFilter() {}

    // Initialize
    void Initialize(int id, double x, double y, double theta);

    // Prediction
    /*
      Prediction:
      Prediction is simply the sampling step.
    */
};

#endif
