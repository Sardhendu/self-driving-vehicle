#include <iostream>
#include <cmath>
#include <random>
#include "particle_filter.h"


using namespace std;
using std::normal_distribution;


void ParticleFilter::init(double gps_x, double gps_y, double theta, double std[]){

  std::default_random_engine gen;

  num_particles = 10;

  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];

  // Create normal distribution for x,y and theta
  normal_distribution<double> dist_x(gps_x, std_x);
  normal_distribution<double> dist_y(gps_y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  for (int i=0; i<num_particles; ++i){
    Particle single_particle;

    single_particle.id = i;
    single_particle.x = dist_x(gen);
    single_particle.y = dist_y(gen);
    single_particle.theta = dist_theta(gen);
    single_particle.weight = 1.0;

    particles.push_back(single_particle);
  }
  is_initialized = true;
}


void ParticleFilter::predict(
  double delta_t, double velocity, double yaw_rate, double std[]
){

  std::default_random_engine gen;
  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];

  double vy = velocity/yaw_rate;
  double yaw_theta = yaw_rate*delta_t;

  for (int i=0; i<particles.size(); ++i){
    // Calculate the new postion
    Particle single_particle = particles[i];

    double prev_x = single_particle.x;
    double prev_y = single_particle.y;
    double prev_theta = single_particle.theta;

    double pred_x, pred_y, pred_theta;


    pred_x = prev_x + (vy*(sin(prev_theta + yaw_theta) - sin(prev_theta)));
    pred_y = prev_y + (vy*(cos(prev_theta) - cos(prev_theta +  yaw_theta)));
    pred_theta = prev_theta + yaw_theta;


    // Now we add noise to the preciction
    // (This noise accounts for uncertainty in sensor measurements about the velocity and yawrate)
    // The best way to add noise is to sample from the gaussian distribution with std
    normal_distribution<double> dist_x(pred_x, std_x);
    normal_distribution<double> dist_y(pred_y, std_y);
    normal_distribution<double> dist_theta(pred_theta, std_theta);

    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
  }

}


void ParticleFilter::dataAssociation(
  vector<landmark> observed_landmarks, vector<Map::single_landmark_s> map_landmarks, Particle single_particle
){
    for (int ol=0; ol<observed_landmarks.size(); ol++){
      int assosiated_lankmark_id = 0;


      // Transform the observed_landmarks into map frame using particle as the reference
      cout << "\n\tBefore Transformation:" << "\n\t\t Associated map_landmark = " << observed_landmarks[ol].id << "\n\t\t position vector = " << observed_landmarks[ol].x << ", " << observed_landmarks[ol].y << "\n";
      double l_obs_x = (observed_landmarks[ol].x*cos(single_particle.theta)) - (observed_landmarks[ol].y*sin(single_particle.theta)) + single_particle.x;
      double l_obs_y = (observed_landmarks[ol].y*sin(single_particle.theta)) + (observed_landmarks[ol].y*cos(single_particle.theta)) + single_particle.y;


      int assosiated_l_map_id;
      double dist_buffer = 100000;
      for (int ml=0; ml<map_landmarks.size(); ml++){
          // Calculate the euclidean distance
          double l_map_x = map_landmarks[ml].x_f;
          double l_map_y = map_landmarks[ml].y_f;

          double dist_ = sqrt(pow(l_map_x-l_obs_x, 2) + pow(l_map_y - l_obs_y, 2));

          if (dist_ < dist_buffer){
              dist_buffer = dist_;
              assosiated_l_map_id = map_landmarks[ml].id_i;
          }
      }


      observed_landmarks[ol].id = assosiated_l_map_id;
      observed_landmarks[ol].x = l_obs_x;
      observed_landmarks[ol].y = l_obs_y;
      cout << "\n\tAfter Transformation:" << "\n\t\t Associated map_landmark = " << observed_landmarks[ol].id << "\n\t\t position vector = " << observed_landmarks[ol].x << ", " << observed_landmarks[ol].y << "\n";
    }
}


void ParticleFilter::updateWeights(
  double sensor_range, double std_landmark[],
  const vector<landmark> &observed_landmarks, const Map &map_landmarks
) {
  // Calculate the probability density/ new weights for each particels
  for (int p=0; p<particles.size(); p++){
    Particle single_particle = particles[p];
    dataAssociation(observed_landmarks, map_landmarks.landmark_list, single_particle);
  }

}


void ParticleFilter::print_particle_attributes(int particle_cnt){
  cout << "\t" << "Particle Attributes ........." << "\n";
  for (int i=0; i<particle_cnt; i++){
    cout << "\t" << i << "  x: " << particles[i].x << "  y: " << particles[i].y << "  theta: " << particles[i].theta << "  weight: " << particles[i].weight <<  "\n";
  }
}
