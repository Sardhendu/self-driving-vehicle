#include "particle_filter.h"


#include <random>
#include <set>
#include <cmath>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include "utils.hpp"
#include "parser.h"

using std::string;
using std::vector;



using namespace std;
using std::normal_distribution;


void ParticleFilter::init(double gps_x, double gps_y, double theta, double std[]){

  std::default_random_engine gen;

  num_particles = 100;

  // Create normal distribution for x,y and theta
  normal_distribution<double> dist_x(0, std[0]);
  normal_distribution<double> dist_y(0, std[1]);
  normal_distribution<double> dist_theta(0, std[2]);
  // cout << "input: " << gps_x << " " << gps_y << " " << theta << "\n";

  for (int i=0; i<num_particles; i++){
    Particle single_particle;

    single_particle.id = i;
    single_particle.x = gps_x + dist_x(gen);
    single_particle.y = gps_y + dist_y(gen);
    single_particle.theta = theta + dist_theta(gen);
    single_particle.weight = 1.0;

    // cout << "Particle generated = " << single_particle.x << " " << single_particle.y << " " << single_particle.theta << "\n";
    particles.push_back(single_particle);
  }
  is_initialized = true;
}


void ParticleFilter::prediction(
  double delta_t, double std[], double velocity, double yaw_rate
){
  std::default_random_engine gen;
  normal_distribution<double> dist_x(0, std[0]);
  normal_distribution<double> dist_y(0, std[1]);
  normal_distribution<double> dist_theta(0, std[2]);

  // Handle when yaw_rate = 0 to avoid division by zero
  if (yaw_rate < 0.000001){
    yaw_rate += 0.0000001;
  }

  double vyaw = velocity/yaw_rate;
  double yaw_dt = yaw_rate*delta_t;

  for (unsigned int p=0; p<particles.size(); p++){
    // if (p<10){
    //   cout << "\t\tRUNNING FOR PARTICLE =================> " << p << "\n";
    //   cout << "\t\tbefore predict " << particles[p].id << " " << particles[p].x << " " << particles[p].y << " " << particles[p].theta << " " << particles[p].weight << "\n";
    //   cout << "\t\tparms " << velocity << " " << yaw_rate << " " << vyaw << " " << yaw_dt << " " << sin(particles[p].theta + yaw_dt) << " "  << sin(particles[p].theta) << " " << cos(particles[p].theta) << " "  << cos(particles[p].theta +  yaw_dt) << "\n";
    //   cout << "\t\tafter predict " << particles[p].id << " " << particles[p].x << " " << particles[p].y << " " << particles[p].theta << " " << particles[p].weight << "\n";
    // }
    particles[p].x += vyaw*(sin(particles[p].theta + yaw_dt) - sin(particles[p].theta));
    particles[p].y += vyaw*(cos(particles[p].theta) - cos(particles[p].theta +  yaw_dt));
    particles[p].theta += yaw_dt;


    // Now we add noise to the preciction
    // (This noise accounts for uncertainty in sensor measurements about the velocity and yawrate)
    particles[p].x += dist_x(gen);
    particles[p].y += dist_y(gen);
    particles[p].theta += dist_theta(gen);


  }

}


void ParticleFilter::dataAssociation(
  vector<landmark> &observed_landmarks, vector<landmark> map_landmarks
){
    for (unsigned ol=0; ol<observed_landmarks.size(); ol++){
      // cout << "\n\t\tobserved_landmark = " << observed_landmarks[ol].x << " " << observed_landmarks[ol].y << " " << "\n";
      int map_id = -1;
      double min_dist = 100000;
      for (unsigned int ml=0; ml<map_landmarks.size(); ml++){
          // Calculate the euclidean distance
          double dist_ = euclid_dist(
            observed_landmarks[ol].x, observed_landmarks[ol].y,
            map_landmarks[ml].x, map_landmarks[ml].y
          );

          if (dist_ < min_dist){
              min_dist = dist_;
              map_id = map_landmarks[ml].id;
          }
      }

      // Validate: -1 is needed becasue id go from 1-n and index go from 0-n-1
      observed_landmarks[ol].id = map_id;

      // for (int kk=0; kk<map_landmarks.size(); kk++){
      //   if (map_id == map_landmarks[kk].id){
      //     cout << "\t\tmap_id " << map_id << "\n";
      //     cout << "\t\tassociated_map_landmark: " << map_landmarks[kk].x << " " << map_landmarks[kk].y << " " << "\n";
      //     cout << "\t\tmin_dist " << min_dist << "\n";
      //   }
      // }
    }
}


void ParticleFilter::updateWeights(
  double sensor_range, double std_landmark[],
  vector<landmark> &observed_landmarks, const Map &map_landmarks
) {
  double sig_x = std_landmark[0];
  double sig_y = std_landmark[1];

  // Calculate the probability density/ new weights for each particels
  for (unsigned int p=0; p<particles.size(); p++){
    // Reinit the weights everytime
    particles[p].weight = 1.0;

    // cout << "\t\tupdate_weights" << particles[p].id << " " << particles[p].x << " " << particles[p].y << " " << particles[p].weight;


    // ---------------------------------------------------------------------------------
    // Select Map Landmarks that are within the range of the particle
    // ---------------------------------------------------------------------------------
    vector<landmark> gt_landmarks_map;
    for (unsigned int ml=0; ml<map_landmarks.landmark_list.size(); ml++){
        landmark s_obs;
        Map::single_landmark_s s_ml = map_landmarks.landmark_list[ml];
        // cout << s_ml.x_f << " " << particles[p].x << "  " << s_ml.y_f << " " << particles[p].y << "  " << sensor_range << "\n";
        if (fabs(s_ml.x_f - particles[p].x) <= sensor_range && fabs(s_ml.y_f - particles[p].y) <= sensor_range){
          s_obs.id = s_ml.id_i;
          s_obs.x = s_ml.x_f;
          s_obs.y = s_ml.y_f;
          gt_landmarks_map.push_back(s_obs);
        }

    }
    // cout << "\t\tCount of landmark within range = " << gt_landmarks_map.size() << "\n";

    // ---------------------------------------------------------------------------------
    // Transform the observed_landmark in vechicle frame to map frame
    // ---------------------------------------------------------------------------------
    vector<landmark> obs_landmark_map;
    for (unsigned int ol=0; ol<observed_landmarks.size(); ol++){
        landmark transformed_obs;
        transformed_obs.id = observed_landmarks[ol].id;
        transformed_obs.x = (observed_landmarks[ol].x*cos(particles[p].theta)) - (observed_landmarks[ol].y*sin(particles[p].theta)) + particles[p].x;
        transformed_obs.y = (observed_landmarks[ol].x*sin(particles[p].theta)) + (observed_landmarks[ol].y*cos(particles[p].theta)) + particles[p].y;
        obs_landmark_map.push_back(transformed_obs);
    }

    // ---------------------------------------------------------------------------------
    // Assosiate each observed_landmark to the nearest map landmark
    // ---------------------------------------------------------------------------------
    dataAssociation(obs_landmark_map, gt_landmarks_map);


    // ---------------------------------------------------------------------------------
    // Calculate the probability density function for the particle given the observed and map landmark
    // ---------------------------------------------------------------------------------
    for (unsigned int ol=0; ol<obs_landmark_map.size(); ol++){
      double map_id = obs_landmark_map[ol].id;


      // TODO: This is a bad process to loop everytime, try to save map_landmark as dictionaty with key as the map_id
      double gt_lx;
      double gt_ly;
      for (unsigned int ml=0; ml<gt_landmarks_map.size(); ml++){
        if (gt_landmarks_map[ml].id == map_id){
          gt_lx = gt_landmarks_map[ml].x;
          gt_ly = gt_landmarks_map[ml].y;
        }
      }

      double prob = multiv_prob(
        sig_x, sig_y, obs_landmark_map[ol].x, obs_landmark_map[ol].y, gt_lx, gt_ly
      );

      // cout << "\n\t\tobserved_landmark = " << obs_landmark_map[ol].x << " " << obs_landmark_map[ol].y << " " << "\n";
      // cout << "\t\tmap_landmark = " << gt_lx << " " << gt_ly << " " << "\n";
      // cout << "\t\tcurr probability density = " << prob << "\n";
      // cout << "\t\t[Before] cumulative probability density = " << particles[p].weight << "\n";
      particles[p].weight *= prob;
      // cout << "\t\t[After] cumulative probability density = " << particles[p].weight << "\n";
    }

//     if (particles[p].weight > 1){
//         exit;
//     }
    // if (particles[p].weight > 0){
    //   cout << "\n\tFinal pdf = " << particles[p].weight << "\n";
    // }

  }

}

void ParticleFilter::resample(){
  std::default_random_engine gen;
  double beta = 0;
  // Select a random index within the range of (0, num_particles)
  auto index = int(rand() % num_particles);

  // Fetch the weight of every partice
  vector<double> weights;
  double max_particle_weight = 0;
  for (int s=0; s<num_particles; s++){
     weights.push_back(particles[s].weight);
     if (particles[s].weight > max_particle_weight){
       max_particle_weight = particles[s].weight;
     }
  }

  // cout << "\t\tmax_particle_weight " << max_particle_weight << "\n";

  // Define a random generator
  uniform_real_distribution<double> unirealdist(0.0, max_particle_weight);
  set<int> unq_particle_ids;
  // Spin the sampling wheel
  for (int i=0; i<num_particles; i++){
    double gen_val = unirealdist(gen);
    beta +=  gen_val * 2 * max_particle_weight;
    while (weights[index] < beta){
        beta -= weights[index];
        index = (index + 1) % particles.size(); // the index should not exceed num_particles
    }

    // Update the particles attributes with the new sampled particles
    particles[i].id = particles[index].id;
    particles[i].x = particles[index].x;
    particles[i].y = particles[index].y;
    particles[i].theta = particles[index].theta;
    particles[i].weight = particles[index].weight;

    // cout << "particle selected = " << particles[i].id << "\n";
    unq_particle_ids.insert(particles[i].id);
  }

  // vector<int>::iterator it;
  // it = unique(particle_ids.begin(), particle_ids.end());
  // cout << "Unique particle count " << unq_particle_ids.size() << "\n";

}


void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

void ParticleFilter::print_particle_attributes(int particle_cnt){
  cout << "\t" << "Particle Attributes ........." << "\n";
  for (int i=0; i<particle_cnt; i++){
    cout << "\t" << i << "  x: " << particles[i].x << "  y: " << particles[i].y << "  theta: " << particles[i].theta << "  weight: " << particles[i].weight <<  "\n";
  }
}
