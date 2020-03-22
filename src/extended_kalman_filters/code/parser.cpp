#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <vector>
#include "parser.h"

using namespace std;


void MeasurementParser::setMeasurements(vector<string> measurement_vector_in){
  // printInfo(measurement_vector_in);
  measurement_vector=measurement_vector_in;
  sensor_type = measurement_vector_in[0];
}

vector<float> MeasurementParser::getMeasurements(){
  vector<float> out;
  if (sensor_type == "L"){
    cout << "LASER MEASUREMENT" << "\n";
    float px = stof(measurement_vector[1]);
    float py = stof(measurement_vector[2]);
    out = {px, py};
    // float v = stof(measurement_vector[1]);
  }
  else if (sensor_type == "R"){
    cout << "RADAR MEASUREMENT" << "\n";
    float range = stof(measurement_vector[1]);
    float bearing = stof(measurement_vector[2]);
    float r_velocity = stof(measurement_vector[3]);
    out = {range, bearing, r_velocity};
  }
  else{
    cout << "The input sensor type does not match given" << measurement_vector[0] << "\n";
    exit(1);
  }
  return out;
}

long long MeasurementParser::getTimestamp(){
  long long timestamp_;

  if (sensor_type == "L"){
    timestamp_ = stoll(measurement_vector[3]);
  }
  else if (sensor_type == "R"){
    timestamp_ = stoll(measurement_vector[4]);
  }
  else{
    cout << "The input sensor type does not match given" << measurement_vector[0] << "\n";
    exit(1);
  }
}
