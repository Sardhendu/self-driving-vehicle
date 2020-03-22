#ifndef PARSER_H
#define PARSER_H

#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <vector>


using namespace std;

class MeasurementParser{
  public:
    vector<string> measurement_vector;
    string sensor_type;
    long long timestamp_;
    void setMeasurements(vector<string> measurement_vector_in);
    vector<float> getMeasurements();
    long long getTimestamp();
};

#endif
