#ifndef MAP_H_
#define MAP_H_

using namespace std;

class Map{
  /*
    A class representation of the map_data.txt data
  */
  public:
    struct single_landmark_s{
      int id_i;
      float x_f;
      float y_f;
    };

    vector<single_landmark_s> landmark_list;

};


#endif
