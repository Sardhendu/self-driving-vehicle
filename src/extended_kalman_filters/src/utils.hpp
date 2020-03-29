#include <iostream>
#include <vector>

using namespace std;

template <typename T>
void printInfo(vector<T> tokenIn){
  cout << tokenIn.size() << " ";
  for (int i=0; i<=tokenIn.size()-1; i++){
    cout << "token ==> " << tokenIn[i] << "\n";
  }
}
