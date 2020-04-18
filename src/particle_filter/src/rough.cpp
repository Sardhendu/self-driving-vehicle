#include <iostream>

using namespace std;
std::map<int, std::string> values;
for(std::map<int, std::string>::value_type& x : values)
{
    std::cout << x.first << "," << x.second << std::endl;
}
