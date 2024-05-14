#ifndef READ_WBDC_H
#define READ_WBDC_H

#include <iostream>
#include <vector>
#include <fstream>
#include <string>

using namespace std;

struct BreastCancerInstance {
    string id;
    bool diagnosis; 
    vector<float> features;
};

vector<BreastCancerInstance> read_WBDC_data(const string& filename);
void print_WBDC_Data(const vector<BreastCancerInstance>& data);

vector<vector<double>> reverse_BreastCancerInstance_to_features(vector<BreastCancerInstance> vec);
vector<double> reverse_BreastCancerInstance_to_labels(vector<BreastCancerInstance> vec);

#endif
