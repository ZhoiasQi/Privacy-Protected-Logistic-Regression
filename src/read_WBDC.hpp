#ifndef READ_WBDC_HPP
#define READ_WBDC_HPP

#include <iostream>
#include <vector>
#include <fstream>
#include <string>

using namespace std;

struct BreastCancerInstance {
    string id;
    bool diagnosis; // 为真表示恶性肿瘤，为假表示良性
    vector<float> features;
};

vector<BreastCancerInstance> read_WBDC_data(const string& filename);
void print_WBDC_Data(const vector<BreastCancerInstance>& data);

vector<vector<double>> reverse_BreastCancerInstance_to_features(vector<BreastCancerInstance> vec);
vector<double> reverse_BreastCancerInstance_to_labels(vector<BreastCancerInstance> vec);

#endif
