#ifndef PREPARATION_H
#define PREPARATION_H

#include <cmath>
#include <limits>
#include <iostream>
#include <vector>
#include <iomanip>
#include "defines.hpp"

using namespace std;

vector<vector<uint64_t>> Fixed_Point_Representation_Features(vector<vector<double>> data);
vector<uint64_t> Fixed_Point_Representation_Labels(vector<double> data);

void print_Max_Min_and_Precision(const vector<vector<double>>& training_data);

#endif