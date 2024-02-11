#ifndef PREPARATION_H
#define PREPARATION_H

#include <cmath>
#include <limits>
#include <iostream>
#include <vector>
#include <iomanip>
#include "logistic_regression.hpp"
#include "test_logistic_regression.hpp"

using namespace std;

vector<vector<uint64_t>> Fixed_Point_Representation_Features(vector<vector<double>> data);
vector<uint64_t> Fixed_Point_Representation_Labels(vector<double> data);

void print_Max_Min_and_Precision(const vector<vector<double>>& training_data);

vector<vector<double>> Fix_to_Double_F(vector<vector<uint64_t>> data);
vector<double> Fix_to_Double_L(vector<uint64_t> data);

#endif