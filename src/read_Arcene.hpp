#ifndef READ_ARCENE_H
#define READ_ARCENE_H

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>  

using namespace std;

vector<vector<uint64_t>> readData(const string& filename);
vector<uint64_t> readLabel(const string& filename);

vector<vector<double>> readData_(const string& filename);
vector<double> readLabel_(const string& filename);

#endif