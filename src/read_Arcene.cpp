#include "read_Arcene.hpp"

vector<vector<uint64_t>> readData(const string& filename){
    ifstream file(filename);  

    if (!file.is_open()) {  
        throw runtime_error("无法打开文件: " + filename);  
    }  
  
    vector<vector<uint64_t>> data;  
    string line;  

    while (getline(file, line)) {  
        vector<uint64_t> row;  
        istringstream iss(line);  
        uint64_t value;  
        while (iss >> value) {  
            row.push_back(value);  
        }  
        data.push_back(row);  
    }  

    file.close();  

    return data; 
}

vector<uint64_t> readLabel(const string& filename) {  
    ifstream file(filename);  

    if (!file.is_open()) {  
        throw runtime_error("无法打开文件: " + filename);  
    }  

    vector<uint64_t> labels;  
    string line;  

    while (getline(file, line)) {  
        int label;  
        istringstream iss(line);  
        if (!(iss >> label)) {  
            throw runtime_error("文件格式错误: " + filename);  
        }  

        // 将-1映射为0，1保持不变  
        labels.push_back(label == -1 ? 0 : 1);  
    }  

    file.close();  

    return labels;  
}  

vector<vector<double>> readData_(const string& filename){
    ifstream file(filename);  

    if (!file.is_open()) {  
        throw runtime_error("无法打开文件: " + filename);  
    }  
  
    vector<vector<double>> data;  
    string line;  

    while (getline(file, line)) {  
        vector<double> row;  
        istringstream iss(line);  
        double value;  
        while (iss >> value) {  
            row.push_back(value);  
        }  
        data.push_back(row);  
    }  

    file.close();  

    return data; 
}

vector<double> readLabel_(const string& filename) {  
    ifstream file(filename);  

    if (!file.is_open()) {  
        throw runtime_error("无法打开文件: " + filename);  
    }  

    vector<double> labels;  
    string line;  

    while (getline(file, line)) {  
        int label;  
        istringstream iss(line);  
        if (!(iss >> label)) {  
            throw runtime_error("文件格式错误: " + filename);  
        }  

        // 将-1映射为0，1保持不变  
        labels.push_back(label == -1 ? 0 : 1);  
    }  

    file.close();  

    return labels;  
}  