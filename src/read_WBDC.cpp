#include "read_WBDC.hpp"

using namespace std;

//读入数据并存储在结构体向量中
vector<BreastCancerInstance> read_WBDC_data(const string& filename) {
    vector<BreastCancerInstance> data;

    ifstream file(filename);
    if (file.is_open()) {
        string line;
        while (getline(file, line)) {
            BreastCancerInstance instance;
            size_t pos = line.find(',');
            auto id = line.substr(0, pos);
            instance.id = id;
            line.erase(0, pos + 1); 

            pos = line.find(',');
            string diagnosisStr = line.substr(0, pos);
            auto diagnosis = (diagnosisStr == "M");
            instance.diagnosis = diagnosis;
            line.erase(0, pos + 1);

            while ((pos = line.find(',')) != string::npos) {
                string featureStr = line.substr(0, pos);
                float feature = stof(featureStr);
                instance.features.push_back(feature);
                line.erase(0, pos + 1); 
            }

            float feature = stof(line);
            instance.features.push_back(feature);

            data.push_back(instance);
        }

        file.close();
    } else {
        cout << "Error opening file: " << filename << endl;
    }

    return data;
}

//测试代码：打印中间结果
void print_WBDC_Data(const vector<BreastCancerInstance>& data) {
    for (const auto& instance : data) {
        cout << "ID: " << instance.id << endl;
        cout << "Diagnosis: " << (instance.diagnosis ? "Malignant" : "Benign") << endl;
        cout << "Features:";
        for (const auto& feature : instance.features) {
            cout << " " << feature;
        }
        cout << endl << endl;
    }
}

//把特征值转化为double矩阵
vector<vector<double>> reverse_BreastCancerInstance_to_features(vector<BreastCancerInstance> vec){
    vector<vector<double>> res;

    for(const auto& instance : vec){
        vector<double> instanceData;
        instanceData.reserve(instance.features.size());
        for (const auto& feature : instance.features) {
            instanceData.push_back(static_cast<double>(feature));
        }
        res.push_back(instanceData);
    }

    return res;
}

//把diagnosis转化为double向量
vector<double> reverse_BreastCancerInstance_to_labels(vector<BreastCancerInstance> vec) {
    vector<double> labels;

    for (const auto& instance : vec) {
        double label = instance.diagnosis ? 1.0 : 0.0;
        labels.push_back(label);
    }
    
    return labels;
}


