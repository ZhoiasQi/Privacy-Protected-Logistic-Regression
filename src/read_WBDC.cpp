#include "read_WBDC.hpp"

using namespace std;

//读文件，返回结构体数组，后面再进行后续处理
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
            line.erase(0, pos + 1); // 删除已读取的ID和逗号

            pos = line.find(',');
            string diagnosisStr = line.substr(0, pos);
            auto diagnosis = (diagnosisStr == "M");
            instance.diagnosis = diagnosis;
            line.erase(0, pos + 1); // 删除已读取的诊断标签和逗号

            while ((pos = line.find(',')) != string::npos) {
                string featureStr = line.substr(0, pos);
                float feature = stof(featureStr);
                instance.features.push_back(feature);
                line.erase(0, pos + 1); // 删除已读取的特征值和逗号
            }

            // 处理最后一个特征值（无逗号分隔）
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

//打印数据
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

//把结构体数组中的特征值转化为矩阵
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

//将结构体中的diagnosis属性转换为double向量
vector<double> reverse_BreastCancerInstance_to_labels(vector<BreastCancerInstance> vec) {
    vector<double> labels;

    for (const auto& instance : vec) {
        double label = instance.diagnosis ? 1.0 : 0.0;
        labels.push_back(label);
    }
    
    return labels;
}


