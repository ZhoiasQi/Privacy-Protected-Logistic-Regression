#include "preparation.hpp"

using namespace std;

vector<vector<uint64_t>> Fixed_Point_Representation_Features(vector<vector<double>> data){
    vector<vector<uint64_t>> res;

    int m = data.size();
    int n = data[0].size();

    for(int i = 0; i < m; i++){
        vector<uint64_t> temp;
        for(int j = 0; j < n; j++){
            data[i][j] *= SCALING_FACTOR;
            temp.push_back((uint64_t)data[i][j]);
        }
        res.push_back(temp);
    }       

    // for(int i = 0; i < m; i++){
    //     vector<uint64_t> temp;
    //     for(int j = 0; j < n; j++){
    //         cout << res[i][j] << "    ";
    //     }
    //     cout << endl;
    // }       

    return res;
}

vector<uint64_t> Fixed_Point_Representation_Labels(vector<double> data){
    vector<uint64_t> res;

    
    int m = data.size();

    for(int i = 0; i < m; i++){
        data[i] *= SCALING_FACTOR;
        res.push_back((uint64_t)data[i]);

        //cout << res[i] << endl;
    }   

    return res;
}


void print_Max_Min_and_Precision(const vector<vector<double>>& training_data) {
    double max_val = numeric_limits<double>::min(); 
    double min_val = numeric_limits<double>::max(); 
    int max_precision = 0; 
    
    for (const vector<double>& row : training_data) {
        for (double value : row) {
            if (value > max_val) {
                max_val = value;
            }
            if (value < min_val) {
                min_val = value;
            }
            
            string value_str = to_string(value);
            size_t decimal_pos = value_str.find(".");
            if (decimal_pos != string::npos) {
                int precision = value_str.size() - decimal_pos - 1;
                if (precision > max_precision) {
                    max_precision = precision;
                }
            }
        }
    }
    
    cout << "Max value: " << max_val << endl;
    cout << "Min value: " << min_val << endl;
    cout << "Max precision: " << max_precision << endl;
}

vector<vector<double>> Fix_to_Double_F(vector<vector<uint64_t>> data){
    vector<vector<double>> res;

    int m = data.size();
    int n = data[0].size();

    for(int i = 0; i < m; i++){
        vector<double> temp;
        for(int j = 0; j < n; j++){
            double dataij = (double)data[i][j];
            dataij /= SCALING_FACTOR;
            temp.push_back(dataij);
        }
        res.push_back(temp);
    }           

    return res;
}

vector<double> Fix_to_Double_L(vector<uint64_t> data){
    vector<double> res;

    int m = data.size();

    for(int i = 0; i < m; i++){
        double temp = data[i];
        temp /= SCALING_FACTOR;
        res.push_back(temp);
    }   

    return res;
}

void offlineTest(vector<vector<uint64_t>> data, vector<uint64_t> label, TestingParams t_params, LogisticRegression trainModel){

    auto testing_Features = Fix_to_Double_F(data);

    auto testing_Labels = Fix_to_Double_L(label);
    
    RowMatrixXd testX(t_params.n, t_params.d);
    vector2d_to_RowMatrixXd(testing_Features, testX); 

    ColVectorXd testY(t_params.n);  
    vector_to_ColVectorXd(testing_Labels, testY);  

    trainModel.test_model(testX, testY);
}

void normalizeColumns(std::vector<std::vector<double>>& dataFeatures) {  

    if (dataFeatures.empty()) return; // 如果矩阵为空，直接返回  

  

    // 找出每一列的最大值和最小值  

    std::vector<double> minValues(dataFeatures[0].size(), std::numeric_limits<double>::max());  

    std::vector<double> maxValues(dataFeatures[0].size(), std::numeric_limits<double>::lowest());  

    for (const auto& row : dataFeatures) {  

        if (row.size() != minValues.size()) {  

            throw std::runtime_error("Matrix is not rectangular (all rows must have the same number of elements).");  

        }  

        for (size_t j = 0; j < row.size(); ++j) {  

            minValues[j] = std::min(minValues[j], row[j]);  

            maxValues[j] = std::max(maxValues[j], row[j]);  

        }  

    }  

  

    // 按列归一化  

    for (auto& row : dataFeatures) {  

        for (size_t j = 0; j < row.size(); ++j) {  

            // 避免除以0的情况  

            if (maxValues[j] == minValues[j]) {  

                // 可以选择设置一个默认值，例如0或者保持不变  

                // row[j] = 0; // 或者保持原值 row[j] = row[j];  

                continue;  

            }  

            // 归一化到0-1范围  

            row[j] = (row[j] - minValues[j]) / (maxValues[j] - minValues[j]);  

        }  

    }  

}