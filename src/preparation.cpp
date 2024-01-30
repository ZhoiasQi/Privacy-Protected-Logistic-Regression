#include "preparation.hpp"

using namespace std;

vector<vector<uint64_t>> Fixed_Point_Representation_Features(vector<vector<double>> data){
    vector<vector<uint64_t>> res;

    //Lȡ64��sȡ13��ԭ������13λ�õ������������к�������
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

    // //���Դ��룺��ӡ�м���
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
        //TODO: ����Ҫ��Ҫ<<13����Ҫ������
        data[i] *= SCALING_FACTOR;
        res.push_back((uint64_t)data[i]);
        //���Դ��룺��ӡ�м���
        //cout << res[i] << endl;
    }   

    return res;
}

//���Դ��룺��ӡ�м���
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