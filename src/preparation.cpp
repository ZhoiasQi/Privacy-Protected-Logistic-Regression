#include "preparation.hpp"

using namespace std;

vector<vector<uint64_t>> Fixed_Point_Representation_Features(vector<vector<double>> data){
    vector<vector<uint64_t>> res;

    //全部乘放缩因子，即将所有小数左移动13位。得到定点数。后面直接当作大整数运算，l用64
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

    // 测试代码：输出中间结果
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

    //全部乘放缩因子，即将所有小数左移动13位。得到定点数。后面直接当作大整数运算，l用64
    int m = data.size();

    for(int i = 0; i < m; i++){
        data[i] *= SCALING_FACTOR;
        res.push_back((uint64_t)data[i]);
        cout << res[i] << endl;
    }   

    return res;
}

void print_Max_Min_and_Precision(const vector<vector<double>>& training_data) {
    double max_val = numeric_limits<double>::min(); // 最大值初始化为最小
    double min_val = numeric_limits<double>::max(); // 最小值初始化为最大
    int max_precision = 0; // 小数点位数最多的数的小数位数
    
    // 遍历训练数据，找到最大值、最小值和小数点位数最多的数的小数位数
    for (const vector<double>& row : training_data) {
        for (double value : row) {
            if (value > max_val) {
                max_val = value;
            }
            if (value < min_val) {
                min_val = value;
            }
            
            // 计算小数位数
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
    
    // 输出最大值、最小值和小数点位数最多的数的小数位数
    cout << "Max value: " << max_val << endl;
    cout << "Min value: " << min_val << endl;
    cout << "Max precision: " << max_precision << endl;
}