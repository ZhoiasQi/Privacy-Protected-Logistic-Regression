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