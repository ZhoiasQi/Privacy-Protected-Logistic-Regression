//用来测试本地直接进行明文逻辑回归训练的结果，看看和安全逻辑回归的区别。

#include "read_WBDC.hpp"
#include "read_Arcene.hpp"
#include "util.hpp"
#include <math.h>

using namespace Eigen;
using Eigen::Matrix;
using namespace std;

Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", " << ", ";");

struct TrainingParams{
    int n, d;
    double alpha = 1.0/LEARNING_RATE_INV_I;
};

class LogisticRegression{
public:
    double alpha;
    int n, d, t;
    RowMatrixXd X;
    ColVectorXd Y;
    ColVectorXd w;
    LogisticRegression(RowMatrixXd& training_data, ColVectorXd& training_labels, TrainingParams params){
        this->n = params.n;
        this->d = params.d;
        this->t = (params.n)/BATCH_SIZE_I;
        this->alpha = params.alpha;
        X = training_data;
        Y = training_labels;
        w.resize(d);
        for(int i = 0; i < d; i++){
            w[i] = 0;
        }
        train_model();
    }

    double sigmoid(double x){
        double expNum = exp(-x);
        double denominator = 1 + expNum;
        double res = 1 / denominator;
        return res;
    }

    void train_batch(int iter, int indexLo){
        RowMatrixXd Xb = X.block(indexLo, 0, BATCH_SIZE_I, d);
        ColVectorXd Yb = Y.segment(indexLo, BATCH_SIZE_I);

        ColVectorXd Y_(BATCH_SIZE_I);
        ColVectorXd Sig(BATCH_SIZE_I);
        ColVectorXd D(BATCH_SIZE_I);
        ColVectorXd delta(d);

        Y_ = Xb * w;

            // for(int i = 0; i < BATCH_SIZE_I; i++){
            //     cout << Y_(i) << "  ";
            // }
            // cout << endl;

        for (int i = 0; i < BATCH_SIZE_I; i++) {
            Sig(i) = sigmoid(Y_(i));
            //cout << Sig(i) << endl;
        }


        D = Sig - Yb;

        delta = Xb.transpose() * D;

        delta = (delta * alpha)/BATCH_SIZE_I;

        w -= delta;

    }

    void train_model(){
        for (int i = 0; i < t; i++){
            int indexLo = (i * BATCH_SIZE_I) % n;
            train_batch(i, indexLo);
            //cout << "t = " << i << endl << w << endl;
        }
    }

    void test_model(RowMatrixXd& testing_data, ColVectorXd& testing_labels){
        
        ColVectorXd prediction;
    
        prediction = testing_data * w;

        int n_ = prediction.size();

        for(int i = 0; i < n_; i++){
            //cout << prediction(i) << "   ";
            double temp = 1 / (1 + exp(-prediction(i)));
            //cout << temp << "   ";
            prediction[i] = temp;
        }
        //cout << endl;

        int num_correct = 0;
        for (int i = 0; i < n_; i++){
            // cout << prediction[i] << "   " << testing_labels[i] << endl;

            if(testing_labels[i] == 1){
                if(prediction[i] >= 0.5){
                    num_correct++;
                }
            }
            else{
                if(prediction[i] < 0.5){
                    num_correct++;
                }
            }
        }

        double accuracy = num_correct/((double) n_);
        cout << "Accuracy on testing the trained model is " << accuracy * 100 << endl;
    }
};

int main(int argc, char** argv){

    auto start = std::chrono::high_resolution_clock::now();

    //读参数
    //int num_iters = atoi(argv[1]);
    int num_iters = 50;

    //这部分是老数据集的前期数据处理
    
    //读文件
    vector<BreastCancerInstance> dataSet;
    string fileName = "../../Dataset/wdbc.data";

    dataSet = read_WBDC_data(fileName);

    vector<vector<double>> dataFeatures;
    vector<double> dataLabels;

    dataFeatures = reverse_BreastCancerInstance_to_features(dataSet);
    dataLabels = reverse_BreastCancerInstance_to_labels(dataSet);

    // 划分数据集

    size_t trainSize = 400; // 训练集大小  

    std::vector<std::vector<double>> training_Features(dataFeatures.begin(), dataFeatures.begin() + trainSize);  

    std::vector<double> training_Labels(dataLabels.begin(), dataLabels.begin() + trainSize);  

    // 剩余样本作为测试集  

    std::vector<std::vector<double>> testing_Features(dataFeatures.begin() + trainSize, dataFeatures.end());  

    std::vector<double> testing_Labels(dataLabels.begin() + trainSize, dataLabels.end());

    //这部分是新数据集

    // //划分测试集和训练集
    // vector<vector<double>> training_Features = readData_("../../arcene/ARCENE/arcene_train.data");
    // vector<double> training_Labels = readLabel_("../../arcene/ARCENE/arcene_train.labels");

    // vector<vector<double>> testing_Features = readData_("../../arcene/ARCENE/arcene_valid.data");
    // vector<double> testing_Labels = readLabel_("../../arcene/arcene_valid.labels");
    
    auto temp1 = training_Features;
    auto temp2 = training_Labels;

    for(int i = 0; i < num_iters - 1; i++){
        // std::default_random_engine generator; 
        // std::shuffle(temp1.begin(), temp1.end(), generator);  
        // std::shuffle(temp2.begin(), temp2.end(), generator); 
        training_Features.insert(training_Features.end(),temp1.begin(),temp1.end());
        training_Labels.insert(training_Labels.end(),temp2.begin(),temp2.end());
    }

    //训练
    cout << "========" << endl;
    cout << "Training" << endl;
    cout << "========" << endl;

    TrainingParams params;

    params.n = training_Features.size();
    cout << "Number of Instances: " << params.n << endl;
    params.d = training_Features[0].size();
    cout << "Number of Features: " << params.d << endl;

    RowMatrixXd X(params.n, params.d);
    vector2d_to_RowMatrixXd(training_Features, X);

    // for(int i = 0; i < params.n; i++){
    //     for(int j = 0; j < params.d; j++){
    //         cout << X(i,j) << " ";
    //     }
    //     cout << endl;
    // }
    
    ColVectorXd Y(params.n);
    vector_to_ColVectorXd(training_Labels, Y);

    LogisticRegression logisticRegression(X, Y, params);

    //测试
    cout << "=======" << endl;
    cout << "Testing" << endl;
    cout << "=======" << endl;

    int n_;

    n_ = testing_Labels.size();
    cout << "Number of Instances: " << n_ << endl;

    RowMatrixXd testX(n_, params.d);
    vector2d_to_RowMatrixXd(testing_Features, testX);

    ColVectorXd testY(n_);
    vector_to_ColVectorXd(testing_Labels, testY);

    logisticRegression.test_model(testX, testY);

    auto end = std::chrono::high_resolution_clock::now(); 

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);  

    std::cout << "Time: " << duration.count() << "ms" << std::endl;  

    return 0;
}
