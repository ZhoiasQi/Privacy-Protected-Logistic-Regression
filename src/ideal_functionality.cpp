//用来测试本地直接进行明文逻辑回归训练的结果，看看和安全逻辑回归的区别。

// #include "read_WBDC.hpp"
#include "read_Arcene.hpp"
#include "util.hpp"
#include <math.h>

using namespace Eigen;
using Eigen::Matrix;
using namespace std;

Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", " << ", ";");

struct TrainingParams{
    int n, d;
    double alpha = 1.0/LEARNING_RATE_INV;
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
        this->t = (params.n)/BATCH_SIZE;
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
        RowMatrixXd Xb = X.block(indexLo, 0, BATCH_SIZE, d);
        ColVectorXd Yb = Y.segment(indexLo, BATCH_SIZE);

        ColVectorXd Y_(BATCH_SIZE);
        ColVectorXd Sig(BATCH_SIZE);
        ColVectorXd D(BATCH_SIZE);
        ColVectorXd delta(d);

        Y_ = Xb * w;

        for (int i = 0; i < BATCH_SIZE; i++) {
            Sig(i) = sigmoid(Y_(i));
        }

        D = Sig - Yb;

        delta = Xb.transpose() * D;

        delta = (delta * alpha)/BATCH_SIZE;

        w -= delta;

    }

    void train_model(){
        for (int i = 0; i < t; i++){
            int indexLo = (i * BATCH_SIZE) % n;
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
            //cout << prediction[i] << "   " << testing_labels[i] << endl;

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

    //读参数
    int num_iters = atoi(argv[1]);
    
    //读文件
    // vector<BreastCancerInstance> dataSet;
    // string fileName = "../../Dataset/wdbc.data";

    // dataSet = read_WBDC_data(fileName);

    // vector<vector<double>> dataFeatures;
    // vector<double> dataLabels;

    // dataFeatures = reverse_BreastCancerInstance_to_features(dataSet);
    // dataLabels = reverse_BreastCancerInstance_to_labels(dataSet);

    // //按照相同的顺序随机打乱特征和标签
    // //shuffleData(dataFeatures, dataLabels);

    //划分测试集和训练集
    vector<vector<double>> training_Features = readData_("../../arcene/ARCENE/arcene_train.data");
    vector<vector<double>> testing_Features = readData_("../../arcene/ARCENE/arcene_valid.data");
    vector<double> training_Labels = readLabel_("../../arcene/ARCENE/arcene_train.labels");
    vector<double> testing_Labels = readLabel_("../../arcene/arcene_valid.labels");

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

    return 0;
}
