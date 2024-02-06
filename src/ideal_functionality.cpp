//�������Ա���ֱ�ӽ����߼��ع�ѵ���Ľ���������Ͱ�ȫ�߼��ع������

#include "read_WBDC.hpp"
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
        for(int i = 0; i < d; i++)
            w[i] = 0;
        train_model();
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
            Sig(i) = 1 / (1 + exp(-Y_(i)));
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
        }
    }

    void test_model(RowMatrixXd& testing_data, ColVectorXd& testing_labels){
        ColVectorXd prediction;
        prediction = testing_data * w;

        int n_ = prediction.size();

        for(int i = 0; i < n_; i++){
            int temp = 1 / (1 + exp(-prediction(i)));
            prediction[i] = temp;
        }

        int num_correct = 0;
        for (int i = 0; i < n_; i++){
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

    //������
    int num_iters = atoi(argv[1]);
    
    //���ļ�
    vector<BreastCancerInstance> dataSet;
    string fileName = "../../Dataset/wdbc.data";

    dataSet = read_WBDC_data(fileName);

    vector<vector<double>> dataFeatures;
    vector<double> dataLabels;

    dataFeatures = reverse_BreastCancerInstance_to_features(dataSet);
    dataLabels = reverse_BreastCancerInstance_to_labels(dataSet);

    //���ֲ��Լ���ѵ����
    vector<vector<double>> training_Features;
    vector<vector<double>> testing_Features;
    vector<double> training_Labels;
    vector<double> testing_Labels;

    size_t trainingSize = BATCH_SIZE * 3;
    
    training_Features.assign(dataFeatures.begin(), dataFeatures.begin() + trainingSize);
    testing_Features.assign(dataFeatures.begin() + trainingSize, dataFeatures.end());
    training_Labels.assign(dataLabels.begin(), dataLabels.begin() + trainingSize);
    testing_Labels.assign(dataLabels.begin() + trainingSize, dataLabels.end());

    //����ѵ������������
    size_t additionalCopies = num_iters - 1;

    if(additionalCopies > 0){
    
        //����һ��ѵ�����ĸ���
        vector<vector<double>> trainFeaturesCopy(training_Features);
        vector<double> trainLabelsCopy(training_Labels);

        for(size_t i = 0; i < additionalCopies; i++){
            //����ѵ���������ĸ���
            vector<vector<double>> temp1(trainFeaturesCopy);
            vector<double> temp2(trainLabelsCopy);

            // ������ͬ�������������
            random_device rd;
            mt19937 gen(rd());

            // ���� temp1 �Ĵ�С�����������
            vector<int> indices(temp1.size());
            iota(indices.begin(), indices.end(), 0);
            shuffle(indices.begin(), indices.end(), gen);

            // ʹ����ͬ������˳��� temp1 �� temp2 ��������
            vector<vector<double>> shuffled_temp1(temp1.size());
            vector<double> shuffled_temp2(temp2.size());
            for (int i = 0; i < temp1.size(); ++i) {
                shuffled_temp1[i] = temp1[indices[i]];
                shuffled_temp2[i] = temp2[indices[i]];
            }

            // �����ź��������ӵ�ԭ����ѵ��������
            training_Features.insert(training_Features.end(), shuffled_temp1.begin(), shuffled_temp1.end());
            training_Labels.insert(training_Labels.end(), shuffled_temp2.begin(), shuffled_temp2.end());

        }
    }

    //ѵ��
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

    //����
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