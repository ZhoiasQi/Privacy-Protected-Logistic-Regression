#include "logistic_regression.hpp"
#include <cmath>

using namespace Eigen;
using Eigen::Matrix;
using namespace emp;
using namespace std;

void LogisticRegression::train_model(){
    for(int i = 0; i < t; i++){
        int indexLo = (i * BATCH_SIZE) % n;  
        online->train_batch(i, indexLo);
    }

    if (party == BOB){
        send<ColVectorXi64>(io, online->wi);
    }
    else{
        recv<ColVectorXi64>(io, w);
    }
        
    if (party == ALICE){
        send<ColVectorXi64>(io, online->wi);
    }
    else{
        recv<ColVectorXi64>(io, w);
    }

    w += online->wi;

    descale<ColVectorXi64, ColVectorXd>(w, w_d);

    cout << w << endl;
    cout << w_d << endl;
}

void LogisticRegression::test_model(RowMatrixXd& testing_data, ColVectorXd& testing_labels){
    ColVectorXd prediction;
    prediction = testing_data * w_d;

    int n_ = prediction.size();

    for(int i = 0; i < n_; i++){
        int temp = 1 / (1 + pow(M_E, -prediction[i]));
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