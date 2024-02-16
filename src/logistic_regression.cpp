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
        
        cout << "Bob has already sent wi to Alice" << endl;
    }
    else{
        recv<ColVectorXi64>(io, w);

        cout << "Alice has already got wi from Bob" << endl;

        w = w + online->wi;

        descale<ColVectorXi64, ColVectorXd>(w, w_d);

    }

}

void LogisticRegression::test_model(RowMatrixXd& testing_data, ColVectorXd& testing_labels){
    ColVectorXd prediction;

    prediction = testing_data * w_d;

    cout << w_d << endl;

    int n_ = prediction.size();

    //cout << prediction << endl;

    for(int i = 0; i < n_; i++){
        double temp;
        // if(prediction[i] >= 1){
        //     temp = 1.0;
        // }
        // else if(prediction[i] >= -1 && prediction[i] < 1){
        //     temp = prediction[i] + 0.5;
        // }
        // else{
        //     temp = 0;
        // }
        temp = 1.0 / (1 + exp(-prediction[i]));
        prediction[i] = temp;
        
    }

    //cout << prediction << endl;

    int num_correct = 0;
    for (int i = 0; i < n_; i++){
        //cout << prediction[i] << endl;
        if(testing_labels[i] == 1){
            if(prediction[i] >= 0.5){
                num_correct++;
                // cout << 1 << endl;
            }
        }
        else{
            if(prediction[i] < 0.5){
                num_correct++;
                // cout << 0 << endl;
            }
        }
    }

    double accuracy = num_correct/((double) n_);
    cout << "Accuracy on testing the trained model is " << accuracy * 100 << endl;
}