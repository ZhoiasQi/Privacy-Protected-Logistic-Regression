#include "test_logistic_regression.hpp"

void TestLogisticRegression::getW(ColVectorXi64 w){
    this->w = w;
}

void TestLogisticRegression::test_model(){

    for(int i = 0; i < t; i++){
        int indexLo = (i * n) % n;  
        online->test_model(i, indexLo);
    }

    if (party == ALICE){
        send<ColVectorXi64>(io, online->wi);
        
        cout << "Alice has already sent prediction_i to Carol" << endl;
    }
    else{
        recv<ColVectorXi64>(io, w);

        cout << "Carol has already got prediction_i from Alice" << endl;
    }

    if(party == CAROL){
        prediction += online->wi;

        int n_ = prediction.size();

        int num_correct = 0;

        for (int i = 0; i < n_; i++){
            if(Y[i] == SCALING_FACTOR){
                if(prediction[i] >= (SCALING_FACTOR >> 1)){
                    num_correct++;
                }
            }
            else{
                if(prediction[i] < (SCALING_FACTOR >> 1)){
                    num_correct++;
                }
            }
        }

    double accuracy = num_correct/((double) n_);
    cout << "Accuracy on testing the trained model is " << accuracy * 100 << endl;

    }
}